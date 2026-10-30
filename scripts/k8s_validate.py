#!/usr/bin/env python3
"""Run ladder cells on the Kubernetes backend and compare them with the simulator.

The Kubernetes backend keeps game and detection logic in the simulator and
mirrors every isolation onto a provisioned kind cluster (Helm-deployed hosts,
label-based NetworkPolicy isolation). This script therefore checks three
things: (i) that isolation is actually enforced by the cluster's network
policies (a TCP probe before, during and after isolating a host); (ii) the
share of isolation actions that were applied on the cluster; (iii) that
episode records are identical to those of the simulator backend on the same
seeds.

Requires kind, kubectl and helm, and a kind cluster named ``coevsec`` (created
on demand). On LXC hosts see the note in repro/REPRODUCE.md.

Usage:
  python scripts/k8s_validate.py --seeds 100,110,120,130,140,150 --episodes 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coevsec.core.config import ExperimentConfig  # noqa: E402
from coevsec.core.types import AdaptationLevel  # noqa: E402
from coevsec.environment.k8s.provisioner import KindProvisioner, tooling_available  # noqa: E402
from coevsec.experiments.controller import ExperimentController  # noqa: E402

CELLS = [
    ("static_static", AdaptationLevel.STATIC, AdaptationLevel.STATIC),
    ("adaptive_static", AdaptationLevel.EPISODIC, AdaptationLevel.STATIC),
    ("persistent_persistent", AdaptationLevel.PERSISTENT, AdaptationLevel.PERSISTENT),
]


def _cfg(cell: str, seed: int, backend: str, episodes: int, out_dir: str) -> ExperimentConfig:
    att, def_ = next(c[1:] for c in CELLS if c[0] == cell)
    raw = {
        "name": f"k8s_{cell}_{backend}_s{seed}",
        "seed": seed,
        "episodes": episodes,
        "max_turns": 20,
        "output_dir": out_dir,
        "environment": {
            "backend": backend,
            "hosts": 3,
            "vulnerabilities_per_host": 2,
            "params": {"provision": backend == "k8s"},
        },
        "interaction": {"topology": "none", "communication": "none"},
        "agents": [
            {"id_prefix": "attacker", "role": "attacker", "adaptation": att.value,
             "policy": {"kind": "heuristic", "stealth": 0.3}},
            {"id_prefix": "defender", "role": "defender", "adaptation": def_.value,
             "policy": {"kind": "heuristic", "stealth": 0.4}},
        ],
    }
    return ExperimentConfig.model_validate(raw)


def _episodes(run_dir: str) -> list[dict]:
    return [json.loads(line) for line in (Path(run_dir) / "episodes.jsonl").read_text().splitlines()
            if line.strip()]


def _isolations(run_dir: str) -> dict:
    """Isolations that succeeded in the game, split by target kind (decoys exist
    only in the simulator and have no pod), and how many reached the cluster."""
    out = {"hosts": 0, "hosts_applied": 0, "decoys": 0, "decoys_applied": 0}
    for line in (Path(run_dir) / "trajectories.jsonl").read_text().splitlines():
        s = json.loads(line)
        if s["action"] == "isolate" and s["success"]:
            kind = "decoys" if str(s["params"].get("host", "")).startswith("decoy") else "hosts"
            out[kind] += 1
            out[f"{kind}_applied"] += "'k8s_isolate_applied': True" in s["result"]
    return out


def _enforcement_probe() -> dict:
    prov = KindProvisioner()
    prov.create_cluster()
    prov.deploy(3)
    time.sleep(10)
    out = {"before": prov.reachable("gw", "svc_00"), "neighbour_before": prov.reachable("gw", "svc_01")}
    out["isolation_applied"] = prov.isolate_host("svc_00")
    time.sleep(5)
    out["during"] = prov.reachable("gw", "svc_00")
    out["outbound_during"] = prov.reachable("svc_00", "datastore")
    out["neighbour_during"] = prov.reachable("gw", "svc_01")
    prov._clear_isolation_policies()
    time.sleep(5)
    out["after"] = prov.reachable("gw", "svc_00")
    out["enforced"] = (out["before"] and not out["during"] and not out["outbound_during"]
                       and out["neighbour_during"] and out["after"])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="100,110,120,130,140,150")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--run-dir", default="runs/k8s")
    p.add_argument("--out", default="runs/review/k8s_validate.json")
    args = p.parse_args()
    ok, missing = tooling_available()
    if not ok:
        print(f"missing tools: {missing}", file=sys.stderr)
        return 1
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    report: dict = {"seeds": seeds, "episodes": args.episodes, "enforcement": _enforcement_probe(),
                    "cells": {}}
    print("enforcement:", report["enforcement"], flush=True)
    for cell, _, _ in CELLS:
        rows = []
        for seed in seeds:
            print(f"running {cell} seed={seed}", flush=True)
            sim = ExperimentController(_cfg(cell, seed, "sim", args.episodes, args.run_dir)).run()
            k8s = ExperimentController(_cfg(cell, seed, "k8s", args.episodes, args.run_dir)).run()
            iso = _isolations(k8s["run_dir"])
            rows.append({
                "seed": seed,
                "sim": {"asr": sim["attack_success_rate"], "dr": sim["detection_rate"]},
                "k8s": {"asr": k8s["attack_success_rate"], "dr": k8s["detection_rate"]},
                "episodes_identical": _episodes(sim["run_dir"]) == _episodes(k8s["run_dir"]),
                "isolations": iso,
                "sim_run_dir": sim["run_dir"],
                "k8s_run_dir": k8s["run_dir"],
            })
        report["cells"][cell] = {
            "runs": rows,
            "asr_sim": sum(r["sim"]["asr"] for r in rows) / len(rows),
            "asr_k8s": sum(r["k8s"]["asr"] for r in rows) / len(rows),
            "dr_sim": sum(r["sim"]["dr"] for r in rows) / len(rows),
            "dr_k8s": sum(r["k8s"]["dr"] for r in rows) / len(rows),
            "all_episodes_identical": all(r["episodes_identical"] for r in rows),
            "isolations": {k: sum(r["isolations"][k] for r in rows) for k in rows[0]["isolations"]},
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
