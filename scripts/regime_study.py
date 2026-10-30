#!/usr/bin/env python3
"""Heuristic adaptation ladder across environment regimes and arenas.

Crosses attacker and defender adaptation level (L0, L1, L2) with the
environment regime (iid, recurring, drift) and the detector arena (saturated,
headroom). For each cell it reports, over ``--seeds`` independent seeds, the
attack success rate, detection rate, engagement rate, detection rate given
engagement, and co-evolutionary pressure adjusted against the same-seed L0/L0
cell of the same arena and regime.

The design tests when adaptation helps: whether persistent memory lowers the
detection rate or raises attack success only when the environment carries
transferable structure (recurring), and whether it does so through evasion or
non-engagement.

Usage:
  python scripts/regime_study.py --seeds 100-290 --episodes 40 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coevsec.core.arenas import detector  # noqa: E402
from coevsec.core.config import ExperimentConfig  # noqa: E402
from coevsec.core.types import AdaptationLevel  # noqa: E402
from coevsec.experiments.controller import ExperimentController  # noqa: E402
from coevsec.metrics import detection_rate_given_engagement, engagement_rate  # noqa: E402

LEVELS = {"L0": AdaptationLevel.STATIC, "L1": AdaptationLevel.EPISODIC, "L2": AdaptationLevel.PERSISTENT}
CELLS = [f"{a}_{d}" for a in LEVELS for d in LEVELS]
ARENAS = ["saturated", "headroom"]
REGIMES = ["iid", "recurring", "drift"]


def _cfg(arena: str, regime: str, cell: str, seed: int, episodes: int) -> ExperimentConfig:
    a, d = cell.split("_")
    return ExperimentConfig.model_validate({
        "name": f"reg_{arena}_{regime}_{cell}_s{seed}",
        "seed": seed, "episodes": episodes, "max_turns": 20,
        "output_dir": f"runs/regime/{arena}/{regime}",
        "environment": {
            "hosts": 3, "vulnerabilities_per_host": 2, "detector": detector(arena),
            "regime": regime, "n_profiles": 4, "drift_at": 0.5,
        },
        "interaction": {"topology": "none", "communication": "none"},
        "agents": [
            {"id_prefix": "attacker", "role": "attacker", "adaptation": LEVELS[a].value,
             "policy": {"kind": "heuristic", "stealth": 0.3}},
            {"id_prefix": "defender", "role": "defender", "adaptation": LEVELS[d].value,
             "policy": {"kind": "heuristic", "stealth": 0.4}},
        ],
    })


def _run_one(task: tuple[str, str, str, int, int]) -> dict:
    arena, regime, cell, seed, episodes = task
    agg = ExperimentController(_cfg(arena, regime, cell, seed, episodes)).run()
    eps = [json.loads(line) for line in (Path(agg["run_dir"]) / "episodes.jsonl").read_text().splitlines()
           if line.strip()]
    dge = detection_rate_given_engagement(eps)
    return {
        "arena": arena, "regime": regime, "cell": cell, "seed": seed,
        "asr": agg["attack_success_rate"], "dr": agg["detection_rate"],
        "er": engagement_rate(eps),
        "dr_engaged": None if dge != dge else dge,
        "cep": agg["coevolutionary_pressure"],
        "run_dir": agg["run_dir"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="100-290", help="range a-b (step 10) or comma list")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="runs/review/regime_study.json")
    args = p.parse_args()
    if "-" in args.seeds:
        lo, hi = (int(x) for x in args.seeds.split("-"))
        seeds = list(range(lo, hi + 1, 10))
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    tasks = [(ar, rg, c, s, args.episodes)
             for ar in ARENAS for rg in REGIMES for c in CELLS for s in seeds]
    print(f"{len(tasks)} runs ({len(ARENAS)} arenas x {len(REGIMES)} regimes x "
          f"{len(CELLS)} cells x {len(seeds)} seeds)", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_run_one, tasks, chunksize=4))

    report: dict = {"seeds": seeds, "episodes": args.episodes, "arenas": ARENAS,
                    "regimes": REGIMES, "cells": CELLS, "results": {}}
    for ar in ARENAS:
        report["results"][ar] = {}
        for rg in REGIMES:
            base = {r["seed"]: r["cep"] for r in rows
                    if r["arena"] == ar and r["regime"] == rg and r["cell"] == "L0_L0"}
            cells: dict = {}
            for c in CELLS:
                rs = sorted((r for r in rows if r["arena"] == ar and r["regime"] == rg
                             and r["cell"] == c), key=lambda r: r["seed"])
                for r in rs:
                    r["cep_adj"] = r["cep"] - base.get(r["seed"], 0.0)
                cells[c] = {"runs": [{k: r[k] for k in
                            ("seed", "asr", "dr", "er", "dr_engaged", "cep", "cep_adj", "run_dir")}
                            for r in rs]}
            report["results"][ar][rg] = cells

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
