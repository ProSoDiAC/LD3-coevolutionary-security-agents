#!/usr/bin/env python3
"""Detector, pacing and engagement ablations on the attacker- and defender-side
contrasts of the isomorphic ladder (all roles use the heuristic policy class).

Usage:
  python scripts/detector_ablation.py --seeds 100,110,120,130,140,150 --episodes 40
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

from scipy import stats as _scipy_stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coevsec.core.config import ExperimentConfig  # noqa: E402
from coevsec.core.types import AdaptationLevel  # noqa: E402
from coevsec.experiments.controller import ExperimentController  # noqa: E402

ABLATIONS = [
    ("baseline", {}),
    ("decay_off", {"detector": {"suspicion_decay": 1.0}}),
    ("lambda_low", {"detector": {"suspicion_lambda": 0.06}}),
    ("lambda_high", {"detector": {"suspicion_lambda": 0.24}}),
    ("no_decay_foothold", {"detector": {"decay_while_foothold": False}}),
    ("lambda_very_low", {"detector": {"suspicion_lambda": 0.03}}),
    ("no_stealth_pacing", {"attacker_params": {"disable_stealth_pacing": True}}),
    ("abandon_failed_hosts", {"attacker_params": {"abandon_failed_hosts": True}}),
    ("no_investigate", {"defender_params": {"disable_investigate": True}}),
    ("no_decoy", {"defender_params": {"disable_decoy": True}}),
    ("no_investigate_no_decoy", {"defender_params": {"disable_investigate": True, "disable_decoy": True}}),
]

# cell -> (attacker level, defender level)
CELL_LEVELS = {
    "static_static": (AdaptationLevel.STATIC, AdaptationLevel.STATIC),
    "adaptive_static": (AdaptationLevel.EPISODIC, AdaptationLevel.STATIC),
    "persistent_static": (AdaptationLevel.PERSISTENT, AdaptationLevel.STATIC),
    "static_adaptive": (AdaptationLevel.STATIC, AdaptationLevel.EPISODIC),
    "static_persistent": (AdaptationLevel.STATIC, AdaptationLevel.PERSISTENT),
}


def _base_cell(cell: str, seed: int, episodes: int) -> dict:
    att_ad, def_ad = CELL_LEVELS[cell]
    return {
        "name": f"abl_{cell}_s{seed}",
        "seed": seed,
        "episodes": episodes,
        "max_turns": 20,
        "output_dir": "runs/ablation",
        "environment": {
            "backend": "sim",
            "hosts": 3,
            "vulnerabilities_per_host": 2,
            "detection_base_rate": 0.15,
        },
        "interaction": {"topology": "none", "communication": "none"},
        "agents": [
            {
                "id_prefix": "attacker",
                "role": "attacker",
                "adaptation": att_ad.value,
                "policy": {"kind": "heuristic", "stealth": 0.3, "params": {}},
            },
            {
                "id_prefix": "defender",
                "role": "defender",
                "adaptation": def_ad.value,
                "policy": {"kind": "heuristic", "stealth": 0.4},
            },
        ],
    }


def _apply_ablation(raw: dict, overrides: dict) -> ExperimentConfig:
    data = copy.deepcopy(raw)
    if "detector" in overrides:
        data["environment"].setdefault("detector", {})
        data["environment"]["detector"].update(overrides["detector"])
    if "defender_params" in overrides:
        data["agents"][1]["policy"].setdefault("params", {})
        data["agents"][1]["policy"]["params"].update(overrides["defender_params"])
    if "attacker_params" in overrides:
        data["agents"][0]["policy"].setdefault("params", {})
        data["agents"][0]["policy"]["params"].update(overrides["attacker_params"])
    return ExperimentConfig.model_validate(data)


def _mean_ci(xs: list[float]) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci95": float("nan")}
    mean = sum(xs) / n
    if n == 1:
        return {"n": 1, "mean": mean, "ci95": 0.0}
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    tcrit = float(_scipy_stats.t.ppf(0.975, df=n - 1))
    return {"n": n, "mean": mean, "std": std, "ci95": tcrit * std / math.sqrt(n)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="100,110,120,130,140,150")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--cells", default=",".join(CELL_LEVELS))
    p.add_argument("--out", default="runs/ablation/detector.json")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cells = [x.strip() for x in args.cells.split(",") if x.strip()]

    report: dict = {"seeds": seeds, "episodes": args.episodes, "cells": {}}
    for cell in cells:
        report["cells"][cell] = {}
        for abl_name, overrides in ABLATIONS:
            rows = []
            for seed in seeds:
                print(f"running {cell}/{abl_name} seed={seed}", flush=True)
                cfg = _apply_ablation(_base_cell(cell, seed, args.episodes), overrides)
                cfg.name = f"abl_{cell}_{abl_name}_s{seed}"
                agg = ExperimentController(cfg).run()
                rows.append(agg)
            report["cells"][cell][abl_name] = {
                "dr": _mean_ci([r["detection_rate"] for r in rows]),
                "asr": _mean_ci([r["attack_success_rate"] for r in rows]),
                "asr_undetected": _mean_ci([r.get("attack_success_undetected_rate", 0.0) for r in rows]),
                "engagement": _mean_ci([r["engagement_rate"] for r in rows]),
                "runs": [
                    {
                        "seed": seeds[i],
                        "dr": rows[i]["detection_rate"],
                        "asr": rows[i]["attack_success_rate"],
                        "engagement": rows[i]["engagement_rate"],
                        "dr_engaged": rows[i]["detection_rate_given_engagement"],
                        "run_dir": rows[i].get("run_dir"),
                    }
                    for i in range(len(rows))
                ],
            }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
