#!/usr/bin/env python3
"""Cross-play matrix from persistent ladder checkpoints.

Usage:
  python scripts/crossplay_matrix.py --seeds 100,110,120 --checkpoint-ep 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coevsec.core.config import ExperimentConfig  # noqa: E402
from coevsec.core.types import AdaptationLevel  # noqa: E402
from coevsec.experiments.controller import ExperimentController  # noqa: E402
from coevsec.experiments.crossplay import run_with_checkpoints  # noqa: E402

SOURCE_CELLS = ["static_static", "adaptive_static", "persistent_persistent"]

# Native per-role adaptation level of each source cell (attacker, defender).
CELL_ADAPTATION = {
    "static_static": (AdaptationLevel.STATIC, AdaptationLevel.STATIC),
    "adaptive_static": (AdaptationLevel.EPISODIC, AdaptationLevel.STATIC),
    "persistent_persistent": (AdaptationLevel.PERSISTENT, AdaptationLevel.PERSISTENT),
}


def _iso_cfg(name: str, seed: int, att_ad: AdaptationLevel, def_ad: AdaptationLevel, episodes: int) -> ExperimentConfig:
    raw = {
        "name": f"xplay_{name}_s{seed}",
        "seed": seed,
        "episodes": episodes,
        "max_turns": 20,
        "output_dir": "runs/crossplay",
        "environment": {"backend": "sim", "hosts": 3, "vulnerabilities_per_host": 2},
        "interaction": {"topology": "none", "communication": "none"},
        "agents": [
            {
                "id_prefix": "attacker",
                "role": "attacker",
                "adaptation": att_ad.value,
                "policy": {"kind": "heuristic", "stealth": 0.3},
            },
            {
                "id_prefix": "defender",
                "role": "defender",
                "adaptation": def_ad.value,
                "policy": {"kind": "heuristic", "stealth": 0.4},
            },
        ],
    }
    return ExperimentConfig.model_validate(raw)


def _cell_cfg(cell: str, seed: int, episodes: int) -> ExperimentConfig:
    att, def_ = CELL_ADAPTATION[cell]
    return _iso_cfg(cell, seed, att, def_, episodes)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="100,110,120,130,140,150")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--checkpoint-ep", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument(
        "--eval-adaptation",
        choices=["native", "static"],
        default="native",
        help=(
            "'native' (default): evaluate each swapped-in checkpoint under its "
            "own cell's adaptation level per role (e.g. an attacker checkpoint "
            "from adaptive_static is replayed at EPISODIC), so within-episode "
            "stealth/vigilance escalation behaves as it would have in the "
            "checkpoint's cell of origin. 'static' forces L0 on both sides for "
            "every combination regardless of origin; this reproduces the "
            "original (pre-fix) behaviour, which disables that escalation "
            "mechanism entirely and produced a uniform null matrix."
        ),
    )
    p.add_argument("--out", default="runs/crossplay/matrix.json")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    ckpt_ep = args.checkpoint_ep

    checkpoints: dict[str, dict[int, dict]] = {}
    for cell in SOURCE_CELLS:
        checkpoints[cell] = {}
        for seed in seeds:
            print(f"collect checkpoints {cell} seed={seed}", flush=True)
            cfg = _cell_cfg(cell, seed, args.episodes)
            ctrl = ExperimentController(cfg, checkpoint_episodes=(ckpt_ep,))
            result = ctrl._run_config(seed, writer=None)
            ckpt = {}
            for c in result.checkpoints:
                if c.get("episode", -1) + 1 == ckpt_ep:
                    ckpt = {"attacker": c.get("attacker"), "defender": c.get("defender")}
            checkpoints[cell][seed] = ckpt

    matrix: dict = {
        "checkpoint_episode": ckpt_ep,
        "seeds": seeds,
        "eval_episodes": args.eval_episodes,
        "cells": SOURCE_CELLS,
        "eval_adaptation": args.eval_adaptation,
        "results": {},
    }
    # Each checkpoint is loaded once per evaluation run; the cross-episode
    # update keeps running between evaluation episodes, so an L2 checkpoint
    # continues to adapt. Under L0/L1 the carried parameters never change,
    # so those checkpoints equal the initial parameters.
    eval_cfg_cache: dict[tuple[str, str], ExperimentConfig] = {}

    def _eval_cfg_for(att_cell: str, def_cell: str) -> ExperimentConfig:
        key = (att_cell, def_cell)
        if key not in eval_cfg_cache:
            if args.eval_adaptation == "static":
                att_ad = def_ad = AdaptationLevel.STATIC
            else:
                att_ad = CELL_ADAPTATION[att_cell][0]
                def_ad = CELL_ADAPTATION[def_cell][1]
            eval_cfg_cache[key] = _iso_cfg(f"eval_{att_cell}_{def_cell}", seeds[0], att_ad, def_ad, 1)
        return eval_cfg_cache[key]

    for att_cell in SOURCE_CELLS:
        matrix["results"][att_cell] = {}
        for def_cell in SOURCE_CELLS:
            eval_cfg = _eval_cfg_for(att_cell, def_cell)
            drs, asrs, engs = [], [], []
            for seed in seeds:
                att_ck = checkpoints[att_cell][seed].get("attacker")
                def_ck = checkpoints[def_cell][seed].get("defender")
                r = run_with_checkpoints(
                    eval_cfg, seed + 9000, att_ck, def_ck, eval_episodes=args.eval_episodes,
                )
                drs.append(r["detection_rate"])
                engs.append(r["engagement_rate"])
                asrs.append(r["attack_success_rate"])
            matrix["results"][att_cell][def_cell] = {
                "dr_mean": sum(drs) / len(drs),
                "asr_mean": sum(asrs) / len(asrs),
                "dr_by_seed": drs,
                "asr_by_seed": asrs,
                "engagement_by_seed": engs,
            }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(matrix, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
