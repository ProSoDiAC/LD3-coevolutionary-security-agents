#!/usr/bin/env python3
"""Multi-seed 1v1 ladder with mean ± 95% CI for ASR, DR, CEP and adjusted metrics.

Usage:
  python scripts/ladder_ci.py --seeds 100,110,120,130,140,150 --episodes 40
  python scripts/ladder_ci.py --cells persistent_static --seeds 100,110
  python scripts/ladder_ci.py --attacker-param abandon_failed_hosts=true  # ablation
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from scipy import stats as _scipy_stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from coevsec.core.env import load_dotenv  # noqa: E402

from coevsec.analysis.engine import analyze_run  # noqa: E402
from coevsec.core.config import ExperimentConfig  # noqa: E402
from coevsec.core.types import AdaptationLevel  # noqa: E402
from coevsec.experiments.controller import ExperimentController  # noqa: E402
from coevsec.metrics.coevolution import cep_baseline_adjusted  # noqa: E402

ATT_ENCODED = [
    "reconnaissance",
    "privilege_escalation",
    "resource_targeting",
    "stealth_pacing",
    "delayed_attack",
]
DEF_ENCODED = ["adaptive_monitoring", "dynamic_isolation"]

# name, att_adapt, att_kind, def_adapt, def_kind
# Isomorphic ladder: same heuristic policy class; only adaptation level varies.
# (The former "legacy" ladder used kind=rule_based for L0 roles; the factory
# maps rule_based to the heuristic policy at L0, so both ladders are the same
# experiment and produce identical episodes.)
CELLS_ISO = [
    ("static_static", AdaptationLevel.STATIC, "heuristic", AdaptationLevel.STATIC, "heuristic"),
    ("static_adaptive", AdaptationLevel.STATIC, "heuristic", AdaptationLevel.EPISODIC, "heuristic"),
    ("adaptive_static", AdaptationLevel.EPISODIC, "heuristic", AdaptationLevel.STATIC, "heuristic"),
    ("adaptive_adaptive", AdaptationLevel.EPISODIC, "heuristic", AdaptationLevel.EPISODIC, "heuristic"),
    ("persistent_persistent", AdaptationLevel.PERSISTENT, "heuristic", AdaptationLevel.PERSISTENT, "heuristic"),
    ("persistent_static", AdaptationLevel.PERSISTENT, "heuristic", AdaptationLevel.STATIC, "heuristic"),
    ("static_persistent", AdaptationLevel.STATIC, "heuristic", AdaptationLevel.PERSISTENT, "heuristic"),
    ("persistent_adaptive", AdaptationLevel.PERSISTENT, "heuristic", AdaptationLevel.EPISODIC, "heuristic"),
    ("adaptive_persistent", AdaptationLevel.EPISODIC, "heuristic", AdaptationLevel.PERSISTENT, "heuristic"),
]


def _cfg(
    name: str, seed: int, episodes: int, att_ad, att_kind, def_ad, def_kind,
    att_params: dict | None = None,
) -> ExperimentConfig:
    raw = {
        "name": f"ci_{name}_s{seed}",
        "description": f"CI ladder cell {name} seed={seed}",
        "seed": seed,
        "episodes": episodes,
        "max_turns": 20,
        "output_dir": "runs/ci",
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
                "policy": {"kind": att_kind, "stealth": 0.3, "params": dict(att_params or {})},
                "encoded_behaviours": ATT_ENCODED,
            },
            {
                "id_prefix": "defender",
                "role": "defender",
                "adaptation": def_ad.value,
                "policy": {"kind": def_kind, "stealth": 0.4},
                "encoded_behaviours": DEF_ENCODED,
            },
        ],
    }
    return ExperimentConfig.model_validate(raw)


def _mean_ci(xs: list[float]) -> dict:
    """Mean +/- 95% CI using the t-critical value (not a fixed z), since every
    caller aggregates a handful of seeds (n=6 in the primary studies) where
    the normal approximation understates the interval width."""
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    mean = sum(xs) / n
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95": 0.0}
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    std = math.sqrt(var)
    tcrit = float(_scipy_stats.t.ppf(0.975, df=n - 1))
    ci = tcrit * std / math.sqrt(n)
    return {"n": n, "mean": mean, "std": std, "ci95": ci, "lo": mean - ci, "hi": mean + ci}


def main() -> int:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="100,110,120,130,140,150")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--cells", default="", help="Comma-separated cell names (default: all)")
    p.add_argument("--iso", action="store_true", help="Accepted for compatibility (always isomorphic)")
    p.add_argument(
        "--attacker-param", action="append", default=[],
        help="Attacker policy param as key=value (bool/float), e.g. abandon_failed_hosts=true",
    )
    p.add_argument("--out", default="runs/ci/ladder_iso.json")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    wanted = {x.strip() for x in args.cells.split(",") if x.strip()}
    cells = [c for c in CELLS_ISO if not wanted or c[0] in wanted]
    att_params = {}
    for kv in args.attacker_param:
        k, v = kv.split("=", 1)
        att_params[k] = {"true": True, "false": False}.get(v.lower(), v)
        if isinstance(att_params[k], str):
            att_params[k] = float(v)

    per_cell: dict[str, list[dict]] = {c[0]: [] for c in cells}
    baseline_by_seed: dict[int, float] = {}
    out_path = Path(args.out)
    if "static_static" not in per_cell and out_path.exists():
        # This is a partial rerun (--cells) that doesn't include the baseline
        # cell: recover its per-seed CEP from the existing output instead of
        # silently treating a missing baseline as 0.0.
        try:
            prior = json.loads(out_path.read_text()).get("cells", {}).get("static_static", {})
            for r in prior.get("runs", []):
                baseline_by_seed[r["seed"]] = r["cep"]
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    for seed in seeds:
        for name, att_ad, att_kind, def_ad, def_kind in cells:
            print(f"running {name} seed={seed} …", flush=True)
            cfg = _cfg(name, seed, args.episodes, att_ad, att_kind, def_ad, def_kind, att_params)
            agg = ExperimentController(cfg).run()
            if agg.get("run_dir"):
                rep = analyze_run(agg["run_dir"])
                dyn = rep.get("strategy_dynamics", {}) or {}
                agg["dynamics_class"] = dyn.get("classification")
                agg["dynamics_class_full"] = dyn.get("classification_full_signature")
            if name == "static_static":
                baseline_by_seed[seed] = agg["coevolutionary_pressure"]
            if seed not in baseline_by_seed:
                raise RuntimeError(
                    f"no static_static CEP baseline for seed={seed}: include "
                    "static_static in --cells, or rerun against an --out file "
                    "that already has it, before computing cep_adj."
                )
            agg["cep_adj"] = cep_baseline_adjusted(
                agg["coevolutionary_pressure"], baseline_by_seed[seed]
            )
            per_cell[name].append(agg)

    merged_cells: dict = {}
    if wanted and out_path.exists():
        try:
            merged_cells = {
                k: v for k, v in json.loads(out_path.read_text()).get("cells", {}).items()
                if k not in per_cell
            }
        except (json.JSONDecodeError, OSError):
            merged_cells = {}

    report = {
        "seeds": seeds, "episodes": args.episodes, "isomorphic": True,
        "attacker_params": att_params, "cells": {},
    }
    lines = [
        "# Ladder multi-seed confidence intervals",
        f"seeds={seeds} episodes={args.episodes}",
        "",
        "| cell | ASR mean±CI95 | DR mean±CI95 | CEP mean±CI95 | CEP_adj mean±CI95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, rows in per_cell.items():
        asr = _mean_ci([r["attack_success_rate"] for r in rows])
        dr = _mean_ci([r["detection_rate"] for r in rows])
        cep = _mean_ci([r["coevolutionary_pressure"] for r in rows])
        cep_adj = _mean_ci([r.get("cep_adj", 0.0) for r in rows])
        report["cells"][name] = {
            "asr": asr,
            "dr": dr,
            "engagement": _mean_ci([r["engagement_rate"] for r in rows]),
            "cep": cep,
            "cep_adj": cep_adj,
            "runs": [
                {
                    "seed": seeds[i],
                    "asr": rows[i]["attack_success_rate"],
                    "dr": rows[i]["detection_rate"],
                    "engagement": rows[i].get("engagement_rate"),
                    "dr_engaged": rows[i].get("detection_rate_given_engagement"),
                    "cep": rows[i]["coevolutionary_pressure"],
                    "cep_adj": rows[i].get("cep_adj"),
                    "cep_attacker": rows[i].get("cep_attacker"),
                    "cep_defender": rows[i].get("cep_defender"),
                    "cross_lagged_def_to_att": rows[i].get("cross_lagged_def_to_att"),
                    "cross_lagged_att_to_def": rows[i].get("cross_lagged_att_to_def"),
                    "attack_success_count": rows[i].get("attack_success_count"),
                    "detection_count": rows[i].get("detection_count"),
                    "dynamics_class": rows[i].get("dynamics_class"),
                    "run_dir": rows[i].get("run_dir"),
                }
                for i in range(len(rows))
            ],
        }
        lines.append(
            f"| {name} | {asr['mean']:.3f}±{asr['ci95']:.3f} | "
            f"{dr['mean']:.3f}±{dr['ci95']:.3f} | {cep['mean']:.3f}±{cep['ci95']:.3f} | "
            f"{cep_adj['mean']:.3f}±{cep_adj['ci95']:.3f} |"
        )

    report["cells"].update(merged_cells)
    for name in sorted(report["cells"].keys()):
        if name in per_cell:
            continue
        cell = report["cells"][name]
        asr, dr, cep, cep_adj = cell["asr"], cell["dr"], cell["cep"], cell.get("cep_adj", {})
        lines.append(
            f"| {name} | {asr['mean']:.3f}±{asr['ci95']:.3f} | "
            f"{dr['mean']:.3f}±{dr['ci95']:.3f} | {cep['mean']:.3f}±{cep['ci95']:.3f} | "
            f"{cep_adj.get('mean', float('nan')):.3f}±{cep_adj.get('ci95', float('nan')):.3f} |"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    md = out.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n")
    print(md.read_text())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
