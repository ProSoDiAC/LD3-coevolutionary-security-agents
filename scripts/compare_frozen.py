#!/usr/bin/env python3
"""Compare the frozen pre-review results with the regenerated ones.

Reads the frozen bundle (``data/frozen/2026-09-15_pre-review``) and the
current data (``data/paper_runs``) and writes
``data/paper_runs/frozen_comparison.json`` plus a Markdown table on stdout.
Frozen statistics are recomputed from the frozen per-seed values with the
same paired tests used for the current results.

Usage:
  python scripts/compare_frozen.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_paper_artifacts import ladder_series, paired  # noqa: E402

FROZEN = ROOT / "data" / "frozen" / "2026-09-15_pre-review"
CURRENT = ROOT / "data" / "paper_runs"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _ladder_rows(lad: dict) -> dict:
    out = {}
    for cell in ("static_static", "adaptive_static", "persistent_static", "static_adaptive"):
        out[f"DR {cell}"] = lad["cells"][cell]["dr"]["mean"]
    for hid, a, key in (("H1", "adaptive_static", "dr"), ("H1'", "persistent_static", "dr"),
                        ("H2", "static_adaptive", "dr"), ("C", "persistent_persistent", "cep")):
        t = paired(ladder_series(lad, a, key), ladder_series(lad, "static_static", key))
        out[f"{hid} diff"] = t["diff"]
        out[f"{hid} p"] = t["p"]
    return out


def _crossplay_range(xp: dict) -> dict:
    drs = [xp["results"][a][d]["dr_mean"] for a in xp["results"] for d in xp["results"][a]]
    return {"cross-play DR min": min(drs), "cross-play DR max": max(drs)}


def main() -> int:
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    rows: dict[str, dict] = {}

    def put(name: str, frozen: float | None, current: float | None) -> None:
        rows[name] = {"frozen": frozen, "regenerated": current}

    f_lad, c_lad = _load(FROZEN / "ladder_iso.json"), _load(CURRENT / "ladder_iso.json")
    fr, cr = _ladder_rows(f_lad), _ladder_rows(c_lad)
    for k in fr:
        put(k, fr[k], cr[k])
    fx, cx = _crossplay_range(_load(FROZEN / "crossplay.json")), _crossplay_range(_load(CURRENT / "crossplay.json"))
    for k in fx:
        put(k, fx[k], cx[k])
    f_abl, c_abl = _load(FROZEN / "detector_ablation.json"), _load(CURRENT / "detector_ablation.json")
    for cond in ("baseline", "no_stealth_pacing"):
        put(f"ablation {cond}: DR L1/L0", f_abl["cells"]["adaptive_static"][cond]["dr"]["mean"],
            c_abl["cells"]["adaptive_static"][cond]["dr"]["mean"])

    f_camp = _load(FROZEN / "campaign_v2.json")["phase1_hybrid_ci"]
    c_stats = _load(CURRENT / "paper_stats.json")
    for lab, name in (("b3_static", "B3"), ("b4_episodic", "B4"), ("b5_persistent", "B5")):
        put(f"ASR {name}", f_camp[lab]["ASR"]["mean"], c_stats["llm"]["phase1"][name]["asr"]["mean"])
        put(f"DR {name}", f_camp[lab]["DR"]["mean"], c_stats["llm"]["phase1"][name]["dr"]["mean"])
    cot_f = _load(FROZEN / "cot_pilot_summary.json")["gate_frac"]
    cot_c = next(t for t in c_stats["tests_all"] if t["id"] == "R1")
    put("CoT gate fraction diff (on - off)", cot_f["mean_diff"], cot_c["diff"])
    put("CoT gate fraction p", cot_f["p"], cot_c["p"])

    (CURRENT / "frozen_comparison.json").write_text(json.dumps(rows, indent=2))
    print("| Quantity | Frozen | Regenerated |\n|---|---:|---:|")
    for k, v in rows.items():
        f = "n/a" if v["frozen"] is None else f"{v['frozen']:.3f}"
        c = "n/a" if v["regenerated"] is None else f"{v['regenerated']:.3f}"
        print(f"| {k} | {f} | {c} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
