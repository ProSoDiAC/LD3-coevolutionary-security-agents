#!/usr/bin/env python3
"""Statistics for the heuristic adaptation x regime x arena study.

Reads ``runs/review/regime_study.json`` and, for each arena and regime, tests
the memory contrasts (attacker L2 vs L0, defender L2 vs L0, symmetric L2/L2 vs
L0/L0) on the detection rate, attack success rate and engagement rate. Paired
differences that are constant across seeds use the exact two-sided sign-flip
permutation p-value; others use the paired t-test. Each contrast also reports
the engagement decomposition, how much of a detection-rate change is matched by
an engagement change rather than by evasion. Writes
``data/paper_runs/regime_stats.json`` and a Markdown summary on stdout.

Usage:
  python scripts/analyze_regime.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_paper_artifacts import bh, boot_ci  # noqa: E402
from scipy import stats  # noqa: E402

STUDY = ROOT / "runs" / "review" / "regime_study.json"
OUT = ROOT / "data" / "paper_runs" / "regime_stats.json"

CONTRASTS = [
    ("att_mem", "L2_L0", "L0_L0", "attacker L2 vs L0"),
    ("def_mem", "L0_L2", "L0_L0", "defender L2 vs L0"),
    ("both_mem", "L2_L2", "L0_L0", "L2/L2 vs L0/L0"),
]


def _series(cells: dict, cell: str, key: str) -> list[float]:
    return [r[key] for r in sorted(cells[cell]["runs"], key=lambda r: r["seed"])
            if r[key] is not None]


def _paired(a: list[float], b: list[float]) -> dict:
    n = len(a)
    diff = [x - y for x, y in zip(a, b)]
    m = sum(diff) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in diff) / (n - 1)) if n > 1 else 0.0
    if all(abs(x) < 1e-12 for x in diff):
        p = 1.0 if abs(m) < 1e-12 else 2.0 / 2 ** n
        t = 0.0 if abs(m) < 1e-12 else math.inf
    elif sd < 1e-12:
        p, t = 2.0 / 2 ** n, math.inf
    else:
        res = stats.ttest_rel(a, b)
        t, p = float(res.statistic), float(res.pvalue)
    lo, hi = boot_ci(diff)
    return {"n": n, "diff": m, "t": t, "p": p, "boot_lo": lo, "boot_hi": hi}


def main() -> int:
    d = json.loads(STUDY.read_text())
    out: dict = {"seeds": d["seeds"], "episodes": d["episodes"], "results": {}}
    lines = ["# Heuristic memory contrasts by arena and regime", "",
             "Delta is the paired mean difference (contrast cell minus baseline). "
             "p is exact sign-flip when the difference is constant across seeds, "
             "else paired t; p_BH adjusted within each arena x regime family.", ""]
    for ar in d["arenas"]:
        out["results"][ar] = {}
        for rg in d["regimes"]:
            cells = d["results"][ar][rg]
            tests = []
            for cid, cell, base, label in CONTRASTS:
                dr = _paired(_series(cells, cell, "dr"), _series(cells, base, "dr"))
                asr = _paired(_series(cells, cell, "asr"), _series(cells, base, "asr"))
                er = _paired(_series(cells, cell, "er"), _series(cells, base, "er"))
                # Engagement decomposition: fraction of the DR change matched by
                # the ER change (close to 1 => non-engagement, not evasion).
                frac = None if abs(dr["diff"]) < 1e-9 else round(er["diff"] / dr["diff"], 2)
                tests.append({"id": cid, "label": label, "dr": dr, "asr": asr, "er": er,
                              "dr_from_engagement": frac})
            qs = bh([tt["dr"]["p"] for tt in tests])
            for tt, q in zip(tests, qs):
                tt["dr"]["p_bh"] = q
            out["results"][ar][rg] = {
                "L0_L0_dr": sum(_series(cells, "L0_L0", "dr")) / len(d["seeds"]),
                "contrasts": tests,
            }
            lines.append(f"## {ar} / {rg} (L0/L0 DR = {out['results'][ar][rg]['L0_L0_dr']:.3f})")
            lines.append("| contrast | Delta DR | p | p_BH | Delta ASR | Delta ER | DR from engagement |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for tt in tests:
                pf = "<.001" if tt["dr"]["p"] < 0.001 else f"{tt['dr']['p']:.3f}"
                qf = "<.001" if tt["dr"]["p_bh"] < 0.001 else f"{tt['dr']['p_bh']:.3f}"
                frac = "n/a" if tt["dr_from_engagement"] is None else f"{tt['dr_from_engagement']:.2f}"
                lines.append(f"| {tt['label']} | {tt['dr']['diff']:+.3f} | {pf} | {qf} | "
                             f"{tt['asr']['diff']:+.3f} | {tt['er']['diff']:+.3f} | {frac} |")
            lines.append("")
    OUT.write_text(json.dumps(out, indent=2))
    md = "\n".join(lines) + "\n"
    (ROOT / "data" / "paper_runs" / "regime_stats.md").write_text(md)
    print(md)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
