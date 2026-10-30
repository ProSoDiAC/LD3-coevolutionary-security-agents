#!/usr/bin/env python3
"""Analyse the 2x2 (attacker memory x defender memory) language-model campaign.

Reads the campaign state written by ``scripts/llm_campaign.py`` and the
per-episode records of each completed run, and reports for every cell the
attack success rate, detection rate, engagement rate, detection rate given
engagement and the corrected gate fraction (controller steps over model
decisions, with the forced initial reconnaissance excluded from the
denominator). It then estimates the three factorial effects, attacker memory,
defender memory and their interaction, as seed-paired differences with exact
sign-flip p-values where a difference is constant across seeds and paired t
otherwise, and contrasts the recurring cells with the iid control when present.

Runs against partial state, so it can be used while the campaign is still
filling in. Writes ``data/paper_runs/llm2x2_stats.json`` and a Markdown
summary on stdout.

Usage:
  python scripts/analyze_llm2x2.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_paper_artifacts import boot_ci  # noqa: E402
from scipy import stats  # noqa: E402

STATE = ROOT / "runs" / "review" / "llm2x2" / "state.json"
OUT = ROOT / "data" / "paper_runs" / "llm2x2_stats.json"
CELLS = ["A0D0", "A0D1", "A1D0", "A1D1"]


def _run_metrics(run_dir: str) -> dict:
    eps = [json.loads(line) for line in (Path(run_dir) / "episodes.jsonl").read_text().splitlines()
           if line.strip()]
    n = len(eps)
    eng = [e for e in eps if e.get("foothold_gained")]

    def _gate(role: str) -> float:
        origin: dict[str, float] = defaultdict(float)
        for e in eps:
            side = (e.get("hybrid_gate_stats") or {}).get(role, {})
            for k, v in (side.get("origin") or {}).items():
                origin[k] += v
        # Model decisions exclude the forced curriculum reconnaissance.
        model = sum(v for k, v in origin.items() if k != "curriculum")
        ctrl = sum(v for k, v in origin.items() if k.startswith("controller"))
        return (ctrl / model if model else float("nan")), dict(origin)

    gate_att, origin_att = _gate("attacker")
    gate_def, _ = _gate("defender")
    return {
        "asr": sum(bool(e.get("attack_success")) for e in eps) / n,
        "dr": sum(bool(e.get("detected")) for e in eps) / n,
        "er": len(eng) / n,
        "dr_engaged": (sum(bool(e.get("detected")) for e in eng) / len(eng)) if eng else float("nan"),
        # Gate fraction (controller share of model decisions) for each role.
        "gate_frac": gate_att,
        "gate_frac_def": gate_def,
        "origin": origin_att,
    }


def _cell_runs(state: dict, regime: str) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {c: {} for c in CELLS}
    for name, r in state.get("runs", {}).items():
        if r.get("status") != "ok" or r.get("regime") != regime:
            continue
        seed = int(name.rsplit("_s", 1)[1])
        out[r["cell"]][seed] = _run_metrics(r["run_dir"])
    return out


def _mean_ci(xs: list[float]) -> dict:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci95": float("nan")}
    m = sum(xs) / n
    if n < 2:
        return {"n": n, "mean": m, "ci95": 0.0}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return {"n": n, "mean": m, "ci95": float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)}


def _paired(a: list[float], b: list[float]) -> dict:
    n = len(a)
    if n < 2:
        return {"n": n, "diff": (a[0] - b[0]) if n else float("nan"), "p": float("nan")}
    diff = [x - y for x, y in zip(a, b)]
    m = sum(diff) / n
    if all(abs(x) < 1e-12 for x in diff):
        p, t = (1.0, 0.0) if abs(m) < 1e-12 else (2.0 / 2 ** n, math.inf)
    else:
        res = stats.ttest_rel(a, b)
        t, p = float(res.statistic), float(res.pvalue)
    lo, hi = boot_ci(diff)
    return {"n": n, "diff": m, "t": t, "p": p, "boot_lo": lo, "boot_hi": hi}


def _bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, preserving input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        prev = min(prev, pvals[i] * m / k)
        adj[i] = prev
    return adj


def _seed_series(cell: dict[int, dict], other: dict[int, dict], key: str) -> tuple[list, list]:
    seeds = sorted(set(cell) & set(other))
    return ([cell[s][key] for s in seeds], [other[s][key] for s in seeds])


def _factorial(cells: dict[str, dict[int, dict]], key: str) -> dict:
    # Seed is the unit of analysis: for each seed the main effect is the mean
    # over the other role's two levels, so each seed contributes one paired
    # difference (n = number of seeds), avoiding pseudo-replication. The
    # interaction is the per-seed difference-in-differences.
    out = {}
    seeds = sorted(set(cells["A0D0"]) & set(cells["A0D1"]) & set(cells["A1D0"]) & set(cells["A1D1"]))

    def val(c, s):
        return cells[c][s][key]

    att_hi = [(val("A1D0", s) + val("A1D1", s)) / 2 for s in seeds]
    att_lo = [(val("A0D0", s) + val("A0D1", s)) / 2 for s in seeds]
    out["attacker_memory"] = _paired(att_hi, att_lo)
    def_hi = [(val("A0D1", s) + val("A1D1", s)) / 2 for s in seeds]
    def_lo = [(val("A0D0", s) + val("A1D0", s)) / 2 for s in seeds]
    out["defender_memory"] = _paired(def_hi, def_lo)
    inter = [(val("A1D1", s) - val("A1D0", s)) - (val("A0D1", s) - val("A0D0", s)) for s in seeds]
    if len(inter) >= 2:
        m = sum(inter) / len(inter)
        res = stats.ttest_1samp(inter, 0.0)
        lo, hi = boot_ci(inter)
        out["interaction"] = {"n": len(inter), "diff": m, "p": float(res.pvalue),
                              "boot_lo": lo, "boot_hi": hi}
    else:
        out["interaction"] = {"n": len(inter), "diff": float("nan"), "p": float("nan")}
    return out


def main() -> int:
    if not STATE.exists():
        print(f"no campaign state at {STATE}", file=sys.stderr)
        return 1
    state = json.loads(STATE.read_text())
    out: dict = {"cells": {}, "effects": {}}
    lines = ["# 2x2 language-model campaign (attacker memory x defender memory)", ""]
    for regime in ("recurring", "iid"):
        cells = _cell_runs(state, regime)
        if not any(cells.values()):
            continue
        out["cells"][regime] = {
            c: {k: _mean_ci([m[k] for m in cells[c].values()])
                for k in ("asr", "dr", "er", "dr_engaged", "gate_frac", "gate_frac_def")}
            for c in CELLS if cells[c]}
        out["effects"][regime] = {k: _factorial(cells, k) for k in ("asr", "dr")}
        # Benjamini-Hochberg within the 2x2 family (both metrics, the two main
        # effects and the interaction), as declared in the methodology.
        fam = [(metric, eff)
               for metric in ("asr", "dr")
               for eff in ("attacker_memory", "defender_memory", "interaction")]
        raw = [out["effects"][regime][m][e].get("p") for m, e in fam]
        valid = [(i, p) for i, p in enumerate(raw)
                 if p is not None and not (isinstance(p, float) and math.isnan(p))]
        if valid:
            adj = _bh([p for _, p in valid])
            for (i, _), q in zip(valid, adj):
                metric, eff = fam[i]
                out["effects"][regime][metric][eff]["p_bh"] = q
        lines.append(f"## {regime}")
        lines.append("| cell | n | ASR | DR | ER | DR|E | gate |")
        lines.append("|---|--:|--:|--:|--:|--:|--:|")
        for c in CELLS:
            if not cells[c]:
                continue
            cc = out["cells"][regime][c]
            lines.append(f"| {c} | {cc['asr']['n']} | {cc['asr']['mean']:.3f} | {cc['dr']['mean']:.3f} "
                         f"| {cc['er']['mean']:.3f} | {cc['dr_engaged']['mean']:.3f} "
                         f"| {cc['gate_frac']['mean']:.3f} |")
        lines.append("")
        for metric in ("asr", "dr"):
            lines.append(f"### effect on {metric.upper()} ({regime})")
            for eff in ("attacker_memory", "defender_memory", "interaction"):
                e = out["effects"][regime][metric][eff]

                def _fp(p):
                    return "n/a" if p is None or (isinstance(p, float) and math.isnan(p)) \
                        else ("<.001" if p < 0.001 else f"{p:.3f}")
                ci = (f", 95% CI [{e['boot_lo']:+.3f}, {e['boot_hi']:+.3f}]"
                      if "boot_lo" in e else "")
                lines.append(f"- {eff}: Delta={e['diff']:+.3f} (n={e['n']}, "
                             f"p={_fp(e.get('p'))}, p_BH={_fp(e.get('p_bh'))}{ci})")
            lines.append("")
    OUT.write_text(json.dumps(out, indent=2))
    (ROOT / "data" / "paper_runs" / "llm2x2_stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
