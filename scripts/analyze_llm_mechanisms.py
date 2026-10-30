#!/usr/bin/env python3
"""Behavioural mechanisms behind the 2x2 language-model effects.

The main analysis reports outcomes (ASR, DR, ...) and the controller gate
fraction. This script measures, directly from the step trajectories, the
behaviours that the results section attributes to persistent memory, so that
the mechanistic reading is supported by measurement rather than assertion:

  - defender: how early and how often it watches (monitor or investigate) the
    service host that recurs across episodes;
  - attacker: how much it paces (wait fraction) and how many deceptive actions
    it takes per episode.

For each behaviour it reports the seed-paired defender-memory effect (D1 vs
D0, averaged over the attacker's two levels) and attacker-memory effect (A1 vs
A0), the same seed-as-unit contrasts used for the outcomes. Reads the campaign
state, writes ``data/paper_runs/llm_mechanisms.json`` and a Markdown summary.

Usage:
  python scripts/analyze_llm_mechanisms.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_llm2x2 import _bh, _paired  # noqa: E402

STATE = ROOT / "runs" / "review" / "llm2x2" / "state.json"
OUT = ROOT / "data" / "paper_runs" / "llm_mechanisms.json"
CELLS = ["A0D0", "A0D1", "A1D0", "A1D1"]
WATCH = {"monitor", "investigate"}


def _run_mechanisms(run_dir: str) -> dict:
    run = Path(run_dir)
    eps = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()
           if line.strip()]
    # The host that recurs for this run: the modal service foothold host.
    hosts = [e.get("service_foothold_host") for e in eps if e.get("service_foothold_host")]
    recur = Counter(hosts).most_common(1)[0][0] if hosts else None

    steps: dict[int, list] = {}
    for line in (run / "trajectories.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        steps.setdefault(s["episode"], []).append(s)

    first_turns, watched_flags = [], []
    att_wait, att_total = 0, 0
    for ss in steps.values():
        ss.sort(key=lambda x: x.get("turn", 0))
        watch_turn = None
        for s in ss:
            role = s.get("role") or s.get("agent")
            if role == "defender" and s.get("action") in WATCH and s.get("target") == recur:
                watch_turn = s.get("turn")
                break
        if recur is not None:
            watched_flags.append(1.0 if watch_turn is not None else 0.0)
            if watch_turn is not None:
                first_turns.append(float(watch_turn))
        for s in ss:
            if (s.get("role") or s.get("agent")) == "attacker" and s.get("reason") != "curriculum: initial recon":
                att_total += 1
                if s.get("action") == "wait":
                    att_wait += 1

    return {
        # Lower turn = the defender watches the recurring host earlier.
        "def_watch_turn": (sum(first_turns) / len(first_turns)) if first_turns else float("nan"),
        # Fraction of episodes in which the recurring host is watched at all.
        "def_watch_frac": (sum(watched_flags) / len(watched_flags)) if watched_flags else float("nan"),
        "att_wait_frac": (att_wait / att_total) if att_total else float("nan"),
        "att_deceptive": sum(e.get("deceptive_actions", 0) for e in eps) / len(eps),
    }


def _cells(state: dict, regime: str) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {c: {} for c in CELLS}
    for name, r in state.get("runs", {}).items():
        if r.get("status") != "ok" or r.get("regime") != regime:
            continue
        seed = int(name.rsplit("_s", 1)[1])
        out[r["cell"]][seed] = _run_mechanisms(r["run_dir"])
    return out


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return (sum(xs) / len(xs)) if xs else float("nan")


def _effect(cells: dict[str, dict[int, dict]], key: str, role: str) -> dict:
    seeds = sorted(set.intersection(*(set(cells[c]) for c in CELLS)))
    hi_cells = ("A1D0", "A1D1") if role == "attacker" else ("A0D1", "A1D1")
    lo_cells = ("A0D0", "A0D1") if role == "attacker" else ("A0D0", "A1D0")

    def avg(cs, s):
        vals = [cells[c][s][key] for c in cs if not math.isnan(cells[c][s][key])]
        return sum(vals) / len(vals) if vals else float("nan")

    hi = [avg(hi_cells, s) for s in seeds]
    lo = [avg(lo_cells, s) for s in seeds]
    pairs = [(h, l) for h, l in zip(hi, lo) if not (math.isnan(h) or math.isnan(l))]
    if len(pairs) < 2:
        return {"n": len(pairs), "diff": float("nan"), "p": float("nan")}
    return _paired([h for h, _ in pairs], [l for _, l in pairs])


def main() -> int:
    if not STATE.exists():
        print(f"no campaign state at {STATE}", file=sys.stderr)
        return 1
    state = json.loads(STATE.read_text())
    out: dict = {"cells": {}, "effects": {}}
    lines = ["# 2x2 behavioural mechanisms", ""]
    metrics = [("def_watch_turn", "defender"), ("def_watch_frac", "defender"),
               ("att_wait_frac", "attacker"), ("att_deceptive", "attacker")]
    for regime in ("recurring", "iid"):
        cells = _cells(state, regime)
        if not any(cells.values()):
            continue
        out["cells"][regime] = {
            c: {k: _mean([m[k] for m in cells[c].values()]) for k, _ in metrics}
            for c in CELLS if cells[c]}
        eff = {k: _effect(cells, k, role) for k, role in metrics}
        ps = [(k, eff[k]["p"]) for k, _ in metrics
              if eff[k].get("p") is not None and not math.isnan(eff[k]["p"])]
        for (k, _), q in zip(ps, _bh([p for _, p in ps])):
            eff[k]["p_bh"] = q
        out["effects"][regime] = eff

        lines.append(f"## {regime}")
        lines.append("| cell | def_watch_turn | def_watch_frac | att_wait_frac | att_deceptive |")
        lines.append("|---|--:|--:|--:|--:|")
        for c in CELLS:
            if not cells[c]:
                continue
            cc = out["cells"][regime][c]
            lines.append(f"| {c} | {cc['def_watch_turn']:.2f} | {cc['def_watch_frac']:.2f} "
                         f"| {cc['att_wait_frac']:.3f} | {cc['att_deceptive']:.2f} |")
        lines.append("")
        for k, role in metrics:
            e = eff[k]
            role_lbl = "defender memory" if role == "defender" else "attacker memory"

            def _fp(p):
                return "n/a" if p is None or (isinstance(p, float) and math.isnan(p)) \
                    else ("<.001" if p < 0.001 else f"{p:.3f}")
            ci = (f", 95% CI [{e['boot_lo']:+.3f}, {e['boot_hi']:+.3f}]" if "boot_lo" in e else "")
            lines.append(f"- {k} ({role_lbl}): Delta={e['diff']:+.3f} "
                         f"(n={e['n']}, p={_fp(e.get('p'))}, p_BH={_fp(e.get('p_bh'))}{ci})")
        lines.append("")

    OUT.write_text(json.dumps(out, indent=2))
    (ROOT / "data" / "paper_runs" / "llm_mechanisms.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
