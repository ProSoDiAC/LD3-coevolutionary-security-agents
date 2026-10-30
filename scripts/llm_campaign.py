#!/usr/bin/env python3
"""Factorial language-model campaign: attacker memory x defender memory.

Both roles are hybrid language-model agents (a model proposing tool calls under
a heuristic fallback). The design crosses attacker memory (L0 static vs L2
persistent) with defender memory (L0 vs L2), a 2x2 that separates the effect of
each side's memory and their interaction, instead of moving both together. The
cells run on the non-saturated ``headroom`` arena so outcomes are not at a
ceiling, in the ``recurring`` regime where past episodes carry information about
future ones, so persistent memory has transferable signal to exploit. An
optional ``iid`` control (``--control``) runs the two diagonal cells where the
structure is redrawn every episode, in which memory should not help.

Checkpointed and resumable: completed runs are skipped. Safe to run alongside
other GPU work with ``--workers 1``.

Usage:
  python scripts/llm_campaign.py --seeds 100-290 --episodes 30 --workers 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coevsec.core.arenas import detector  # noqa: E402
from coevsec.core.env import load_dotenv  # noqa: E402
from coevsec.experiments import run_experiment  # noqa: E402

MODEL = "qwen2.5:14b"
# cell -> (attacker level, defender level); 0 = L0 static, 2 = L2 persistent.
CELLS = {"A0D0": ("static", "static"), "A0D1": ("static", "persistent"),
         "A1D0": ("persistent", "static"), "A1D1": ("persistent", "persistent")}
CONTROL_CELLS = ["A0D0", "A1D1"]  # diagonal, run under iid as a regime control


def _agent(role: str, adaptation: str, stealth: float) -> dict:
    return {
        "id_prefix": role, "role": role, "adaptation": adaptation,
        "policy": {"kind": "hybrid", "stealth": stealth,
                   "params": {"stall_limit": 1, "force_legal_hints": True}},
        "llm": {"provider": "ollama", "model": MODEL, "temperature": 0.25,
                "max_tokens": 192, "timeout_s": 180},
        "token_budget": 80000,
    }


def _cfg_dict(cell: str, regime: str, seed: int, episodes: int, out_dir: str) -> dict:
    att, deff = CELLS[cell]
    return {
        "name": f"llm_{regime}_{cell}_s{seed}",
        "seed": seed, "episodes": episodes, "max_turns": 18, "output_dir": out_dir,
        "environment": {
            "backend": "sim", "hosts": 3, "vulnerabilities_per_host": 2,
            "detector": detector("headroom"), "regime": regime, "n_profiles": 4,
        },
        "interaction": {"topology": "none", "communication": "none"},
        "agents": [_agent("attacker", att, 0.3), _agent("defender", deff, 0.4)],
    }


def _jobs(seeds: list[int], episodes: int, out_dir: str, control: bool,
          iid_full: bool = False) -> list[dict]:
    jobs = []
    for cell in CELLS:
        for s in seeds:
            jobs.append({"cell": cell, "regime": "recurring",
                         "cfg": _cfg_dict(cell, "recurring", s, episodes, out_dir)})
    # The full iid 2x2 (all four cells) lets the memory-by-structure interaction
    # be estimated for the language-model agents, not only the diagonal control.
    iid_cells = CELLS if iid_full else (CONTROL_CELLS if control else [])
    for cell in iid_cells:
        for s in seeds:
            jobs.append({"cell": cell, "regime": "iid",
                         "cfg": _cfg_dict(cell, "iid", s, episodes, out_dir)})
    return jobs


def main() -> int:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="100-290")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--control", action="store_true", help="also run the iid diagonal control")
    p.add_argument("--iid-full", action="store_true",
                   help="run the full four-cell 2x2 under the iid regime")
    p.add_argument("--out-dir", default="runs/review/llm2x2")
    p.add_argument("--state", default="runs/review/llm2x2/state.json")
    args = p.parse_args()
    if "-" in args.seeds:
        lo, hi = (int(x) for x in args.seeds.split("-"))
        seeds = list(range(lo, hi + 1, 10))
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    state_path = ROOT / args.state
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.exists() else {"runs": {}}
    jobs = _jobs(seeds, args.episodes, args.out_dir, args.control, args.iid_full)
    print(f"{len(jobs)} jobs; {sum(1 for j in jobs if j['cfg']['name'] in state['runs'])} already done",
          flush=True)

    def run_job(job: dict) -> None:
        name = job["cfg"]["name"]
        if name in state["runs"] and state["runs"][name].get("status") == "ok":
            return
        cfg_path = state_path.parent / f"{name}.json"
        cfg_path.write_text(json.dumps(job["cfg"]))
        t0 = time.time()
        try:
            agg = run_experiment(str(cfg_path))
            state["runs"][name] = {
                "status": "ok", "cell": job["cell"], "regime": job["regime"],
                "run_dir": agg["run_dir"], "asr": agg["attack_success_rate"],
                "dr": agg["detection_rate"], "elapsed_min": round((time.time() - t0) / 60, 1),
            }
            print(f"ok {name} ASR={agg['attack_success_rate']:.3f} DR={agg['detection_rate']:.3f} "
                  f"({state['runs'][name]['elapsed_min']} min)", flush=True)
        except Exception as exc:  # noqa: BLE001
            state["runs"][name] = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
            print(f"FAIL {name}: {exc}\n{traceback.format_exc()}", flush=True)
        state_path.write_text(json.dumps(state, indent=2))

    if args.workers == 1:
        for j in jobs:
            run_job(j)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_job, jobs))
    ok = sum(1 for r in state["runs"].values() if r.get("status") == "ok")
    print(f"done: {ok}/{len(jobs)} ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
