#!/usr/bin/env python3
"""Language-model campaign on the L40 (Ollama).

Phases (checkpointed):
  0 smoke, 1 hybrid B3-B5 (six seeds, primary), 1b asymmetric memory cells,
  2b LLM-only vs heuristic-only ablation, 7 reasoning-field pilot.

Safe to re-run: completed jobs are skipped. ``--workers`` runs jobs
concurrently against the same Ollama server.

Usage:
  python scripts/campaign_2day.py --out-dir runs/campaign --workers 4
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from scipy import stats as _scipy_stats
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from coevsec.core.env import load_dotenv  # noqa: E402

CFG_DIR = ROOT / "configs" / "campaign" / "generated"
OUT_DIR = ROOT / "runs" / "campaign"  # overridden by --out-dir

# Main evidence seeds (phase 1). Smaller sets for expensive phases.
SEEDS_MAIN = [520, 530, 540, 550, 560, 570]
SEEDS_ABLATION = [520, 540, 560]

MODEL_MAIN = "qwen2.5:14b"


@dataclass
class Job:
    job_id: str
    phase: str
    path: str
    note: str = ""
    priority: int = 50


@dataclass
class JobResult:
    job_id: str
    status: str
    started_at: str = ""
    finished_at: str = ""
    elapsed_s: float = 0.0
    run_dir: str | None = None
    summary: dict = field(default_factory=dict)
    error: str | None = None


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _agent(
    role: str,
    adaptation: str,
    policy_kind: str,
    model: str,
    *,
    stealth: float,
    token_budget: int,
    stall_limit: int = 1,
    force_legal_hints: bool = True,
) -> dict[str, Any]:
    policy: dict[str, Any] = {"kind": policy_kind, "stealth": stealth}
    if policy_kind == "hybrid":
        policy["params"] = {"stall_limit": stall_limit, "force_legal_hints": force_legal_hints}
    elif policy_kind == "llm":
        policy["params"] = {"stall_limit": stall_limit, "force_legal_hints": force_legal_hints}
    agent: dict[str, Any] = {
        "id_prefix": role,
        "role": role,
        "adaptation": adaptation,
        "policy": policy,
        "encoded_behaviours": [],
    }
    if policy_kind in {"llm", "hybrid"}:
        agent["llm"] = {
            "provider": "ollama",
            "model": model,
            "temperature": 0.25,
            "max_tokens": 192,
            "timeout_s": 180,
        }
        agent["token_budget"] = token_budget
    if policy_kind == "heuristic":
        if role == "attacker":
            agent["encoded_behaviours"] = [
                "reconnaissance", "privilege_escalation", "resource_targeting",
            ]
        else:
            agent["encoded_behaviours"] = ["adaptive_monitoring", "dynamic_isolation"]
    return agent


def _write_cfg(
    stem: str,
    *,
    seed: int,
    episodes: int,
    max_turns: int,
    att: dict[str, Any],
    deff: dict[str, Any],
    description: str,
    n_attackers: int = 1,
    n_defenders: int = 1,
    topology: str = "none",
    communication: str = "none",
) -> Path:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    path = CFG_DIR / f"{stem}.yaml"
    raw = {
        "name": stem,
        "description": description,
        "seed": seed,
        "episodes": episodes,
        "max_turns": max_turns,
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "environment": {
            "backend": "sim",
            "hosts": 3 if n_attackers == 1 else 4,
            "vulnerabilities_per_host": 2,
            "detection_base_rate": 0.15,
        },
        "interaction": {"topology": topology, "communication": communication},
        "agents": [
            {**att, "count": n_attackers},
            {**deff, "count": n_defenders},
        ],
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def _build_jobs() -> list[Job]:
    jobs: list[Job] = []

    # Phase 0 — smoke
    p = _write_cfg(
        "p0_smoke_hybrid",
        seed=601,
        episodes=4,
        max_turns=18,
        att=_agent("attacker", "persistent", "hybrid", MODEL_MAIN, stealth=0.3, token_budget=40000),
        deff=_agent("defender", "persistent", "heuristic", MODEL_MAIN, stealth=0.4, token_budget=40000),
        description="Fail-fast hybrid smoke on L40",
    )
    jobs.append(Job("p0_smoke_hybrid", "0_smoke", str(p), "hybrid smoke 4ep", 0))

    # Phase 1 — hybrid B3/B4/B5 multi-seed (main claim)
    adapt_map = {
        "b3_static": "static",
        "b4_episodic": "episodic",
        "b5_persistent": "persistent",
    }
    for label, adapt in adapt_map.items():
        for seed in SEEDS_MAIN:
            stem = f"p1_hybrid_{label}_s{seed}"
            p = _write_cfg(
                stem,
                seed=seed,
                episodes=30,
                max_turns=18,
                att=_agent("attacker", adapt, "hybrid", MODEL_MAIN, stealth=0.3, token_budget=80000),
                deff=_agent("defender", adapt, "hybrid", MODEL_MAIN, stealth=0.4, token_budget=80000),
                description=f"Hybrid {label} seed={seed} (campaign phase 1)",
            )
            jobs.append(Job(stem, "1_hybrid_ci", str(p), f"hybrid {label} s={seed}", 10))

    # Phase 1b — asymmetric memory (attacker hybrid / defender heuristic and vice versa)
    for label, adapt in adapt_map.items():
        for seed in SEEDS_MAIN:
            stem = f"p1b_hybatt_heurdef_{label}_s{seed}"
            p = _write_cfg(
                stem,
                seed=seed + 1,
                episodes=30,
                max_turns=18,
                att=_agent("attacker", adapt, "hybrid", MODEL_MAIN, stealth=0.3, token_budget=80000),
                deff=_agent("defender", "static", "heuristic", MODEL_MAIN, stealth=0.4, token_budget=80000),
                description=f"Asymmetric hybrid att {label} / heur def seed={seed}",
            )
            jobs.append(Job(stem, "1b_asymmetric", str(p), f"hybAtt/heurDef {label} s={seed}", 15))
            stem = f"p1b_heuratt_hybdef_{label}_s{seed}"
            p = _write_cfg(
                stem,
                seed=seed + 2,
                episodes=30,
                max_turns=18,
                att=_agent("attacker", "static", "heuristic", MODEL_MAIN, stealth=0.3, token_budget=80000),
                deff=_agent("defender", adapt, "hybrid", MODEL_MAIN, stealth=0.4, token_budget=80000),
                description=f"Asymmetric heur att / hybrid def {label} seed={seed}",
            )
            jobs.append(Job(stem, "1b_asymmetric", str(p), f"heurAtt/hybDef {label} s={seed}", 16))

    # Phase 2b — LLM-only vs gate-only (heuristic) on B4 episodic
    for seed in SEEDS_ABLATION:
        stem = f"p2b_llm_only_b4_s{seed}"
        p = _write_cfg(
            stem,
            seed=seed + 10,
            episodes=20,
            max_turns=18,
            att=_agent(
                "attacker", "episodic", "llm", MODEL_MAIN, stealth=0.3, token_budget=60000,
                stall_limit=99, force_legal_hints=False,
            ),
            deff=_agent(
                "defender", "episodic", "llm", MODEL_MAIN, stealth=0.4, token_budget=60000,
                stall_limit=99, force_legal_hints=False,
            ),
            description=f"LLM-only B4 seed={seed}",
        )
        jobs.append(Job(stem, "2b_ablation_modes", str(p), f"LLM-only B4 s={seed}", 12))
        stem = f"p2b_gate_only_b4_s{seed}"
        p = _write_cfg(
            stem,
            seed=seed + 11,
            episodes=20,
            max_turns=18,
            att=_agent("attacker", "episodic", "heuristic", MODEL_MAIN, stealth=0.3, token_budget=60000),
            deff=_agent("defender", "episodic", "heuristic", MODEL_MAIN, stealth=0.4, token_budget=60000),
            description=f"Gate-only (heuristic) B4 seed={seed}",
        )
        jobs.append(Job(stem, "2b_ablation_modes", str(p), f"gate-only B4 s={seed}", 12))

    # Phase 7: reasoning-field (chain-of-thought) pilot. Hybrid episodic
    # attacker vs heuristic episodic defender, with and without the schema's
    # leading "reasoning" field; the two arms share seeds (paired design).
    for seed in SEEDS_MAIN:
        for tag, use_cot in (("on", True), ("off", False)):
            stem = f"p7_cot_{tag}_s{seed}"
            att = _agent("attacker", "episodic", "hybrid", MODEL_MAIN, stealth=0.3, token_budget=80000)
            att["policy"]["params"]["use_cot"] = use_cot
            p = _write_cfg(
                stem,
                seed=seed + 3,
                episodes=30,
                max_turns=18,
                att=att,
                deff=_agent("defender", "episodic", "heuristic", MODEL_MAIN, stealth=0.4, token_budget=80000),
                description=f"Reasoning-field pilot ({tag}) seed={seed}",
            )
            jobs.append(Job(stem, "7_cot_pilot", str(p), f"CoT {tag} s={seed}", 11))

    jobs.sort(key=lambda j: (j.priority, j.job_id))
    return jobs


def _models_in_config(path: str) -> list[str]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    models: list[str] = []
    for agent in data.get("agents", []):
        policy = (agent or {}).get("policy") or {}
        llm = (agent or {}).get("llm") or {}
        if policy.get("kind") in {"llm", "hybrid"} and llm.get("provider", "ollama") == "ollama":
            m = llm.get("model") or os.environ.get("COEVSEC_OLLAMA_MODEL")
            if m:
                models.append(str(m))
    return sorted(set(models))


def _wait_for_ollama_model(model: str, timeout_s: float = 7200.0, log=print) -> None:
    import requests

    base = os.environ.get("COEVSEC_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base}/api/tags", timeout=15)
            resp.raise_for_status()
            names = {m.get("name") for m in resp.json().get("models", [])}
            if model in names or f"{model}:latest" in names:
                log(f"[ollama] model ready: {model}")
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        log(f"[ollama] waiting for model {model} …")
        time.sleep(20)
    raise TimeoutError(f"model {model} not available on {base} within {timeout_s}s ({last_err})")


def _mean_ci(xs: list[float]) -> dict[str, float]:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci95": float("nan")}
    mean = sum(xs) / n
    if n == 1:
        return {"n": 1, "mean": mean, "ci95": 0.0}
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    se = (var ** 0.5) / (n ** 0.5)
    tcrit = float(_scipy_stats.t.ppf(0.975, df=n - 1))
    return {"n": float(n), "mean": mean, "ci95": tcrit * se}


class Campaign:
    def __init__(
        self, *, phases: set[str] | None = None, force: bool = False, workers: int = 1,
    ) -> None:
        self.workers = max(1, workers)
        self._lock = threading.Lock()
        load_dotenv(ROOT / ".env")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.state_path = OUT_DIR / "suite_state.json"
        self.log_path = OUT_DIR / "suite.log"
        self.state = self._load_state()
        self.force = force
        all_jobs = _build_jobs()
        if phases:
            self.jobs = [j for j in all_jobs if j.phase in phases]
        else:
            self.jobs = all_jobs

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"results": {}, "started_at": None, "finished_at": None}

    def _save_state(self) -> None:
        with self._lock:
            self.state_path.write_text(json.dumps(self.state, indent=2, default=str))

    def log(self, msg: str) -> None:
        line = f"[{_utc()}] {msg}"
        with self._lock:
            print(line, flush=True)
            with self.log_path.open("a") as fh:
                fh.write(line + "\n")

    def notify(self, msg: str) -> None:
        self.log(msg)

    def _already_ok(self, job_id: str) -> bool:
        if self.force:
            return False
        r = self.state.get("results", {}).get(job_id)
        return bool(r and r.get("status") == "ok")

    def run_job(self, job: Job) -> JobResult:
        if self._already_ok(job.job_id):
            prev = self.state["results"][job.job_id]
            self.log(f"SKIP {job.job_id} → {prev.get('run_dir')}")
            return JobResult(
                job_id=job.job_id, status="skip",
                run_dir=prev.get("run_dir"), summary=prev.get("summary", {}),
            )

        from coevsec.analysis.engine import analyze_run
        from coevsec.experiments import run_experiment

        started = time.time()
        result = JobResult(job_id=job.job_id, status="fail", started_at=_utc())
        try:
            self.notify(f"START {job.job_id}\nphase: {job.phase}\n{job.note}")
            for model in _models_in_config(job.path):
                _wait_for_ollama_model(model, log=self.log)
            agg = run_experiment(job.path)
            run_dir = agg.get("run_dir")
            if run_dir:
                report = analyze_run(run_dir)
                agg["analysis_classification"] = (
                    report.get("strategy_dynamics", {}) or {}
                ).get("classification")
                emergent = (report.get("emergence", {}) or {}).get("candidate_emergent", {})
                agg["top_emergent"] = list(emergent.keys())[:5]
            result.status = "ok"
            result.run_dir = run_dir
            result.summary = {
                k: agg.get(k)
                for k in (
                    "name", "episodes", "attack_success_rate", "detection_rate",
                    "coevolutionary_pressure", "behavioral_novelty", "esr",
                    "coalition_rate", "analysis_classification", "top_emergent",
                    "hybrid_gate_stats", "run_dir",
                )
                if k in agg or agg.get(k) is not None
            }
            result.elapsed_s = time.time() - started
            result.finished_at = _utc()
            self.notify(
                f"DONE {job.job_id} ({result.elapsed_s/60:.1f} min)\n"
                f"ASR={result.summary.get('attack_success_rate', float('nan'))}\n"
                f"DR={result.summary.get('detection_rate', float('nan'))}\n"
                f"CEP={result.summary.get('coevolutionary_pressure', float('nan'))}"
            )
        except Exception as exc:  # noqa: BLE001
            result.status = "fail"
            result.error = f"{type(exc).__name__}: {exc}"
            result.elapsed_s = time.time() - started
            result.finished_at = _utc()
            self.log(traceback.format_exc())
            self.notify(
                f"FAIL {job.job_id} after {result.elapsed_s/60:.1f} min\n"
                f"{result.error}\n(continuing)"
            )
        with self._lock:
            self.state["results"][job.job_id] = asdict(result)
        self._save_state()
        return result

    def run(self) -> int:
        self.state["started_at"] = self.state.get("started_at") or _utc()
        self._save_state()

        total = len(self.jobs)
        self.notify(
            "coevsec LLM campaign STARTED\n"
            f"jobs: {total}\n"
            f"ollama: {os.environ.get('COEVSEC_OLLAMA_BASE_URL', '?')}\n"
            f"main model: {MODEL_MAIN}\n"
            f"log: {self.log_path}"
        )
        ok = fail = skip = 0
        # The smoke job runs first on its own so a broken setup fails fast.
        smoke = [j for j in self.jobs if j.phase == "0_smoke"]
        rest = [j for j in self.jobs if j.phase != "0_smoke"]
        results = [self.run_job(j) for j in smoke]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results += list(pool.map(self.run_job, rest))
        for r in results:
            if r.status == "ok":
                ok += 1
            elif r.status == "skip":
                skip += 1
            else:
                fail += 1

        self.state["finished_at"] = _utc()
        self._save_state()
        report = self._write_final_report()
        self.notify(
            "coevsec LLM campaign FINISHED\n"
            f"ok={ok} skip={skip} fail={fail} / {total}\n"
            f"started: {self.state['started_at']}\n"
            f"finished: {self.state['finished_at']}\n"
            f"report: {report}"
        )
        return 0 if fail == 0 else 1

    def _write_final_report(self) -> Path:
        rows = []
        for jid, r in self.state.get("results", {}).items():
            if r.get("status") != "ok":
                continue
            s = r.get("summary") or {}
            if "attack_success_rate" not in s:
                continue
            rows.append({"job_id": jid, "phase": jid.split("_")[0], **s})

        # Aggregate CI for phase-1 hybrid cells by adaptation label
        groups: dict[str, dict[str, list[float]]] = {}
        for row in rows:
            jid = row["job_id"]
            if not jid.startswith("p1_hybrid_"):
                continue
            # p1_hybrid_b3_static_s520 → b3_static
            parts = jid.split("_")
            label = "_".join(parts[2:4]) if len(parts) >= 5 else "unknown"
            g = groups.setdefault(label, {"asr": [], "dr": [], "cep": [], "gate_frac": []})
            g["asr"].append(float(row["attack_success_rate"]))
            g["dr"].append(float(row["detection_rate"]))
            g["cep"].append(float(row.get("coevolutionary_pressure") or 0.0))
            gate = (row.get("hybrid_gate_stats") or {})
            if gate.get("attacker_gate_frac") is not None:
                g["gate_frac"].append(float(gate["attacker_gate_frac"]))

        ci_table = {
            label: {
                "ASR": _mean_ci(vals["asr"]),
                "DR": _mean_ci(vals["dr"]),
                "CEP": _mean_ci(vals["cep"]),
                "gate_frac": _mean_ci(vals["gate_frac"]),
            }
            for label, vals in sorted(groups.items())
        }

        gate_table = {
            label: stats["gate_frac"]
            for label, stats in ci_table.items()
            if stats.get("gate_frac", {}).get("n", 0)
        }

        payload = {
            "started_at": self.state.get("started_at"),
            "finished_at": self.state.get("finished_at"),
            "n_results": len(self.state.get("results", {})),
            "phase1_hybrid_ci": ci_table,
            "phase1_gate_frac": gate_table,
            "metric_rows": rows,
            "raw": self.state.get("results"),
        }
        path = OUT_DIR / "final_report.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        v2 = OUT_DIR / "final_report_v2.json"
        v2.write_text(json.dumps(payload, indent=2, default=str))

        md = OUT_DIR / "final_report.md"
        lines = [
            "# Two-day L40 campaign report",
            f"- started: {payload['started_at']}",
            f"- finished: {payload['finished_at']}",
            "",
            "## Phase 1 hybrid mean±CI95",
            "",
            "| cell | ASR mean±CI95 | DR mean±CI95 | CEP mean±CI95 |",
            "|---|---:|---:|---:|",
        ]
        for label, stats in ci_table.items():
            a, d, c = stats["ASR"], stats["DR"], stats["CEP"]
            lines.append(
                f"| {label} | {a['mean']:.3f}±{a['ci95']:.3f} | "
                f"{d['mean']:.3f}±{d['ci95']:.3f} | "
                f"{c['mean']:.3f}±{c['ci95']:.3f} |"
            )
        lines += [
            "",
            "## All successful jobs",
            "",
            "| job | ASR | DR | CEP | novelty | class |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row.get('job_id')} | "
                f"{float(row.get('attack_success_rate', float('nan'))):.3f} | "
                f"{float(row.get('detection_rate', float('nan'))):.3f} | "
                f"{float(row.get('coevolutionary_pressure') or float('nan')):.3f} | "
                f"{float(row.get('behavioral_novelty') or float('nan')):.3f} | "
                f"{row.get('analysis_classification', '')} |"
            )
        md.write_text("\n".join(lines) + "\n")
        return path


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        action="append",
        default=[],
        help="Run only these phases (e.g. 1_hybrid_ci). Repeatable.",
    )
    p.add_argument("--force", action="store_true", help="Re-run jobs even if previously ok")
    p.add_argument("--out-dir", default="runs/campaign", help="Output directory (relative to repo)")
    p.add_argument("--workers", type=int, default=1, help="Concurrent jobs against Ollama")
    args = p.parse_args()
    global OUT_DIR
    OUT_DIR = ROOT / args.out_dir
    os.chdir(ROOT)  # generated configs use repo-relative output paths
    phases = set(args.phase) if args.phase else None
    return Campaign(phases=phases, force=args.force, workers=args.workers).run()


if __name__ == "__main__":
    raise SystemExit(main())
