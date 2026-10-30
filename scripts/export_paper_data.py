#!/usr/bin/env python3
"""Export the experiment outputs behind the manuscript into ``data/paper_runs``.

Copies each study's aggregate JSON, bundles the per-episode records of every
run it references (gzipped JSONL, one line per episode) and writes a manifest
with the code revision, SHA-256 digests of every exported file and of each
run's config and episode log. Tables and figures are generated from this
directory only (``scripts/make_paper_artifacts.py``), never from ``runs/``.

Usage:
  python scripts/export_paper_data.py --runs runs/review --out data/paper_runs
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# artifact label -> file name under --runs
STUDIES = {
    "ladder_iso": "ladder_iso.json",
    "ladder_iso_20seeds": "ladder_iso_20seeds.json",
    "ladder_iso_abandon": "ladder_iso_abandon.json",
    "detector_ablation": "detector.json",
    "crossplay": "crossplay.json",
    "k8s_validate": "k8s_validate.json",
    # Checkpointed job state (also embedded as "raw" in final_report.json);
    # usable while the campaign is still running.
    "campaign": "campaign/suite_state.json",
    # The 2x2 language-model campaign: its state maps every run to its cell,
    # seed and regime, so the per-episode records of the headline result are
    # exported and become verifiable from the public artifact.
    "llm2x2": "llm2x2/state.json",
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(jsonl_path: Path) -> str:
    """SHA-256 over records re-serialised with sorted keys: independent of the
    key order in which the writer happened to emit them."""
    h = hashlib.sha256()
    for line in jsonl_path.read_text().splitlines():
        if line.strip():
            h.update((json.dumps(json.loads(line), sort_keys=True) + "\n").encode())
    return h.hexdigest()


def _action_counts(traj_path: Path) -> dict:
    """role -> origin (model or fallback) -> tool -> count."""
    out: dict = {}
    for line in traj_path.read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            origin = "fallback" if str(s.get("reason", "")).startswith("hybrid/heuristic") else "policy"
            by_tool = out.setdefault(s["role"], {}).setdefault(origin, {})
            by_tool[s["action"]] = by_tool.get(s["action"], 0) + 1
    return out


def _actions_digest(traj_path: Path) -> str:
    h = hashlib.sha256()
    for line in traj_path.read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            key = [s["episode"], s["turn"], s["agent"], s["action"], s.get("params", {})]
            h.update((json.dumps(key, sort_keys=True) + "\n").encode())
    return h.hexdigest()


def _code_digest() -> str:
    """SHA-256 over the tracked source files under src/ and scripts/ (path and
    content), independent of the git history."""
    files = _git("ls-files", "src", "scripts").splitlines()
    h = hashlib.sha256()
    for rel in sorted(files):
        path = ROOT / rel
        if path.is_file():
            h.update(rel.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _run_dirs(label: str, data: dict) -> list[str]:
    """Run directories referenced by a study's aggregate JSON."""
    out: list[str] = []
    if label == "campaign":
        for res in (data.get("raw") or data.get("results") or {}).values():
            # The smoke job only checks the setup and is not a reported result.
            if res.get("status") == "ok" and res.get("run_dir") and "p0_smoke" not in res["run_dir"]:
                out.append(res["run_dir"])
        return out
    if label == "detector_ablation":
        for cell in data.get("cells", {}).values():
            for cond in cell.values():
                out += [r["run_dir"] for r in cond.get("runs", []) if r.get("run_dir")]
        return out
    if label == "llm2x2":
        for r in data.get("runs", {}).values():
            if r.get("status") == "ok" and r.get("run_dir"):
                out.append(r["run_dir"])
        return out
    cells = data.get("cells", {})
    for cell in (cells.values() if isinstance(cells, dict) else []):
        if isinstance(cell, dict):
            out += [r["run_dir"] for r in cell.get("runs", []) if r.get("run_dir")]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs/review", help="Directory holding the study outputs")
    p.add_argument("--out", default="data/paper_runs")
    args = p.parse_args()

    runs_root = ROOT / args.runs
    out = ROOT / args.out
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # Content digest of the code that exports the data; the runs themselves
        # were produced by the code described in docs/REVIEW_REPORT.md.
        "code_sha256": _code_digest(),
        "code_dirty": bool(_git("status", "--porcelain", "--", "src", "scripts")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "artifacts": {},
        "runs": {},
    }

    for label, rel in STUDIES.items():
        src = runs_root / rel
        if not src.exists():
            print(f"missing {src}; skipped")
            continue
        data = json.loads(src.read_text())
        dest = out / f"{label}.json"
        dest.write_text(json.dumps(data, indent=2, default=str))
        manifest["artifacts"][label] = {"file": _rel(dest), "sha256": _sha256(dest)}

        run_dirs = _run_dirs(label, data)
        if not run_dirs:
            continue
        bundle = out / "episodes" / f"{label}.jsonl.gz"
        runs_meta = {}
        # mtime=0 keeps the gzip bytes (and hence the digest) reproducible.
        with gzip.GzipFile(bundle, "wb", mtime=0) as gz:
            for rd in sorted(set(run_dirs)):
                run = ROOT / rd
                ep_path, cfg_path = run / "episodes.jsonl", run / "config.json"
                if not ep_path.exists():
                    print(f"missing {ep_path}")
                    continue
                for line in ep_path.read_text().splitlines():
                    if line.strip():
                        rec = {"run": run.name, "episode": json.loads(line)}
                        gz.write((json.dumps(rec, sort_keys=True) + "\n").encode())
                traj_path = run / "trajectories.jsonl"
                runs_meta[run.name] = {
                    "episodes_sha256": _sha256(ep_path),
                    "episodes_canonical_sha256": _canonical_digest(ep_path),
                    # Digest of the executed (episode, turn, agent, tool, params)
                    # sequence: supports behavioural-identity checks without
                    # shipping the full step logs.
                    "actions_sha256": _actions_digest(traj_path) if traj_path.exists() else None,
                    "action_counts": _action_counts(traj_path) if traj_path.exists() else None,
                    "config_sha256": _sha256(cfg_path) if cfg_path.exists() else None,
                    "config": json.loads(cfg_path.read_text()) if cfg_path.exists() else None,
                }
        manifest["artifacts"][label]["episodes"] = {
            "file": _rel(bundle), "sha256": _sha256(bundle), "n_runs": len(runs_meta),
        }
        manifest["runs"][label] = runs_meta

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    n = sum(len(v) for v in manifest["runs"].values())
    print(f"wrote {out / 'manifest.json'} ({len(manifest['artifacts'])} artifacts, {n} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
