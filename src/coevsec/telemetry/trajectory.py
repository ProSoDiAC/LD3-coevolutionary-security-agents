"""Trajectory records and the JSONL telemetry writer.

Each step emits a record with a fixed set of fields. The dataset
of trajectories is a first-class research artifact enabling downstream work on
behavioural analysis, attack prediction and emergence detection.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrajectoryRecord:
    episode: int
    turn: int
    agent: str
    role: str
    observation: str
    action: str
    params: dict[str, Any]
    target: str | None
    reason: str  # the agent's rationale
    result: str
    success: bool
    strategy: str
    memory_update: str
    cost: float
    timestamp: float = field(default_factory=time.time)
    reasoning: str = ""  # optional CoT scratchpad, never fed to the validator

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


class TelemetryWriter:
    """Writes run metadata, per-step trajectories and per-episode summaries."""

    def __init__(self, output_dir: str, run_name: str, config_hash: str, seed: int) -> None:
        self.run_dir = Path(output_dir) / f"{run_name}_{config_hash}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.traj_path = self.run_dir / "trajectories.jsonl"
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.meta_path = self.run_dir / "meta.json"
        self._traj_fh = self.traj_path.open("w")
        self._ep_fh = self.episodes_path.open("w")
        self.meta = {
            "run_name": run_name,
            "config_hash": config_hash,
            "seed": seed,
            "git_sha": _git_sha(),
            "started_at": time.time(),
        }
        self.meta_path.write_text(json.dumps(self.meta, indent=2))

    def log_step(self, record: TrajectoryRecord) -> None:
        self._traj_fh.write(record.to_json() + "\n")
        self._traj_fh.flush()

    def log_episode(self, summary: dict[str, Any]) -> None:
        self._ep_fh.write(json.dumps(summary, sort_keys=True) + "\n")
        self._ep_fh.flush()

    def write_config(self, config: dict[str, Any]) -> None:
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))

    def write_summary(self, aggregate: dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(json.dumps(aggregate, indent=2))

    def close(self) -> None:
        self.meta["finished_at"] = time.time()
        self.meta_path.write_text(json.dumps(self.meta, indent=2))
        self._traj_fh.close()
        self._ep_fh.close()

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
