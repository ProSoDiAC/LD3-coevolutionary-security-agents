"""Load ``KEY=value`` pairs from the repository's ``.env`` without overriding
variables already set in the environment."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
