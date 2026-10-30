#!/usr/bin/env python3
"""Reproducible experiment entry point.

Usage:
    python run_experiment.py --config configs/examples/1v1_static.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from coevsec.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
