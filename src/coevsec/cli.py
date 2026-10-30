"""Command-line interface: run experiments, analyse runs, ingest into Postgres."""

from __future__ import annotations

import argparse
import json
import math
import sys

from coevsec.core.env import load_dotenv
from coevsec.experiments import run_experiment


def _json_safe(obj):
    """Replace NaN/inf (invalid in JSON) by null."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _cmd_run(args: argparse.Namespace) -> int:
    load_dotenv()
    aggregate = run_experiment(args.config)
    print(json.dumps(_json_safe(aggregate), indent=2))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from coevsec.analysis.engine import analyze_run

    report = analyze_run(args.run, out_dir=args.out)
    print(json.dumps(report, indent=2, default=str))
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from coevsec.telemetry.postgres import available, ingest_run

    if not available():
        print("Postgres not available (install extras + set COEVSEC_PG_DSN)", file=sys.stderr)
        return 1
    rows = ingest_run(args.run)
    print(f"ingested {rows} rows")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coevsec", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an experiment from a YAML config")
    p_run.add_argument("--config", required=True)
    p_run.set_defaults(func=_cmd_run)

    p_an = sub.add_parser("analyze", help="analyse a completed run directory")
    p_an.add_argument("--run", required=True)
    p_an.add_argument("--out", default=None)
    p_an.set_defaults(func=_cmd_analyze)

    p_in = sub.add_parser("ingest", help="ingest a run's trajectories into Postgres")
    p_in.add_argument("--run", required=True)
    p_in.set_defaults(func=_cmd_ingest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
