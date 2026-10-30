"""Post-hoc analysis of a completed run.

Loads the run's telemetry, produces per-episode metric tables, characterises the
strategy-evolution dynamics (stable equilibrium vs oscillating arms race, RQ5),
and tags candidate emergent behaviours against the taxonomy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from coevsec.core.taxonomy import CATEGORY_TO_BEHAVIOURS
from coevsec.environment.sim.tools import build_tool_registry
from coevsec.metrics.coevolution import strategy_distance


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # A truncated trailing line (e.g. from an interrupted run) should not
            # discard every other record already written for this run.
            continue
    return out


def analyze_run(run_dir: str | Path, out_dir: str | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    episodes = _load_jsonl(run_dir / "episodes.jsonl")
    trajectories = _load_jsonl(run_dir / "trajectories.jsonl")
    config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}

    out = Path(out_dir) if out_dir else run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    ep_df = pd.DataFrame(episodes)
    report: dict[str, Any] = {"run_dir": str(run_dir), "n_episodes": len(episodes)}

    if not ep_df.empty:
        ep_df.to_csv(out / "episodes.csv", index=False)
        report["metric_table"] = _metric_table(ep_df)
        report["strategy_dynamics"] = _strategy_dynamics(episodes)

    if trajectories:
        report["emergence"] = _emergence(trajectories, config)
        report["action_distribution"] = _action_distribution(trajectories)

    report["coalition_graph"] = _coalition_graph(episodes, out)

    _maybe_plot(episodes, out, report)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _metric_table(ep_df: pd.DataFrame) -> dict[str, Any]:
    cols = [c for c in ["attack_success", "detected", "attacker_cost", "defender_cost",
                        "detection_cost"] if c in ep_df.columns]
    table = {c: float(ep_df[c].mean()) for c in cols}
    # First-half vs second-half ASR trend (recovery / co-evolution signal, H2).
    if "attack_success" in ep_df.columns and len(ep_df) >= 4:
        half = len(ep_df) // 2
        table["asr_first_half"] = float(ep_df["attack_success"].iloc[:half].mean())
        table["asr_second_half"] = float(ep_df["attack_success"].iloc[half:].mean())
    return table


def _param_distance(
    a: dict[str, float], b: dict[str, float], keys: tuple[str, ...] = ("stealth", "vigilance"),
) -> float:
    """Distance on adaptation parameters only (stealth / vigilance)."""
    return math.sqrt(sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in keys))


_CARRIED_KEYS = ("base_stealth", "base_vigilance")


def _strategy_dynamics(episodes: list[dict]) -> dict[str, Any]:
    """Characterise co-evolution: full-signature and parameter-only dynamics."""
    att = [e.get("attacker_strategy", {}) for e in episodes]
    dfn = [e.get("defender_strategy", {}) for e in episodes]
    deltas = []
    param_deltas = []
    for t in range(1, len(episodes)):
        d_a = strategy_distance(att[t], att[t - 1])
        d_d = strategy_distance(dfn[t], dfn[t - 1])
        deltas.append((d_a + d_d) / 2.0)
        p_a = _param_distance(att[t], att[t - 1])
        p_d = _param_distance(dfn[t], dfn[t - 1])
        param_deltas.append((p_a + p_d) / 2.0)
    if not deltas:
        return {"classification": "insufficient_data", "mean_delta": 0.0}

    def _classify(ds: list[float]) -> dict[str, Any]:
        half = max(1, len(ds) // 2)
        early = sum(ds[:half]) / half
        late = sum(ds[half:]) / max(1, len(ds) - half)
        mean_delta = sum(ds) / len(ds)
        if late < 0.02:
            classification = "stable_equilibrium"
        elif late >= 0.02 and late >= 0.6 * early:
            classification = "non_convergent"
        else:
            classification = "converging"
        return {
            "classification": classification,
            "mean_delta": round(mean_delta, 4),
            "early_delta": round(early, 4),
            "late_delta": round(late, 4),
        }

    full = _classify(deltas)
    param = _classify(param_deltas)
    # Inter-episode dynamics: parameters carried into the next episode. Under
    # L0/L1 these never change, so only L2 roles can move them.
    carried_deltas = []
    att_c = [e.get("attacker_carried_params") or {} for e in episodes]
    dfn_c = [e.get("defender_carried_params") or {} for e in episodes]
    has_carried = any(att_c) or any(dfn_c)
    if has_carried:
        for t in range(1, len(episodes)):
            carried_deltas.append((
                _param_distance(att_c[t], att_c[t - 1], _CARRIED_KEYS)
                + _param_distance(dfn_c[t], dfn_c[t - 1], _CARRIED_KEYS)
            ) / 2.0)
        carried = _classify(carried_deltas)
    att_deltas = [strategy_distance(att[t], att[t - 1]) for t in range(1, len(att))]
    def_deltas = [strategy_distance(dfn[t], dfn[t - 1]) for t in range(1, len(dfn))]
    return {
        # Primary label: carried-over (inter-episode) parameter dynamics when
        # logged; end-of-episode parameter dynamics otherwise.
        "classification": carried["classification"] if has_carried else param["classification"],
        "classification_end_of_episode": param["classification"],
        "carried_early_delta": carried["early_delta"] if has_carried else None,
        "carried_late_delta": carried["late_delta"] if has_carried else None,
        "classification_full_signature": full["classification"],
        "mean_delta": full["mean_delta"],
        "early_delta": full["early_delta"],
        "late_delta": full["late_delta"],
        "param_mean_delta": param["mean_delta"],
        "param_early_delta": param["early_delta"],
        "param_late_delta": param["late_delta"],
        "delta_series": [round(d, 4) for d in deltas],
        "param_delta_series": [round(d, 4) for d in param_deltas],
        "attacker_delta_series": [round(d, 4) for d in att_deltas],
        "defender_delta_series": [round(d, 4) for d in def_deltas],
    }


def _emergence(trajectories: list[dict], config: dict) -> dict[str, Any]:
    """Tag observed behaviours as encoded vs emergent."""
    reg = build_tool_registry()
    encoded: set[str] = set()
    for group in config.get("agents", []):
        encoded |= set(group.get("encoded_behaviours", []))

    observed: dict[str, int] = {}
    for r in trajectories:
        tool = reg.get(r.get("action", ""))
        if tool is None:
            continue
        for b in CATEGORY_TO_BEHAVIOURS.get(tool.category, set()):
            observed[b] = observed.get(b, 0) + 1

    emergent = {b: n for b, n in observed.items() if b not in encoded}
    return {
        "encoded_behaviours": sorted(encoded),
        "observed_behaviours": observed,
        "candidate_emergent": dict(sorted(emergent.items(), key=lambda kv: -kv[1])),
    }


def _action_distribution(trajectories: list[dict]) -> dict[str, dict[str, int]]:
    dist: dict[str, dict[str, int]] = {}
    for r in trajectories:
        role = r.get("role", "?")
        dist.setdefault(role, {})
        tool = r.get("action", "?")
        dist[role][tool] = dist[role].get(tool, 0) + 1
    return dist


def _coalition_graph(episodes: list[dict], out: Path) -> dict[str, Any]:
    """Build an aggregate interaction/coalition graph across episodes (NetworkX)."""
    import networkx as nx

    g = nx.Graph()
    n_with = 0
    for e in episodes:
        coals = e.get("coalitions") or []
        if coals:
            n_with += 1
        for group in coals:
            members = list(group)
            for i, u in enumerate(members):
                g.add_node(u)
                for v in members[i + 1 :]:
                    if g.has_edge(u, v):
                        g[u][v]["weight"] = g[u][v].get("weight", 0) + 1
                    else:
                        g.add_edge(u, v, weight=1)
    payload = {
        "episodes_with_coalitions": n_with,
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "density": round(nx.density(g), 4) if g.number_of_nodes() > 1 else 0.0,
        "components": [sorted(c) for c in nx.connected_components(g)] if g.number_of_nodes() else [],
    }
    if g.number_of_nodes():
        nx.write_edgelist(g, out / "coalition_graph.edgelist", data=["weight"])
        payload["edgelist"] = str(out / "coalition_graph.edgelist")
    return payload


def _maybe_plot(episodes: list[dict], out: Path, report: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        report["plots"] = "matplotlib not installed; skipped"
        return
    if not episodes:
        return

    idx = [e.get("episode", i) for i, e in enumerate(episodes)]
    asr = [1.0 if e.get("attack_success") else 0.0 for e in episodes]
    det = [1.0 if e.get("detected") else 0.0 for e in episodes]
    stealth = [e.get("attacker_strategy", {}).get("stealth", None) for e in episodes]
    vigil = [e.get("defender_strategy", {}).get("vigilance", None) for e in episodes]

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(idx, _rolling(asr), label="attack success (rolling)")
    ax[0].plot(idx, _rolling(det), label="detected (rolling)")
    ax[0].set_ylabel("rate")
    ax[0].legend()
    ax[0].set_title("Outcome dynamics")
    if any(s is not None for s in stealth):
        ax[1].plot(idx, stealth, label="attacker stealth")
    if any(v is not None for v in vigil):
        ax[1].plot(idx, vigil, label="defender vigilance")
    ax[1].set_xlabel("episode")
    ax[1].set_ylabel("strategy param")
    ax[1].legend()
    ax[1].set_title("Strategy evolution")
    fig.tight_layout()
    fig.savefig(out / "dynamics.png", dpi=120)
    plt.close(fig)
    report["plots"] = str(out / "dynamics.png")


def _rolling(xs: list[float], w: int = 3) -> list[float]:
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        window = xs[lo : i + 1]
        out.append(sum(window) / len(window))
    return out
