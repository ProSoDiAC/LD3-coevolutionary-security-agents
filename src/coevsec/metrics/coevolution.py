"""Co-evolutionary metrics: diversity, novelty, CEP and ESR."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def strategy_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Euclidean distance between two strategy signatures over their key union."""
    keys = set(a) | set(b)
    return math.sqrt(sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in keys))


def strategy_diversity(signatures: Sequence[dict[str, float]]) -> float:
    """Mean pairwise strategy distance within a set of signatures."""
    sigs = list(signatures)
    if len(sigs) < 2:
        return 0.0
    total, n = 0.0, 0
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            total += strategy_distance(sigs[i], sigs[j])
            n += 1
    return total / n if n else 0.0


def behavioral_novelty(per_episode_behaviours: Sequence[set[str]]) -> float:
    """Fraction of episodes that introduce a behaviour not seen before."""
    seen: set[str] = set()
    novel_episodes = 0
    for behaviours in per_episode_behaviours:
        if behaviours - seen:
            novel_episodes += 1
        seen |= behaviours
    return novel_episodes / len(per_episode_behaviours) if per_episode_behaviours else 0.0


def _episode_deltas(sigs: Sequence[dict[str, float]]) -> list[float]:
    if len(sigs) < 2:
        return []
    return [strategy_distance(sigs[t], sigs[t - 1]) for t in range(1, len(sigs))]


def coevolutionary_pressure(
    attacker_sigs: Sequence[dict[str, float]],
    defender_sigs: Sequence[dict[str, float]],
) -> float:
    """CEP = mean over episodes of (ΔS_A + ΔS_D)/2."""
    n = min(len(attacker_sigs), len(defender_sigs))
    if n < 2:
        return 0.0
    total = 0.0
    for t in range(1, n):
        d_a = strategy_distance(attacker_sigs[t], attacker_sigs[t - 1])
        d_d = strategy_distance(defender_sigs[t], defender_sigs[t - 1])
        total += (d_a + d_d) / 2.0
    return total / (n - 1)


def coevolutionary_pressure_attacker(attacker_sigs: Sequence[dict[str, float]]) -> float:
    deltas = _episode_deltas(attacker_sigs)
    return sum(deltas) / len(deltas) if deltas else 0.0


def coevolutionary_pressure_defender(defender_sigs: Sequence[dict[str, float]]) -> float:
    deltas = _episode_deltas(defender_sigs)
    return sum(deltas) / len(deltas) if deltas else 0.0


def cep_baseline_adjusted(cep: float, baseline_cep: float) -> float:
    """Subtract static/static CEP from a cell (same seed) to remove ambient variability."""
    return cep - baseline_cep


def episode_shuffle_null(
    attacker_sigs: Sequence[dict[str, float]],
    defender_sigs: Sequence[dict[str, float]],
    *,
    rng: random.Random | None = None,
) -> float:
    """Null CEP: shuffle episode order of signatures within a run."""
    rng = rng or random.Random(0)
    att = list(attacker_sigs)
    dfn = list(defender_sigs)
    rng.shuffle(att)
    rng.shuffle(dfn)
    return coevolutionary_pressure(att, dfn)


def cross_lagged_correlation(
    attacker_sigs: Sequence[dict[str, float]],
    defender_sigs: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Pearson r between Δdefender_t and Δattacker_{t+1} (and reverse)."""
    n = min(len(attacker_sigs), len(defender_sigs))
    if n < 3:
        return {"def_to_att": float("nan"), "att_to_def": float("nan")}

    d_att = _episode_deltas(attacker_sigs[:n])
    d_def = _episode_deltas(defender_sigs[:n])
    if len(d_att) < 2 or len(d_def) < 2:
        return {"def_to_att": float("nan"), "att_to_def": float("nan")}

    def _pearson(xs: list[float], ys: list[float]) -> float:
        m = min(len(xs), len(ys))
        if m < 2:
            return float("nan")
        xs, ys = xs[:m], ys[:m]
        mx, my = sum(xs) / m, sum(ys) / m
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
        den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
        if den_x == 0 or den_y == 0:
            return float("nan")
        return num / (den_x * den_y)

    # defender change at t predicts attacker change at t+1
    def_to_att = _pearson(d_def[:-1], d_att[1:])
    att_to_def = _pearson(d_att[:-1], d_def[1:])
    return {"def_to_att": def_to_att, "att_to_def": att_to_def}


def emergent_security_risk(p_fail_interaction: float, p_fail_isolation: float) -> float:
    """ESR = P(failure|interaction) / P(failure|isolation).

    Returns +inf if isolation never fails but interaction does; 1.0 if both zero.
    """
    if p_fail_isolation <= 0.0:
        return float("inf") if p_fail_interaction > 0.0 else 1.0
    return p_fail_interaction / p_fail_isolation
