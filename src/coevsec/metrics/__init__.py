"""Metrics for co-evolutionary agent security."""

from coevsec.metrics.core import (
    attack_success_rate,
    attack_success_undetected_rate,
    detection_rate,
    detection_rate_given_engagement,
    engagement_rate,
    mean_cost,
    adaptation_gain,
    recovery_time,
    coalition_rate,
    deception_rate,
)
from coevsec.metrics.coevolution import (
    strategy_distance,
    strategy_diversity,
    behavioral_novelty,
    coevolutionary_pressure,
    coevolutionary_pressure_attacker,
    coevolutionary_pressure_defender,
    cross_lagged_correlation,
    emergent_security_risk,
    episode_shuffle_null,
)

__all__ = [
    "attack_success_rate",
    "attack_success_undetected_rate",
    "detection_rate",
    "detection_rate_given_engagement",
    "engagement_rate",
    "mean_cost",
    "adaptation_gain",
    "recovery_time",
    "coalition_rate",
    "deception_rate",
    "strategy_distance",
    "strategy_diversity",
    "behavioral_novelty",
    "coevolutionary_pressure",
    "coevolutionary_pressure_attacker",
    "coevolutionary_pressure_defender",
    "cross_lagged_correlation",
    "episode_shuffle_null",
    "emergent_security_risk",
]
