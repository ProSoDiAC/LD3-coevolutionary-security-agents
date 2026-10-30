"""Emergent-behaviour taxonomy and emergence definition.

This module provides the *labels* used for post-hoc analysis. Crucially, none of
these behaviours are hard-coded into agent objectives; they are detected after
the fact from trajectories. A behaviour counts as *emergent* only if it is not
encoded in the agent's objective/policy/script yet consistently arises from
interaction.
"""

from __future__ import annotations

# Offensive behaviours.
OFFENSIVE = {
    "reconnaissance",
    "stealth",
    "deception",
    "attack_diversification",
    "delayed_attack",
    "privilege_escalation",
    "strategic_retreat",
    "resource_targeting",
}

# Defensive behaviours.
DEFENSIVE = {
    "adaptive_monitoring",
    "deception",
    "dynamic_isolation",
    "decoy_deployment",
    "strategic_resource_allocation",
    "coordinated_defense",
}

# Social behaviours.
SOCIAL = {
    "cooperation",
    "coalition_formation",
    "trust_formation",
    "distrust",
    "information_sharing",
    "information_withholding",
    "deception",
    "retaliation",
}

ALL_BEHAVIOURS = OFFENSIVE | DEFENSIVE | SOCIAL

# Maps a tool category to the primary behaviour tags it can evidence. Used by the
# analysis engine as a first-pass heuristic before finer trajectory analysis.
CATEGORY_TO_BEHAVIOURS = {
    "recon": {"reconnaissance"},
    "stealth": {"stealth", "delayed_attack"},
    "exploit": {"privilege_escalation"},
    "exfiltrate": {"resource_targeting"},
    "wait": {"strategic_patience", "delayed_attack"},
    "communicate": {"information_sharing", "cooperation"},
    "monitor": {"adaptive_monitoring"},
    "isolate": {"dynamic_isolation"},
    "decoy": {"decoy_deployment", "deception"},
    "rotate": {"strategic_resource_allocation"},
    "investigate": {"adaptive_monitoring"},
}


def is_emergent(behaviour: str, encoded_behaviours: set[str]) -> bool:
    """Return True if ``behaviour`` was not explicitly encoded in the agent config.

    ``encoded_behaviours`` is the set of behaviours the agent was directly
    instructed to perform (via objective or policy). Anything outside that set,
    but observed consistently, is a candidate emergent behaviour.
    """
    return behaviour not in encoded_behaviours
