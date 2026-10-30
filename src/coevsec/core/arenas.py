"""Named detector presets for the two evaluation arenas.

``saturated`` is the original arena, where the defender detects nearly every
engaged attacker (DR close to 1), so defender-side hypotheses have no headroom.
``headroom`` weakens accumulation and adds benign background traffic, giving a
static detection rate around 0.65 with the attacker still engaging every
episode, so both roles' adaptation is measurable. Both are used as controls
for each other.
"""

from __future__ import annotations

ARENAS: dict[str, dict[str, float]] = {
    "saturated": {
        "suspicion_lambda": 0.12,
        "suspicion_decay": 0.85,
        "investigate_threshold": 1.5,
        "benign_noise": 0.0,
    },
    "headroom": {
        "suspicion_lambda": 0.05,
        "suspicion_decay": 0.55,
        "investigate_threshold": 2.4,
        "benign_noise": 0.4,
    },
}


def detector(arena: str) -> dict[str, float]:
    """Detector config dict for a named arena."""
    if arena not in ARENAS:
        raise ValueError(f"unknown arena: {arena} (known: {sorted(ARENAS)})")
    return dict(ARENAS[arena])
