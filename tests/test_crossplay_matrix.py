"""Regression coverage for the stateful cross-play fix: each source cell's
native per-role adaptation level (CELL_ADAPTATION) must actually reach the
built ExperimentConfig, and building configs for one cell pair must not leak
state into configs for another pair (the original bug forced STATIC on both
roles for every pair regardless of the checkpoint's cell of origin)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from coevsec.core.types import AdaptationLevel  # noqa: E402

import crossplay_matrix as cm  # noqa: E402


def test_cell_adaptation_mapping():
    assert cm.CELL_ADAPTATION["static_static"] == (AdaptationLevel.STATIC, AdaptationLevel.STATIC)
    assert cm.CELL_ADAPTATION["adaptive_static"] == (AdaptationLevel.EPISODIC, AdaptationLevel.STATIC)
    assert cm.CELL_ADAPTATION["persistent_persistent"] == (
        AdaptationLevel.PERSISTENT, AdaptationLevel.PERSISTENT,
    )


def test_iso_cfg_applies_requested_adaptation_levels():
    cfg = cm._iso_cfg("x", seed=1, att_ad=AdaptationLevel.PERSISTENT,
                       def_ad=AdaptationLevel.STATIC, episodes=1)
    by_role = {a.role: a for a in cfg.agents}
    assert by_role["attacker"].adaptation == AdaptationLevel.PERSISTENT
    assert by_role["defender"].adaptation == AdaptationLevel.STATIC


def test_iso_cfg_calls_do_not_leak_between_cell_pairs():
    # Building a PERSISTENT/PERSISTENT config and then a STATIC/STATIC config
    # must not have the second reflect the first's adaptation levels.
    cfg_pp = cm._iso_cfg("pp", 1, AdaptationLevel.PERSISTENT, AdaptationLevel.PERSISTENT, 1)
    cfg_ss = cm._iso_cfg("ss", 1, AdaptationLevel.STATIC, AdaptationLevel.STATIC, 1)
    assert all(a.adaptation == AdaptationLevel.PERSISTENT for a in cfg_pp.agents)
    assert all(a.adaptation == AdaptationLevel.STATIC for a in cfg_ss.agents)
