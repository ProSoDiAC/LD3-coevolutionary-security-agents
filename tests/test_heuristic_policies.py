"""Regression tests for the attacker/defender heuristic policies' per-episode
state: the persistent-adaptation update rule (all four detected/success
combinations) and the episode-boundary reset of the rolling action-category
window that feeds strategy_signature()."""

from __future__ import annotations

import random

from coevsec.agents.attacker.policy import AttackerHeuristicPolicy
from coevsec.agents.defender.policy import DefenderHeuristicPolicy
from coevsec.agents.shared.memory import NullMemory
from coevsec.core.types import AdaptationLevel, Observation, Role
from coevsec.environment.sim.tools import build_tool_registry


def _attacker(adaptation=AdaptationLevel.PERSISTENT, stealth=0.5):
    return AttackerHeuristicPolicy("attacker", adaptation.value, random.Random(0), stealth=stealth)


def test_attacker_end_episode_detected_not_success():
    p = _attacker()
    p.end_episode({"detected": True, "attack_success": False})
    assert p.base_stealth == 0.65  # +0.15


def test_attacker_end_episode_success_not_detected():
    p = _attacker()
    p.end_episode({"detected": False, "attack_success": True})
    assert p.base_stealth == 0.45  # -0.05


def test_attacker_end_episode_neither():
    p = _attacker()
    p.end_episode({"detected": False, "attack_success": False})
    assert p.base_stealth == 0.4  # -0.10


def test_attacker_end_episode_both_detected_and_success():
    p = _attacker()
    p.end_episode({"detected": True, "attack_success": True})
    assert abs(p.base_stealth - 0.55) < 1e-9  # +0.05


def test_attacker_end_episode_noop_unless_persistent():
    p = _attacker(adaptation=AdaptationLevel.EPISODIC)
    p.end_episode({"detected": True, "attack_success": True})
    assert p.base_stealth == 0.5  # unchanged: not PERSISTENT


def test_attacker_recent_categories_reset_between_episodes():
    p = _attacker(adaptation=AdaptationLevel.EPISODIC)
    p._recent_categories = ["exploit"] * 16
    p.reset_episode()
    assert p._recent_categories == []
    assert p.strategy_signature() == {"stealth": p.stealth}


def test_defender_recent_categories_reset_between_episodes():
    p = DefenderHeuristicPolicy("defender", AdaptationLevel.EPISODIC.value, random.Random(0))
    p._recent_categories = ["isolate"] * 16
    p.reset_episode()
    assert p._recent_categories == []
    assert p.strategy_signature() == {"vigilance": p.vigilance}


def _defender_tools():
    reg = build_tool_registry()
    return reg.for_role(Role.DEFENDER)


def _obs(suspicion: float, detections: int = 0) -> Observation:
    return Observation(
        agent_id="defender", turn=0,
        data={
            "hosts": [{"host": "svc_00", "layer": "service", "suspicion": suspicion,
                       "monitored": 1.0, "isolated": False}],
            "detections": detections,
        },
    )


def test_defender_escalates_on_rising_suspicion_without_detection():
    """Pre-detection channel: vigilance should rise from a suspicion jump
    alone, with detections staying at 0 throughout (see docs/notes on H2)."""
    p = DefenderHeuristicPolicy("defender", AdaptationLevel.EPISODIC.value, random.Random(0))
    tools = _defender_tools()
    before = p.vigilance
    p.choose(_obs(suspicion=0.0, detections=0), tools, NullMemory())
    p.choose(_obs(suspicion=0.9, detections=0), tools, NullMemory())
    assert p.vigilance > before


def test_defender_suspicion_channel_noop_when_static():
    p = DefenderHeuristicPolicy("defender", AdaptationLevel.STATIC.value, random.Random(0))
    tools = _defender_tools()
    before = p.vigilance
    p.choose(_obs(suspicion=0.0, detections=0), tools, NullMemory())
    p.choose(_obs(suspicion=0.9, detections=0), tools, NullMemory())
    assert p.vigilance == before
