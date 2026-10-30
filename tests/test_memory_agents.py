"""Memory levels and agent loop accounting."""

from __future__ import annotations

from coevsec.agents.factory import make_agent
from coevsec.agents.shared.memory import (
    EpisodicMemory,
    MemoryEntry,
    NullMemory,
    PersistentMemory,
    make_memory,
)
from coevsec.core.config import AgentConfig, PolicyConfig
from coevsec.core.types import AdaptationLevel, Role


def test_null_memory_is_empty():
    m = NullMemory()
    m.record(MemoryEntry(turn=0, action="recon", result="ok"))
    assert m.context() == ""


def test_episodic_resets():
    m = EpisodicMemory()
    m.record(MemoryEntry(turn=1, action="recon", result="ok"))
    assert "recon" in m.context()
    m.reset_episode()
    assert m.context() == ""


def test_persistent_keeps_reflections():
    m = PersistentMemory()
    m.end_episode({"role": "attacker", "attack_success": True, "detected": False})
    m.reset_episode()
    ctx = m.context()
    assert "Stealthy path" in ctx or "Lessons" in ctx


def test_make_memory_dispatch():
    assert isinstance(make_memory("static"), NullMemory)
    assert isinstance(make_memory("episodic"), EpisodicMemory)
    assert isinstance(make_memory("persistent"), PersistentMemory)


def test_make_agent_rule_based_is_static():
    group = AgentConfig(
        id_prefix="attacker",
        role="attacker",
        adaptation=AdaptationLevel.EPISODIC,
        policy=PolicyConfig(kind="rule_based", stealth=0.3),
    )
    agent = make_agent("attacker", group, seed=1)
    assert agent.role == Role.ATTACKER
    assert agent.policy.adaptation == "static"
    rng_a = make_agent("attacker", group, seed=1).policy.rng
    rng_b = make_agent("attacker", group, seed=1).policy.rng
    assert rng_a.random() == rng_b.random()
