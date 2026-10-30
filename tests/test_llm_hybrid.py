"""Tests for LLM param autofill, legal hints, and hybrid takeover."""

from coevsec.agents.shared.policy import (
    HybridPolicy,
    LLMPolicy,
    _autofill_params,
    _legal_attacker_hints,
)
from coevsec.core.types import Observation, Role
from coevsec.llm.mock import MockBackend
import random


def test_autofill_redirects_harvest_to_service():
    obs = Observation(
        agent_id="a", turn=3,
        data={
            "known_hosts": [
                {"host": "gw", "layer": "gateway", "reachable": True,
                 "foothold": True, "known_vulns": ["gw_v0"]},
                {"host": "svc_00", "layer": "service", "reachable": True,
                 "foothold": True, "known_vulns": ["svc_00_v0"]},
            ],
            "footholds": ["gw", "svc_00"], "credentials": [],
        },
    )
    params = _autofill_params("harvest_credentials", {"host": "gw"}, obs)
    assert params["host"] == "svc_00"



def test_legal_hints_progress_past_recon():
    obs = Observation(
        agent_id="a", turn=2,
        data={
            "known_hosts": [
                {"host": "gw", "layer": "gateway", "reachable": True,
                 "foothold": False, "known_vulns": []},
            ],
            "footholds": [], "credentials": [],
        },
    )
    hints = _legal_attacker_hints(obs)
    assert any("probe_service" in h for h in hints)


def test_hybrid_falls_back_when_llm_stuck(monkeypatch):
    rng = random.Random(0)
    backend = MockBackend()
    llm = LLMPolicy("a", Role.ATTACKER, "episodic", rng, backend, "goal")

    class _Heur:
        def __init__(self):
            self.adaptation = "episodic"
        def choose(self, obs, tools, memory):
            from coevsec.core.types import Action
            return Action(agent_id="a", tool="exploit", params={"host": "gw"}, rationale="h")
        def reset_episode(self): pass
        def end_episode(self, s): pass
        def observe_result(self, a, r): pass
        def strategy_signature(self): return {"stealth": 0.3}

    # Force stuck flag after LLM choose.
    def sticky(obs, tools, memory):
        from coevsec.core.types import Action
        llm.stuck = True
        llm.last_was_repaired = True
        return Action(agent_id="a", tool="recon", params={})
    llm.choose = sticky  # type: ignore

    hybrid = HybridPolicy(llm, _Heur(), stall_limit=1)
    obs = Observation(agent_id="a", turn=1, data={"known_hosts": [{"host": "gw"}], "footholds": []})
    # First call increments stall; stall_limit=1 triggers heur when stall>=1 after choose
    # Actually: stuck -> stall+=1; if stall>=limit use heur. First call stall becomes 1 >= 1 -> heur.
    from coevsec.core.tool import ToolSpec
    tools = [ToolSpec(name="recon", description="", roles=(Role.ATTACKER,)),
             ToolSpec(name="exploit", description="", roles=(Role.ATTACKER,))]
    action = hybrid.choose(obs, tools, memory=type("M", (), {"context": lambda self: ""})())
    assert action.tool == "exploit"
    assert "hybrid" in action.rationale
