"""Build concrete agents from configuration."""

from __future__ import annotations

import hashlib
import random

from coevsec.agents.attacker.policy import AttackerHeuristicPolicy
from coevsec.agents.defender.policy import DefenderHeuristicPolicy
from coevsec.agents.shared.agent import Agent
from coevsec.agents.shared.memory import make_memory
from coevsec.agents.shared.policy import HybridPolicy, LLMPolicy
from coevsec.core.config import AgentConfig
from coevsec.core.types import Role

_DEFAULT_GOALS = {
    Role.ATTACKER: "Compromise and exfiltrate data from the data store while minimising detection.",
    Role.DEFENDER: "Protect the data store while minimising operational cost and false positives.",
}


def _heuristic(agent_id: str, role: Role, adaptation: str, rng: random.Random, group: AgentConfig):
    pacing_off = bool(group.policy.params.get("disable_stealth_pacing", False))
    abandon = bool(group.policy.params.get("abandon_failed_hosts", False))
    if role == Role.ATTACKER:
        return AttackerHeuristicPolicy(
            agent_id, adaptation, rng,
            stealth=group.policy.stealth, aggression=group.policy.aggression,
            disable_stealth_pacing=pacing_off, abandon_failed_hosts=abandon,
        )
    return DefenderHeuristicPolicy(
        agent_id, adaptation, rng, vigilance=group.policy.stealth,
        disable_investigate=bool(group.policy.params.get("disable_investigate", False)),
        disable_decoy=bool(group.policy.params.get("disable_decoy", False)),
    )


def make_agent(agent_id: str, group: AgentConfig, seed: int) -> Agent:
    role = Role(group.role)
    digest = hashlib.sha256(f"{seed}:{agent_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    goal = group.goal or _DEFAULT_GOALS[role]
    memory = make_memory(group.adaptation.value)

    if group.policy.kind in {"llm", "hybrid"}:
        from coevsec.llm import make_backend

        backend = make_backend(group.llm)
        use_cot = bool(group.policy.params.get("use_cot", False))
        sample_digest = hashlib.sha256(f"llm:{seed}:{agent_id}".encode()).digest()
        llm_policy = LLMPolicy(
            agent_id, role, group.adaptation.value, rng, backend, goal, use_cot=use_cot,
            sample_rng=random.Random(int.from_bytes(sample_digest[:8], "big")),
        )
        if group.policy.kind == "hybrid":
            heur = _heuristic(agent_id, role, group.adaptation.value, rng, group)
            stall = int(group.policy.params.get("stall_limit", 2))
            force = bool(group.policy.params.get("force_legal_hints", True))
            policy = HybridPolicy(
                llm_policy, heur, stall_limit=stall, force_legal_hints=force,
            )
        else:
            policy = llm_policy
    else:
        adaptation = "static" if group.policy.kind == "rule_based" else group.adaptation.value
        policy = _heuristic(agent_id, role, adaptation, rng, group)

    return Agent(
        agent_id=agent_id,
        role=role,
        goal=goal,
        policy=policy,
        memory=memory,
        encoded_behaviours=set(group.encoded_behaviours),
        max_turns=group.max_turns,
        token_budget=group.token_budget,
    )
