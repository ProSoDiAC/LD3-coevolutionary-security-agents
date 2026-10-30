"""The agent: a thin wrapper binding goal + policy + memory + budget.

Realises the simple v1 loop of:
    Goal + Observation + Memory -> Policy -> Tool selection -> (env) -> Observation -> Memory
The environment executes actions; the agent only decides and remembers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coevsec.agents.shared.memory import Memory, MemoryEntry
from coevsec.agents.shared.policy import Policy
from coevsec.core.tool import ToolSpec
from coevsec.core.types import Action, Observation, Role, StepResult


@dataclass
class Agent:
    agent_id: str
    role: Role
    goal: str
    policy: Policy
    memory: Memory
    encoded_behaviours: set[str] = field(default_factory=set)
    max_turns: int = 20
    token_budget: int = 20000

    # Runtime accounting (reset each episode).
    turns_used: int = 0
    tokens_used: int = 0

    def reset_episode(self) -> None:
        self.turns_used = 0
        self.tokens_used = 0
        self.memory.reset_episode()
        self.policy.reset_episode()

    def can_act(self) -> bool:
        return self.turns_used < self.max_turns and self.tokens_used < self.token_budget

    def decide(self, obs: Observation, tools: list[ToolSpec]) -> Action:
        remaining = max(0, self.token_budget - self.tokens_used)
        if hasattr(self.policy, "token_remaining"):
            self.policy.token_remaining = remaining
        return self.policy.choose(obs, tools, self.memory)

    def learn(self, action: Action, result: StepResult) -> None:
        self.turns_used += 1
        # LLM backends expose token counts on the policy's last response; heuristic
        # policies contribute zero, so budget accounting is uniform.
        self.tokens_used += getattr(self.policy, "last_tokens", 0)
        summary = result.detail.get("error") or ("ok" if result.success else "no-op")
        self.memory.record(MemoryEntry(turn=obs_turn(result), action=action.tool, result=str(summary)))
        self.policy.observe_result(action, result)

    def end_episode(self, summary: dict) -> None:
        summary = {**summary, "role": self.role.value}
        self.memory.end_episode(summary)
        self.policy.end_episode(summary)

    def strategy_signature(self) -> dict[str, float]:
        return self.policy.strategy_signature()


def obs_turn(result: StepResult) -> int:
    return result.observation.turn
