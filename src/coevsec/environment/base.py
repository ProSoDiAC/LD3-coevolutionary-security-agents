"""Abstract cyber environment interface.

Both the Python simulator and the K8s cyber range implement this contract, so
agents, metrics and telemetry are entirely backend-agnostic.
"""

from __future__ import annotations

import abc

from coevsec.core.tool import ToolRegistry
from coevsec.core.types import Action, Observation, Role, StepResult


class CyberEnvironment(abc.ABC):
    """A controlled, sandboxed cyber environment agents interact with.

    Implementations expose a set of *capabilities* (tools) per role and enforce
    partial observability: :meth:`observe` returns only what a given agent has
    discovered, never the full state.
    """

    #: Tool registry describing the capabilities exposed to each role.
    tools: ToolRegistry

    @abc.abstractmethod
    def reset(self, seed: int, profile_seed: int | None = None) -> None:
        """Reset to a fresh, deterministic initial state for a new episode.

        ``seed`` drives the per-turn stochastic draws and varies every episode;
        ``profile_seed`` (defaulting to ``seed``) drives the cross-episode
        structure, so a recurring environment regime can repeat it.
        """

    @abc.abstractmethod
    def observe(self, agent_id: str) -> Observation:
        """Return the partial observation visible to ``agent_id``."""

    @abc.abstractmethod
    def step(self, action: Action) -> StepResult:
        """Apply one action and return its result (observation, events, cost)."""

    @abc.abstractmethod
    def state_snapshot(self) -> dict:
        """Return a full ground-truth snapshot (for telemetry/analysis only)."""

    @abc.abstractmethod
    def is_compromised(self, asset: str) -> bool:
        """Whether ``asset`` (e.g. the data store) is currently compromised."""

    @abc.abstractmethod
    def objective_met(self) -> bool:
        """Whether the attacker's terminal objective has been achieved."""

    def end_turn(self) -> list:
        """Apply per-turn decay / auto-detection and advance the clock.

        Backends that have no turn-level dynamics may leave this as a no-op.
        """
        return []

    def episode_summary(self) -> dict:
        """Ground-truth outcome used by metrics (ASR, costs, recovery, ...)."""
        return {}

    def available_tools(self, role: Role):
        """Tools usable by ``role`` in this environment."""
        return self.tools.for_role(role)

    def register_agent(self, agent_id: str, role: Role) -> None:  # noqa: B027
        """Optional hook for backends that track per-agent state."""

    def teardown(self) -> None:  # noqa: B027
        """Release any external resources (e.g. tear down a cluster)."""
