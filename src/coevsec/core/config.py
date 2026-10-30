"""Experiment configuration schema.

A single YAML file fully specifies a reproducible experiment: models, number of
agents, topology, capabilities, memory level, communication, seed, episode
count, environment backend and budget. Parsed and validated with pydantic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from coevsec.core.types import AdaptationLevel


class LLMConfig(BaseModel):
    """LLM backend selection. ``mock`` and ``heuristic`` policies ignore this."""

    provider: Literal["ollama", "mock"] = "ollama"
    base_url: str | None = None  # falls back to COEVSEC_OLLAMA_BASE_URL
    model: str | None = None  # falls back to COEVSEC_OLLAMA_MODEL
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_s: float = 120.0
    # Context window requested from Ollama. Fixed explicitly because the
    # server default varies across Ollama versions; prompts stay below 3k tokens.
    num_ctx: int = 8192

    def resolved_base_url(self) -> str:
        return self.base_url or os.environ.get("COEVSEC_OLLAMA_BASE_URL", "http://localhost:11434")

    def resolved_model(self) -> str:
        return self.model or os.environ.get("COEVSEC_OLLAMA_MODEL", "llama3.1:8b")


class PolicyConfig(BaseModel):
    """How an agent chooses actions.

    ``kind`` selects the policy implementation:
        rule_based  alias of the heuristic policy at L0 (static)
        heuristic   parameterised heuristic that can adapt (no LLM required)
        llm         LLM-driven policy via the configured backend
        hybrid      LLM primary with heuristic takeover when stuck (B3–B5 fix)
    """

    kind: Literal["rule_based", "heuristic", "llm", "hybrid"] = "heuristic"
    # Baseline knobs for heuristic policies; interpreted per role.
    stealth: float = 0.5  # initial preference for low-noise actions (0..1)
    aggression: float = 0.5  # initial preference for direct high-value actions
    params: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """A single agent or a homogeneous group of ``count`` agents."""

    id_prefix: str
    role: Literal["attacker", "defender"]
    count: int = 1
    goal: str = ""
    adaptation: AdaptationLevel = AdaptationLevel.EPISODIC
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    # Which behaviours the agent is *explicitly* instructed to use. Used by the
    # emergence detector to distinguish encoded vs emergent.
    encoded_behaviours: list[str] = Field(default_factory=list)
    # Per-episode budget: max turns the agent may act.
    max_turns: int = 20
    token_budget: int = 20000


class DetectorConfig(BaseModel):
    """Noisy IDS / suspicion model (simulator)."""

    suspicion_lambda: float = 0.12  # P(detect) = 1 - exp(-lambda * suspicion)
    suspicion_decay: float = 0.85  # multiplicative decay per turn (1.0 = no decay)
    monitor_decay: float = 0.8
    investigate_threshold: float = 1.5  # base threshold; lowered by monitor_level
    wait_cost: float = 0.1  # action cost for wait (higher = pacing penalty)
    decay_while_foothold: bool = True  # if false, skip suspicion decay on compromised hosts
    # Benign background traffic: each turn adds up to this much suspicion to one
    # random non-compromised host. Positive values make some investigations
    # false positives and lower the signal-to-noise of real detection, giving
    # the defender headroom (a non-saturated arena) instead of DR ~ 1.
    benign_noise: float = 0.0


class EnvironmentConfig(BaseModel):
    """Cyber environment selection and parameters."""

    backend: Literal["sim", "k8s"] = "sim"
    hosts: int = 3  # number of service hosts between gateway and data store
    vulnerabilities_per_host: int = 2
    detection_base_rate: float = 0.15  # legacy alias; synced to suspicion_lambda if unset
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    # Cross-episode environment structure (vulnerability difficulty vectors and
    # which service host holds the easiest credentialed path). ``iid`` re-draws
    # it every episode (no transferable structure); ``recurring`` cycles a fixed
    # set of ``n_profiles`` latent structures; ``drift`` uses one structure for
    # the first ``drift_at`` fraction of episodes and another afterwards. The
    # per-turn stochastic draws (exploit success, detector) always vary.
    regime: Literal["iid", "recurring", "drift"] = "iid"
    n_profiles: int = 4
    drift_at: float = 0.5
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_detector_legacy(self) -> "EnvironmentConfig":
        # Keep YAML ``detection_base_rate`` as the IDS rate knob for older configs.
        if self.detection_base_rate != 0.15 and self.detector.suspicion_lambda == 0.12:
            self.detector.suspicion_lambda = self.detection_base_rate
        return self


class InteractionConfig(BaseModel):
    """Interaction graph / communication settings."""

    topology: Literal["none", "static", "graph", "dynamic"] = "none"
    communication: Literal["none", "direct", "graph"] = "none"
    communication_cost: float = 0.5
    allow_coalitions: bool = False


class ExperimentConfig(BaseModel):
    """Top-level experiment specification."""

    name: str
    description: str = ""
    seed: int = 0
    episodes: int = 10
    max_turns: int = 20  # global cap on turns per episode

    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    interaction: InteractionConfig = Field(default_factory=InteractionConfig)
    agents: list[AgentConfig] = Field(default_factory=list)

    # ESR requires paired isolation runs; when true the controller
    # also runs each population in isolation to estimate P(failure|isolation).
    measure_esr: bool = False

    output_dir: str = "runs"

    @model_validator(mode="after")
    def _check_agents(self) -> "ExperimentConfig":
        if not self.agents:
            raise ValueError("experiment must define at least one agent group")
        roles = {a.role for a in self.agents}
        if "attacker" not in roles or "defender" not in roles:
            raise ValueError("experiment needs at least one attacker and one defender group")
        return self

    def expand_agents(self) -> list[dict[str, Any]]:
        """Expand agent groups into concrete per-agent specs with unique ids."""
        out: list[dict[str, Any]] = []
        for group in self.agents:
            for i in range(group.count):
                agent_id = group.id_prefix if group.count == 1 else f"{group.id_prefix}_{i:02d}"
                out.append({"agent_id": agent_id, "group": group})
        return out

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)

    def config_hash(self) -> str:
        """Stable hash of the config for reproducibility bookkeeping."""
        import hashlib
        import json

        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
