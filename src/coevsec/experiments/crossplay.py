"""Cross-play evaluation: checkpoint policies from different runs."""

from __future__ import annotations

import random
from typing import Any

from coevsec.agents.factory import make_agent
from coevsec.agents.shared.agent import Agent
from coevsec.core.config import ExperimentConfig
from coevsec.core.types import Role, episode_seed
from coevsec.environment import make_environment
from coevsec.interaction import InteractionGraph, MessageBus
from coevsec.metrics import attack_success_rate, detection_rate, engagement_rate


def _policy_with_checkpoint(agent: Agent) -> Any:
    pol = getattr(agent.policy, "heuristic", agent.policy)
    return pol


def run_with_checkpoints(
    cfg: ExperimentConfig,
    seed: int,
    att_checkpoint: dict[str, float] | None,
    def_checkpoint: dict[str, float] | None,
    *,
    eval_episodes: int = 5,
) -> dict[str, Any]:
    """Run eval_episodes loading attacker/defender parameter checkpoints."""
    rng = random.Random(seed)
    env = make_environment(cfg.environment, rng)
    agents = [make_agent(spec["agent_id"], spec["group"], seed) for spec in cfg.expand_agents()]
    for a in agents:
        env.register_agent(a.agent_id, a.role)
        pol = _policy_with_checkpoint(a)
        if a.role == Role.ATTACKER and att_checkpoint and hasattr(pol, "load_checkpoint"):
            pol.load_checkpoint(att_checkpoint)
        if a.role == Role.DEFENDER and def_checkpoint and hasattr(pol, "load_checkpoint"):
            pol.load_checkpoint(def_checkpoint)

    bus = MessageBus(communication=cfg.interaction.communication)
    graph = InteractionGraph(topology=cfg.interaction.topology)
    for a in agents:
        graph.add_agent(a.agent_id, a.role.value)
    graph.initialise()

    episodes: list[dict] = []
    for ep in range(eval_episodes):
        ep_seed = episode_seed(seed, ep)
        env.reset(ep_seed)
        for a in agents:
            a.reset_episode()
        bus.reset_episode()

        for turn in range(cfg.max_turns):
            for agent in agents:
                if not agent.can_act():
                    continue
                obs = env.observe(agent.agent_id)
                action = agent.decide(obs, env.available_tools(agent.role))
                result = env.step(action)
                agent.learn(action, result)
            env.end_turn()
            if env.objective_met():
                break
        summary = env.episode_summary()
        summary["episode"] = ep
        episodes.append(summary)
        # Let PERSISTENT checkpoints keep adapting across eval episodes (the
        # cross-episode update rule), rather than re-loading the checkpoint
        # every episode -- the latter freezes L1 and L2 into the same
        # behaviour for the length of this eval window. EPISODIC/STATIC
        # agents are unaffected: end_episode() is a no-op unless PERSISTENT,
        # and reset_episode() already restores base_stealth/base_vigilance
        # (which never changed) at the start of the next episode.
        for a in agents:
            a.end_episode(summary)

    env.teardown()
    return {
        "attack_success_rate": attack_success_rate(episodes),
        "detection_rate": detection_rate(episodes),
        "engagement_rate": engagement_rate(episodes),
        "episodes": len(episodes),
    }


def collect_checkpoints(cfg: ExperimentConfig, seed: int, at_episodes: tuple[int, ...]) -> dict[int, dict]:
    """Run full training and return checkpoints keyed by 1-based episode index."""
    from coevsec.experiments.controller import ExperimentController

    ctrl = ExperimentController(cfg, checkpoint_episodes=at_episodes)
    # Hack: run via internal loop with checkpoint capture
    result = ctrl._run_config(seed, writer=None)
    out: dict[int, dict] = {}
    for ckpt in result.checkpoints:
        ep = ckpt.get("episode", 0) + 1
        if ep not in at_episodes:
            # The controller's capture condition also fires one episode early
            # (0-based vs. 1-based ambiguity); only keep what was requested.
            continue
        out[ep] = {
            "attacker": ckpt.get("attacker") or ckpt.get("attacker_00"),
            "defender": ckpt.get("defender") or ckpt.get("defender_00"),
        }
    return out
