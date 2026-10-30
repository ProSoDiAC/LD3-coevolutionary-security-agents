"""Experiment controller: drives the co-evolutionary loop.

For each episode: reset -> interact (turn loop) -> outcome -> experience ->
strategy adaptation -> next episode. Fully seeded for reproducibility; every
step is logged to telemetry and per-episode metrics are accumulated.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from coevsec.agents.factory import make_agent
from coevsec.agents.shared.agent import Agent
from coevsec.core.config import ExperimentConfig
from coevsec.core.taxonomy import CATEGORY_TO_BEHAVIOURS
from coevsec.core.types import Action, Role, episode_seed, profile_seed
from coevsec.environment import make_environment
from coevsec.interaction import InteractionGraph, MessageBus, detect_coalitions
from coevsec.metrics import (
    attack_success_rate,
    attack_success_undetected_rate,
    behavioral_novelty,
    coalition_rate,
    coevolutionary_pressure,
    coevolutionary_pressure_attacker,
    coevolutionary_pressure_defender,
    cross_lagged_correlation,
    deception_rate,
    detection_rate,
    detection_rate_given_engagement,
    emergent_security_risk,
    engagement_rate,
    episode_shuffle_null,
    mean_cost,
    recovery_time,
    strategy_diversity,
)
from coevsec.telemetry import TelemetryWriter, TrajectoryRecord


@dataclass
class RunResult:
    episodes: list[dict] = field(default_factory=list)
    attacker_sigs: list[dict[str, float]] = field(default_factory=list)
    defender_sigs: list[dict[str, float]] = field(default_factory=list)
    per_episode_behaviours: list[set[str]] = field(default_factory=list)
    final_signatures: list[dict[str, float]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)


def _mean_signatures(sigs: list[dict[str, float]]) -> dict[str, float]:
    if not sigs:
        return {}
    keys: set[str] = set()
    for s in sigs:
        keys |= set(s)
    return {k: sum(s.get(k, 0.0) for s in sigs) / len(sigs) for k in keys}


class ExperimentController:
    def __init__(self, cfg: ExperimentConfig, *, checkpoint_episodes: tuple[int, ...] = ()) -> None:
        self.cfg = cfg
        self.checkpoint_episodes = checkpoint_episodes

    # ------------------------------------------------------------------ build
    def _build_agents(self, seed: int) -> list[Agent]:
        agents: list[Agent] = []
        for spec in self.cfg.expand_agents():
            agents.append(make_agent(spec["agent_id"], spec["group"], seed))
        return agents

    def _build(self, seed: int, interaction_override: str | None = None):
        rng = random.Random(seed)
        env = make_environment(self.cfg.environment, rng)
        agents = self._build_agents(seed)
        for a in agents:
            env.register_agent(a.agent_id, a.role)
        comm = interaction_override if interaction_override is not None else self.cfg.interaction.communication
        bus = MessageBus(communication=comm, cost=self.cfg.interaction.communication_cost)
        graph = InteractionGraph(topology=self.cfg.interaction.topology)
        for a in agents:
            graph.add_agent(a.agent_id, a.role.value)
        graph.initialise()
        return env, agents, bus, graph

    # -------------------------------------------------------------- run loop
    def _run_config(
        self, seed: int, writer: TelemetryWriter | None, interaction_override: str | None = None
    ) -> RunResult:
        env, agents, bus, graph = self._build(seed, interaction_override)
        result = RunResult()

        try:
            self._run_episodes(env, agents, bus, graph, seed, writer, result)
        finally:
            # Always release the environment (e.g. tear down a provisioned k8s
            # cluster) even if an episode raises mid-run.
            env.teardown()
        return result

    def _run_episodes(self, env, agents, bus, graph, seed, writer, result) -> None:
        roles = {a.agent_id: a.role.value for a in agents}
        env_cfg = self.cfg.environment
        for ep in range(self.cfg.episodes):
            ep_seed = episode_seed(seed, ep)
            # IID keeps the historical single-stream seeding (profile_seed=None);
            # other regimes draw structure from a recurring profile stream.
            prof_seed = None
            if env_cfg.regime != "iid":
                prof_seed = profile_seed(seed, env_cfg.regime, ep, self.cfg.episodes,
                                         env_cfg.n_profiles, env_cfg.drift_at)
            env.reset(ep_seed, prof_seed)
            for a in agents:
                a.reset_episode()
            bus.reset_episode()

            deceptive_actions = 0
            total_actions = 0
            ep_categories: dict[str, set[str]] = {"attacker": set(), "defender": set()}

            for turn in range(self.cfg.max_turns):
                for agent in agents:
                    if not agent.can_act():
                        continue
                    obs = env.observe(agent.agent_id)
                    peers = [
                        {"id": other.agent_id, "role": other.role.value}
                        for other in agents
                        if other.agent_id != agent.agent_id
                        and graph.can_send(agent.agent_id, other.agent_id, bus.communication)
                    ]
                    obs.data["peers"] = peers
                    if bus.enabled():
                        inbox = bus.inbox_text(agent.agent_id)
                        if inbox:
                            obs.text = obs.text + "\n\n" + inbox
                    action = agent.decide(obs, env.available_tools(agent.role))

                    if action.tool == "communicate":
                        recipient = str(action.params.get("to", ""))
                        if bus.enabled() and graph.can_send(agent.agent_id, recipient, bus.communication):
                            bus.send(agent.agent_id, recipient,
                                     str(action.params.get("content", "")), turn)
                            env.cost[agent.agent_id] = env.cost.get(agent.agent_id, 0.0) + bus.cost
                        else:
                            action = Action(agent_id=agent.agent_id, tool="wait",
                                            rationale="communication not permitted")

                    result_step = env.step(action)
                    agent.learn(action, result_step)

                    total_actions += 1
                    cat = _category(env, action.tool)
                    ep_categories[agent.role.value].add(cat)
                    if _is_deceptive(agent.role, action, obs.data):
                        deceptive_actions += 1

                    if writer is not None:
                        writer.log_step(TrajectoryRecord(
                            episode=ep, turn=turn, agent=agent.agent_id, role=agent.role.value,
                            observation=obs.text[:500], action=action.tool, params=action.params,
                            target=action.target, reason=action.rationale,
                            result=str(result_step.detail)[:200], success=result_step.success,
                            strategy=action.strategy, memory_update="",
                            cost=result_step.cost,
                            reasoning=str(getattr(agent.policy, "last_reasoning", ""))[:500],
                        ))

                env.end_turn()
                if self.cfg.interaction.topology == "dynamic":
                    graph.set_edges_from_counts(bus.counts)
                graph.snapshot(turn)

                if env.objective_met():
                    break

            summary = env.episode_summary()
            summary["episode"] = ep
            summary["deceptive_actions"] = deceptive_actions
            summary["total_actions"] = total_actions

            coalitions = []
            if self.cfg.interaction.allow_coalitions:
                coalitions = [sorted(c) for c in detect_coalitions(bus.counts, roles)]
            summary["coalitions"] = coalitions
            if graph.snapshots:
                summary["graph_density"] = graph.snapshots[-1]["density"]
                summary["graph_edges"] = graph.snapshots[-1]["edges"]

            for a in agents:
                a.end_episode(summary)

            # Parameters carried into the next episode (after the L2 update);
            # the strategy signatures below hold end-of-episode values instead.
            for role in (Role.ATTACKER, Role.DEFENDER):
                carried = [
                    pol.checkpoint()
                    for a in agents if a.role == role
                    for pol in [getattr(a.policy, "heuristic", a.policy)]
                    if hasattr(pol, "checkpoint")
                ]
                if carried:
                    summary[f"{role.value}_carried_params"] = _mean_signatures(carried)

            att_sig = _mean_signatures(
                [a.strategy_signature() for a in agents if a.role == Role.ATTACKER])
            def_sig = _mean_signatures(
                [a.strategy_signature() for a in agents if a.role == Role.DEFENDER])
            result.attacker_sigs.append(att_sig)
            result.defender_sigs.append(def_sig)

            behaviours: set[str] = set()
            for role_cats in ep_categories.values():
                for c in role_cats:
                    behaviours |= CATEGORY_TO_BEHAVIOURS.get(c, set())
            result.per_episode_behaviours.append(behaviours)

            summary["attacker_strategy"] = att_sig
            summary["defender_strategy"] = def_sig
            hybrid_stats = {}
            for a in agents:
                if hasattr(a.policy, "gate_stats"):
                    hybrid_stats[a.agent_id] = a.policy.gate_stats()
            if hybrid_stats:
                summary["hybrid_gate_stats"] = hybrid_stats
                att_gate = hybrid_stats.get("attacker") or hybrid_stats.get("attacker_00")
                if att_gate:
                    total = att_gate.get("llm_steps", 0) + att_gate.get("heuristic_steps", 0)
                    summary["attacker_gate_frac"] = (
                        att_gate.get("heuristic_steps", 0) / total if total else 0.0
                    )
            result.episodes.append(summary)
            if (ep + 1) in self.checkpoint_episodes:
                ckpt: dict[str, Any] = {"episode": ep}
                for a in agents:
                    pol = getattr(a.policy, "heuristic", a.policy)
                    if hasattr(pol, "checkpoint"):
                        ckpt[a.agent_id] = pol.checkpoint()
                result.checkpoints.append(ckpt)
            if writer is not None:
                writer.log_episode(summary)

        result.final_signatures = [a.strategy_signature() for a in agents]

    # --------------------------------------------------------------- public
    def run(self) -> dict[str, Any]:
        writer = TelemetryWriter(self.cfg.output_dir, self.cfg.name,
                                 self.cfg.config_hash(), self.cfg.seed)
        cfg_dump = self.cfg.model_dump(mode="json")
        cfg_dump["rng_audit"] = {
            "experiment_seed": self.cfg.seed,
            "agent_seed_formula": "sha256(seed:agent_id)[:8]",
        }
        writer.write_config(cfg_dump)

        try:
            main = self._run_config(self.cfg.seed, writer)
            aggregate = self._aggregate(main)

            # ESR requires a paired isolation run, communication off.
            if self.cfg.measure_esr:
                isolation = self._run_config(self.cfg.seed, writer=None, interaction_override="none")
                asr_int = attack_success_rate(main.episodes)
                asr_iso = attack_success_rate(isolation.episodes)
                aggregate["esr"] = emergent_security_risk(asr_int, asr_iso)
                aggregate["asr_isolation"] = asr_iso

            writer.write_summary(aggregate)
        finally:
            writer.close()
        aggregate["run_dir"] = str(writer.run_dir)
        return aggregate

    def _aggregate(self, r: RunResult) -> dict[str, Any]:
        eps = r.episodes
        xl = cross_lagged_correlation(r.attacker_sigs, r.defender_sigs)
        out = {
            "name": self.cfg.name,
            "episodes": len(eps),
            "attack_success_rate": attack_success_rate(eps),
            "attack_success_undetected_rate": attack_success_undetected_rate(eps),
            "detection_rate": detection_rate(eps),
            "engagement_rate": engagement_rate(eps),
            "detection_rate_given_engagement": detection_rate_given_engagement(eps),
            "attack_success_count": sum(1 for e in eps if e.get("attack_success")),
            "detection_count": sum(1 for e in eps if e.get("detected")),
            "attempted_attack_count": sum(1 for e in eps if e.get("attempted_attack", True)),
            "attacker_cost": mean_cost(eps, "attacker_cost"),
            "defender_cost": mean_cost(eps, "defender_cost"),
            "detection_cost": mean_cost(eps, "detection_cost"),
            "recovery_time": recovery_time(eps),
            "coalition_rate": coalition_rate(eps),
            "deception_rate": deception_rate(eps),
            "attacker_strategy_diversity": strategy_diversity(r.attacker_sigs),
            "defender_strategy_diversity": strategy_diversity(r.defender_sigs),
            "behavioral_novelty": behavioral_novelty(r.per_episode_behaviours),
            "coevolutionary_pressure": coevolutionary_pressure(r.attacker_sigs, r.defender_sigs),
            "cep_attacker": coevolutionary_pressure_attacker(r.attacker_sigs),
            "cep_defender": coevolutionary_pressure_defender(r.defender_sigs),
            "cep_shuffle_null": episode_shuffle_null(r.attacker_sigs, r.defender_sigs),
            "cross_lagged_def_to_att": xl["def_to_att"],
            "cross_lagged_att_to_def": xl["att_to_def"],
            "mean_graph_density": (
                sum(e.get("graph_density", 0.0) for e in eps) / len(eps) if eps else 0.0
            ),
        }
        gate = _summarize_hybrid_gate_stats(eps)
        if gate:
            out["hybrid_gate_stats"] = gate
        term = _termination_counts(eps)
        if term:
            out["termination_counts"] = term
        if eps:
            out["mean_turns_used"] = sum(e.get("turns_used", 0) for e in eps) / len(eps)
            ttd = [e["time_to_detect"] for e in eps if e.get("time_to_detect") is not None]
            out["mean_time_to_detect"] = sum(ttd) / len(ttd) if ttd else None
            out["mean_kill_chain_phase"] = sum(e.get("kill_chain_phase_at_end", 0) for e in eps) / len(eps)
        return out


def _termination_counts(episodes: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in episodes:
        r = e.get("termination_reason", "unknown")
        counts[r] = counts.get(r, 0) + 1
    return counts


def _summarize_hybrid_gate_stats(episodes: list[dict]) -> dict[str, float]:
    """Sum per-episode hybrid gate counters for attacker and defender."""
    sides = {
        "attacker": {"llm": 0.0, "heur": 0.0, "illegal": 0.0, "stall": 0.0, "progress": 0.0},
        "defender": {"llm": 0.0, "heur": 0.0, "illegal": 0.0, "stall": 0.0, "progress": 0.0},
    }
    found = False
    for ep in episodes:
        for agent_id, stats in (ep.get("hybrid_gate_stats") or {}).items():
            side = "attacker" if "attacker" in agent_id else "defender"
            found = True
            sides[side]["llm"] += float(stats.get("llm_steps", 0.0))
            sides[side]["heur"] += float(stats.get("heuristic_steps", stats.get("gate_interventions", 0.0)))
            sides[side]["illegal"] += float(stats.get("gate_illegal", 0.0))
            sides[side]["stall"] += float(stats.get("gate_stall", 0.0))
            sides[side]["progress"] += float(stats.get("gate_progress", 0.0))
    if not found:
        return {}
    att_total = sides["attacker"]["llm"] + sides["attacker"]["heur"]
    def_total = sides["defender"]["llm"] + sides["defender"]["heur"]
    return {
        "attacker_llm_steps": sides["attacker"]["llm"],
        "attacker_heuristic_steps": sides["attacker"]["heur"],
        "attacker_gate_frac": sides["attacker"]["heur"] / att_total if att_total else 0.0,
        "attacker_gate_illegal": sides["attacker"]["illegal"],
        "attacker_gate_stall": sides["attacker"]["stall"],
        "attacker_gate_progress": sides["attacker"]["progress"],
        "defender_llm_steps": sides["defender"]["llm"],
        "defender_heuristic_steps": sides["defender"]["heur"],
        "defender_gate_frac": sides["defender"]["heur"] / def_total if def_total else 0.0,
        "defender_gate_illegal": sides["defender"]["illegal"],
        "defender_gate_stall": sides["defender"]["stall"],
        "defender_gate_progress": sides["defender"]["progress"],
    }


def _category(env, tool_name: str) -> str:
    t = env.tools.get(tool_name)
    return t.category if t else "misc"


def _is_deceptive(role: Role, action: Action, obs_data: dict) -> bool:
    if role == Role.DEFENDER and action.tool == "deploy_decoy":
        return True
    if role == Role.ATTACKER and action.tool == "wait" and obs_data.get("footholds"):
        return True
    return False


def run_experiment(config_path: str) -> dict[str, Any]:
    cfg = ExperimentConfig.from_yaml(config_path)
    return ExperimentController(cfg).run()
