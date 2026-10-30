"""Regression tests for defects found in the 2026-09 review: overlapping
episode seeds across run seeds, the adaptive attacker abandoning the only
gateway after one failed exploit, the dead false-positive branch of the
defender reflection, and logging of carried-over (inter-episode) parameters."""

from __future__ import annotations

import random

from coevsec.agents.attacker.policy import AttackerHeuristicPolicy
from coevsec.agents.shared.memory import NullMemory, PersistentMemory
from coevsec.core.config import ExperimentConfig
from coevsec.core.types import Action, AdaptationLevel, Observation, StepResult, episode_seed
from coevsec.environment.sim.tools import build_tool_registry
from coevsec.experiments.controller import ExperimentController


def test_episode_seeds_of_neighbouring_runs_are_disjoint():
    a = {episode_seed(100, ep) for ep in range(40)}
    b = {episode_seed(110, ep) for ep in range(40)}
    assert not a & b


def _gateway_obs(failed_once: bool) -> Observation:
    data = {
        "role": "attacker",
        "turn": 3,
        "footholds": [],
        "credentials": [],
        "known_hosts": [{
            "host": "gw", "layer": "gateway", "reachable": True, "isolated": False,
            "known_vulns": ["gw_v0"], "foothold": False,
        }],
    }
    return Observation(agent_id="attacker", turn=3, data=data, text="")


def _after_failed_gateway_exploit(**kw) -> AttackerHeuristicPolicy:
    p = AttackerHeuristicPolicy(
        "attacker", AdaptationLevel.EPISODIC.value, random.Random(0), stealth=0.0, **kw,
    )
    failed = Action(agent_id="attacker", tool="exploit", params={"host": "gw"}, target="gw")
    obs = _gateway_obs(True)
    p.observe_result(failed, StepResult(observation=obs, success=False, detail={}))
    return p


def test_adaptive_attacker_retries_the_only_gateway():
    p = _after_failed_gateway_exploit()
    tools = build_tool_registry().for_role(p.role)
    action = p.choose(_gateway_obs(True), tools, NullMemory())
    assert action.tool == "exploit" and action.params["host"] == "gw"


def test_abandon_flag_reproduces_the_legacy_stall():
    p = _after_failed_gateway_exploit(abandon_failed_hosts=True)
    tools = build_tool_registry().for_role(p.role)
    action = p.choose(_gateway_obs(True), tools, NullMemory())
    assert action.tool == "recon"


def test_false_positive_reflection_is_reachable():
    mem = PersistentMemory()
    mem.end_episode({"role": "defender", "attack_success": False, "false_positives": 1})
    assert "false positives" in mem.reflections[0]


def test_persistent_reflection_names_transferable_host():
    mem = PersistentMemory()
    mem.end_episode({"role": "attacker", "attack_success": True, "detected": False,
                     "service_foothold_host": "svc_01"})
    assert "svc_01" in mem.reflections[0]


def _tiny_cfg(tmp_path) -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "name": "t", "seed": 7, "episodes": 3, "max_turns": 6, "output_dir": str(tmp_path),
        "agents": [
            {"id_prefix": "attacker", "role": "attacker", "adaptation": "persistent",
             "policy": {"kind": "heuristic", "stealth": 0.3}},
            {"id_prefix": "defender", "role": "defender", "adaptation": "persistent",
             "policy": {"kind": "heuristic", "stealth": 0.4}},
        ],
    })


def test_episode_summary_logs_false_positives_and_carried_params(tmp_path):
    ctl = ExperimentController(_tiny_cfg(tmp_path))
    res = ctl._run_config(7, writer=None)
    for ep in res.episodes:
        assert "false_positives" in ep
        assert "base_stealth" in ep["attacker_carried_params"]
        assert "base_vigilance" in ep["defender_carried_params"]


def test_engagement_metrics_separate_abstention_from_evasion():
    from coevsec.metrics import detection_rate, detection_rate_given_engagement, engagement_rate

    eps = [
        {"foothold_gained": True, "detected": True},
        {"foothold_gained": False, "detected": False},
    ]
    assert detection_rate(eps) == 0.5
    assert engagement_rate(eps) == 0.5
    assert detection_rate_given_engagement(eps) == 1.0


def test_coalition_detection_runs(tmp_path):
    cfg = ExperimentConfig.model_validate({
        "name": "coal", "seed": 3, "episodes": 2, "max_turns": 6, "output_dir": str(tmp_path),
        "interaction": {"topology": "static", "communication": "direct", "allow_coalitions": True},
        "agents": [
            {"id_prefix": "attacker", "role": "attacker", "count": 2, "adaptation": "episodic",
             "policy": {"kind": "heuristic"}},
            {"id_prefix": "defender", "role": "defender", "adaptation": "episodic",
             "policy": {"kind": "heuristic"}},
        ],
    })
    res = ExperimentController(cfg)._run_config(3, writer=None)
    assert all("coalitions" in ep for ep in res.episodes)


def test_first_detection_reason_and_isolations_are_logged(tmp_path):
    res = ExperimentController(_tiny_cfg(tmp_path))._run_config(7, writer=None)
    for ep in res.episodes:
        assert ep["first_detection_reason"] in {None, "automatic", "investigation", "decoy"}
        assert (ep["first_detection_reason"] is None) == (not ep["detected"])
        assert isinstance(ep["isolated_hosts"], list)


def test_defender_channel_ablation_flags():
    import random as _r

    from coevsec.agents.defender.policy import DefenderHeuristicPolicy

    p = DefenderHeuristicPolicy("defender", "static", _r.Random(0), disable_investigate=True,
                                disable_decoy=True)
    obs = Observation(agent_id="defender", turn=1, text="", data={
        "hosts": [{"host": "gw", "layer": "gateway", "monitored": 1.0, "suspicion": 3.0}],
        "detections": 0,
    })
    tools = build_tool_registry().for_role(p.role)
    for _ in range(50):
        assert p.choose(obs, tools, NullMemory()).tool not in {"investigate", "deploy_decoy"}


def _svc_obs(creds: bool) -> Observation:
    known = [
        {"host": "gw", "layer": "gateway", "reachable": True, "isolated": False,
         "known_vulns": ["gw_v0"], "foothold": True},
        {"host": "svc_00", "layer": "service", "reachable": True, "isolated": False,
         "known_vulns": ["svc_00_v0"], "foothold": True},
        {"host": "svc_01", "layer": "service", "reachable": True, "isolated": False,
         "known_vulns": [], "foothold": False},
        {"host": "svc_02", "layer": "service", "reachable": True, "isolated": False,
         "known_vulns": [], "foothold": False},
    ]
    data = {"role": "attacker", "turn": 5, "footholds": ["gw", "svc_00"],
            "credentials": ["svc_00_key"] if creds else [], "known_hosts": known}
    return Observation(agent_id="attacker", turn=5, data=data, text="")


def test_autofill_keeps_a_valid_model_host():
    from coevsec.agents.shared.policy import _autofill_params

    assert _autofill_params("probe_service", {"host": "svc_02"}, _svc_obs(True))["host"] == "svc_02"
    # An owned or unknown host is still repaired.
    assert _autofill_params("probe_service", {"host": "svc_00"}, _svc_obs(True))["host"] == "svc_01"


def test_hints_offer_recon_when_datastore_unknown_with_credentials():
    from coevsec.agents.shared.policy import _legal_attacker_hints

    assert '"recon"' in _legal_attacker_hints(_svc_obs(True))[0]


def test_kill_chain_phase_uses_tool_names(tmp_path):
    res = ExperimentController(_tiny_cfg(tmp_path))._run_config(7, writer=None)
    assert max(ep["kill_chain_phase_at_end"] for ep in res.episodes) >= 2


def _vuln_vector(regime, ep, episodes=8, n_profiles=2):
    import random

    from coevsec.core.config import ExperimentConfig
    from coevsec.core.types import episode_seed, profile_seed
    from coevsec.environment import make_environment

    cfg = ExperimentConfig.model_validate({
        "name": "t", "seed": 100, "episodes": episodes,
        "environment": {"regime": regime, "n_profiles": n_profiles},
        "agents": [
            {"id_prefix": "attacker", "role": "attacker", "policy": {"kind": "heuristic"}},
            {"id_prefix": "defender", "role": "defender", "policy": {"kind": "heuristic"}},
        ],
    })
    env = make_environment(cfg.environment, random.Random(100))
    env.register_agent("attacker", "attacker")
    env.register_agent("defender", "defender")
    ps = None if regime == "iid" else profile_seed(100, regime, ep, episodes, n_profiles, 0.5)
    env.reset(episode_seed(100, ep), ps)
    return tuple(v.difficulty for h in sorted(env.world.hosts)
                 for v in env.world.hosts[h].vulnerabilities)


def test_environment_regimes_control_recurrence():
    # IID: every episode's structure is distinct.
    assert _vuln_vector("iid", 0) != _vuln_vector("iid", 2)
    # Recurring (period 2): episodes 0, 2, 4 share structure; 1 differs.
    assert _vuln_vector("recurring", 0) == _vuln_vector("recurring", 2) == _vuln_vector("recurring", 4)
    assert _vuln_vector("recurring", 0) != _vuln_vector("recurring", 1)
    # Drift (at 0.5 of 8): first half constant, second half a different structure.
    assert _vuln_vector("drift", 0) == _vuln_vector("drift", 3)
    assert _vuln_vector("drift", 0) != _vuln_vector("drift", 4)
    assert _vuln_vector("drift", 4) == _vuln_vector("drift", 7)
