"""Tests for the simulator environment: capabilities, partial observability, kill chain."""

from __future__ import annotations

import random

from coevsec.core.config import EnvironmentConfig
from coevsec.core.types import Action, Role
from coevsec.environment.sim.environment import SimEnvironment


def _env(seed: int = 0) -> SimEnvironment:
    env = SimEnvironment(EnvironmentConfig(hosts=2, vulnerabilities_per_host=2), random.Random(seed))
    env.register_agent("att", Role.ATTACKER)
    env.register_agent("def", Role.DEFENDER)
    env.reset(seed)
    return env


def test_reset_is_deterministic():
    a = _env(7)
    b = _env(7)
    assert a.state_snapshot()["hosts"].keys() == b.state_snapshot()["hosts"].keys()
    assert [h.layer for h in a.world.hosts.values()] == [h.layer for h in b.world.hosts.values()]


def test_partial_observability_attacker_starts_blind():
    env = _env(1)
    obs = env.observe("att")
    assert obs.data["known_hosts"] == []
    assert obs.data["footholds"] == []


def test_recon_reveals_gateway():
    env = _env(1)
    result = env.step(Action(agent_id="att", tool="recon", params={}))
    assert result.success
    names = {h["host"] for h in result.observation.data["known_hosts"]}
    assert "gw" in names


def test_invalid_tool_for_role_is_rejected():
    env = _env(1)
    result = env.step(Action(agent_id="att", tool="isolate", params={"host": "gw"}))
    assert result.success is False
    assert "invalid tool" in result.detail["error"]


def test_invalid_params_rejected():
    env = _env(1)
    result = env.step(Action(agent_id="def", tool="monitor", params={}))
    assert result.success is False


def test_scripted_kill_chain_can_exfiltrate():
    """Drive the environment with a scripted attacker to prove the objective is reachable."""
    env = SimEnvironment(
        EnvironmentConfig(hosts=1, vulnerabilities_per_host=1), random.Random(0)
    )
    env.register_agent("att", Role.ATTACKER)
    env.register_agent("def", Role.DEFENDER)
    env.reset(0)
    # Make every vuln trivial so the scripted path is deterministic.
    for h in env.world.hosts.values():
        for v in h.vulnerabilities:
            v.difficulty = 0.0

    env.step(Action(agent_id="att", tool="recon", params={}))
    env.step(Action(agent_id="att", tool="probe_service", params={"host": "gw"}))
    gw_vuln = env.world.hosts["gw"].vulnerabilities[0].vid
    env.step(Action(agent_id="att", tool="exploit", params={"host": "gw", "vuln": gw_vuln}))
    assert "att" in env.world.hosts["gw"].footholds

    env.step(Action(agent_id="att", tool="recon", params={}))
    svc = "svc_00"
    env.step(Action(agent_id="att", tool="probe_service", params={"host": svc}))
    svc_vuln = env.world.hosts[svc].vulnerabilities[0].vid
    env.step(Action(agent_id="att", tool="exploit", params={"host": svc, "vuln": svc_vuln}))
    env.step(Action(agent_id="att", tool="harvest_credentials", params={"host": svc}))
    env.step(Action(agent_id="att", tool="lateral_move", params={"host": "datastore"}))
    result = env.step(Action(agent_id="att", tool="exfiltrate", params={"host": "datastore"}))
    assert result.success
    assert env.objective_met()
    summary = env.episode_summary()
    assert summary["attack_success"] is True


def test_defender_isolate_evicts_foothold():
    env = _env(3)
    env.world.hosts["gw"].footholds.add("att")
    env.world.hosts["gw"].suspicion = 5.0
    env.world.hosts["gw"].monitor_level = 1.0
    result = env.step(Action(agent_id="def", tool="isolate", params={"host": "gw"}))
    assert result.success
    assert env.world.hosts["gw"].isolated
    assert not env.world.hosts["gw"].footholds


def test_decoy_triggers_detection():
    env = _env(2)
    env.step(Action(agent_id="def", tool="deploy_decoy", params={"layer": "gateway"}))
    decoy = next(h for h in env.world.hosts.values() if h.is_decoy)
    env.knowledge["att"]["hosts"].add(decoy.name)
    env.step(Action(agent_id="att", tool="probe_service", params={"host": decoy.name}))
    assert env.world.detections >= 1
