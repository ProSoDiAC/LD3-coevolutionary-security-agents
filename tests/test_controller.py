"""End-to-end controller tests on the simulator (heuristic + mock LLM)."""

from __future__ import annotations

from coevsec.core.config import AgentConfig, ExperimentConfig, LLMConfig, PolicyConfig
from coevsec.core.types import AdaptationLevel
from coevsec.experiments.controller import ExperimentController
from coevsec.environment.k8s.environment import K8sEnvironment
from coevsec.core.config import EnvironmentConfig
import random


def _tiny(name: str, adaptation: AdaptationLevel, kind: str = "heuristic",
          provider: str = "mock", measure_esr: bool = False) -> ExperimentConfig:
    llm = LLMConfig(provider=provider)
    return ExperimentConfig(
        name=name,
        seed=11,
        episodes=4,
        max_turns=12,
        measure_esr=measure_esr,
        output_dir="runs",
        agents=[
            AgentConfig(
                id_prefix="attacker", role="attacker", adaptation=adaptation,
                policy=PolicyConfig(kind=kind, stealth=0.3), llm=llm,
            ),
            AgentConfig(
                id_prefix="defender", role="defender", adaptation=adaptation,
                policy=PolicyConfig(kind=kind, stealth=0.4), llm=llm,
            ),
        ],
    )


def test_static_experiment_runs_and_writes_summary(tmp_path):
    cfg = _tiny("test_static", AdaptationLevel.STATIC)
    cfg.output_dir = str(tmp_path)
    result = ExperimentController(cfg).run()
    assert result["episodes"] == 4
    assert 0.0 <= result["attack_success_rate"] <= 1.0
    assert 0.0 <= result["detection_rate"] <= 1.0
    assert "coevolutionary_pressure" in result
    run_dir = tmp_path / f"{cfg.name}_{cfg.config_hash()}"
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "trajectories.jsonl").exists()
    assert (run_dir / "config.json").exists()


def test_persistent_experiment_produces_cep_signal(tmp_path):
    cfg = _tiny("test_persist", AdaptationLevel.PERSISTENT)
    cfg.output_dir = str(tmp_path)
    cfg.episodes = 8
    result = ExperimentController(cfg).run()
    # Persistent agents are allowed to change strategy; CEP is defined and finite.
    assert result["coevolutionary_pressure"] >= 0.0


def test_mock_llm_policy_path(tmp_path):
    cfg = _tiny("test_llm_mock", AdaptationLevel.EPISODIC, kind="llm", provider="mock")
    cfg.output_dir = str(tmp_path)
    cfg.episodes = 2
    cfg.max_turns = 6
    result = ExperimentController(cfg).run()
    assert result["episodes"] == 2


def test_esr_isolation_run(tmp_path):
    cfg = _tiny("test_esr", AdaptationLevel.EPISODIC, measure_esr=True)
    cfg.output_dir = str(tmp_path)
    cfg.interaction.communication = "direct"
    cfg.interaction.topology = "static"
    result = ExperimentController(cfg).run()
    assert "esr" in result


def test_k8s_backend_without_provision():
    """K8sEnvironment with provision=False is a drop-in for the simulator."""
    cfg = EnvironmentConfig(backend="k8s", hosts=2, params={"provision": False})
    env = K8sEnvironment(cfg, random.Random(0))
    from coevsec.core.types import Role, Action

    env.register_agent("att", Role.ATTACKER)
    env.reset(0)
    obs = env.observe("att")
    assert obs.data["backend"] == "k8s"
    result = env.step(Action(agent_id="att", tool="recon", params={}))
    assert result.success or "error" in result.detail
    env.teardown()
