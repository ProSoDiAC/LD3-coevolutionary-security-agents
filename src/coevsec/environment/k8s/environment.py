"""K8s cyber-range environment.

Delegates the co-evolutionary game logic and noisy detection model to the
simulator (so metrics are identical and comparable), and, when
``environment.params.provision`` is true and the tooling is present, mirrors the
salient state changes (isolation) onto a real ``kind`` cluster. This lets the
same emergent strategies be re-validated against real Kubernetes objects without
changing agents, metrics or telemetry.
"""

from __future__ import annotations

import random

from coevsec.core.config import EnvironmentConfig
from coevsec.core.types import Action, Observation, Role, StepResult
from coevsec.environment.base import CyberEnvironment
from coevsec.environment.sim.environment import SimEnvironment


class K8sEnvironment(CyberEnvironment):
    def __init__(self, cfg: EnvironmentConfig, rng: random.Random | None = None) -> None:
        self.cfg = cfg
        self._sim = SimEnvironment(cfg, rng)
        self.tools = self._sim.tools
        self.provision = bool(cfg.params.get("provision", False))
        self._provisioner = None
        if self.provision:
            from coevsec.environment.k8s.provisioner import KindProvisioner, tooling_available

            ok, missing = tooling_available()
            if not ok:
                raise RuntimeError(f"K8s provisioning requested but tools missing: {missing}")
            self._provisioner = KindProvisioner(
                cluster_name=cfg.params.get("cluster_name", "coevsec"),
                namespace=cfg.params.get("namespace", "range"),
            )
            self._provisioner.create_cluster()

    @property
    def cost(self) -> dict:
        return self._sim.cost

    def register_agent(self, agent_id: str, role: Role) -> None:
        self._sim.register_agent(agent_id, role)

    def reset(self, seed: int, profile_seed: int | None = None) -> None:
        self._sim.reset(seed, profile_seed)
        if self._provisioner is not None:
            self._provisioner.deploy(self.cfg.hosts)

    def observe(self, agent_id: str) -> Observation:
        obs = self._sim.observe(agent_id)
        obs.data["backend"] = "k8s"
        return obs

    def step(self, action: Action) -> StepResult:
        result = self._sim.step(action)
        if self._provisioner is not None and action.tool == "isolate" and result.success:
            host = action.params.get("host")
            if host:
                # Game outcome (result.success) stays simulator-driven by design
                # (see paper Sec. VII-E); record the real cluster's own outcome
                # for auditing rather than silently discarding it.
                applied = self._provisioner.isolate_host(host)
                result.detail = dict(result.detail)
                result.detail["k8s_isolate_applied"] = applied
        return result

    def end_turn(self):
        return self._sim.end_turn()

    def state_snapshot(self) -> dict:
        snap = self._sim.state_snapshot()
        snap["backend"] = "k8s"
        return snap

    def is_compromised(self, asset: str) -> bool:
        return self._sim.is_compromised(asset)

    def objective_met(self) -> bool:
        return self._sim.objective_met()

    def episode_summary(self) -> dict:
        return self._sim.episode_summary()

    def teardown(self) -> None:
        if self._provisioner is not None:
            self._provisioner.teardown(delete_cluster=self.cfg.params.get("delete_cluster", False))
