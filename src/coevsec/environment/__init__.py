"""Cyber environment interface and backends.

A single :class:`CyberEnvironment` interface is implemented by two interchangeable
backends so the same agents and metrics run on both:

    sim  fast, seedable in-memory state machine for high-volume evolution runs
    k8s  kind-backed cyber range for high-fidelity validation
"""

from coevsec.environment.base import CyberEnvironment

__all__ = ["CyberEnvironment"]


def make_environment(cfg, rng):
    """Factory: build the environment backend named in the config."""
    from coevsec.core.config import EnvironmentConfig

    assert isinstance(cfg, EnvironmentConfig)
    if cfg.backend == "sim":
        from coevsec.environment.sim.environment import SimEnvironment

        return SimEnvironment(cfg, rng)
    if cfg.backend == "k8s":
        from coevsec.environment.k8s.environment import K8sEnvironment

        return K8sEnvironment(cfg, rng)
    raise ValueError(f"unknown environment backend: {cfg.backend}")
