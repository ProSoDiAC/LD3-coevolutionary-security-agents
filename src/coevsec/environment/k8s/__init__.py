"""Kubernetes cyber-range backend.

Implements the same :class:`~coevsec.environment.base.CyberEnvironment` interface
as the simulator. The co-evolutionary game logic and
detection model are shared with the simulator so metrics are directly comparable;
this backend additionally (optionally) provisions a real ``kind`` cluster with
Helm-deployed decoy services and enforces isolation via NetworkPolicies, for
high-fidelity validation of strategies that emerged in simulation.
"""

from coevsec.environment.k8s.environment import K8sEnvironment

__all__ = ["K8sEnvironment"]
