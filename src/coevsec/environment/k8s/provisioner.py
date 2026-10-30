"""Provisioning helpers for the K8s cyber range (kind + Helm).

Kept side-effect free unless explicitly invoked. All cluster mutations shell out
to ``kind``/``kubectl``/``helm`` and are guarded by :func:`tooling_available`,
so importing this module never touches a cluster.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

CHART_DIR = Path(__file__).parent / "chart"
KIND_CONFIG = Path(__file__).parent / "kind-cluster.yaml"


def tooling_available() -> tuple[bool, list[str]]:
    """Return (ok, missing_tools)."""
    missing = [t for t in ("kind", "kubectl", "helm") if shutil.which(t) is None]
    return (not missing, missing)


def _run(cmd: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)


class KindProvisioner:
    def __init__(self, cluster_name: str = "coevsec", namespace: str = "range") -> None:
        self.cluster_name = cluster_name
        self.namespace = namespace

    def cluster_exists(self) -> bool:
        try:
            out = _run(["kind", "get", "clusters"])
        except Exception:
            return False
        return self.cluster_name in out.stdout.split()

    def create_cluster(self) -> None:
        if self.cluster_exists():
            return
        cmd = ["kind", "create", "cluster", "--name", self.cluster_name]
        if KIND_CONFIG.exists():
            cmd += ["--config", str(KIND_CONFIG)]
        _run(cmd)

    def deploy(self, hosts: int) -> None:
        """Deploy the range chart: gateway, `hosts` services, and the data store."""
        _run(["helm", "upgrade", "--install", "range", str(CHART_DIR),
              "--namespace", self.namespace, "--create-namespace",
              "--set", f"services={hosts}"])
        # Helm only manages what its own chart templates; NetworkPolicies applied
        # by isolate_host() below are raw kubectl objects it doesn't know about
        # and won't clean up, so a stale isolation from a previous episode/seed
        # would otherwise silently persist against the freshly-recreated pods.
        self._clear_isolation_policies()

    def _clear_isolation_policies(self) -> None:
        """Lift every isolation left over from a previous episode or seed."""
        subprocess.run(
            ["kubectl", "label", "pods", "--all", "coevsec/isolated-",
             "--namespace", self.namespace],
            capture_output=True, text=True, timeout=60,
        )

    def isolate_host(self, host: str) -> bool:
        """Isolate a host by labelling its pods; the chart's policies then deny
        all traffic to and from them.

        Returns whether ``kubectl`` succeeded; callers should not treat
        isolation as having taken effect on the real cluster otherwise.
        """
        k8s_name = host.replace("_", "-")
        result = subprocess.run(
            ["kubectl", "label", "pods", "-l", f"app={k8s_name}", "coevsec/isolated=true",
             "--overwrite", "--namespace", self.namespace],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0 and "labeled" in result.stdout

    def reachable(self, src: str, dst: str, port: int = 8080, timeout_s: int = 3) -> bool:
        """TCP reachability from host ``src`` to host ``dst`` inside the cluster."""
        src_k8s, dst_k8s = src.replace("_", "-"), dst.replace("_", "-")
        ip = _run(["kubectl", "get", "svc", dst_k8s, "-n", self.namespace,
                   "-o", "jsonpath={.spec.clusterIP}"]).stdout.strip()
        pod = _run(["kubectl", "get", "pod", "-l", f"app={src_k8s}", "-n", self.namespace,
                    "-o", "jsonpath={.items[0].metadata.name}"]).stdout.strip()
        result = subprocess.run(
            ["kubectl", "exec", "-n", self.namespace, pod, "--",
             "/agnhost", "connect", f"{ip}:{port}", f"--timeout={timeout_s}s"],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0

    def teardown(self, delete_cluster: bool = False) -> None:
        if delete_cluster and self.cluster_exists():
            subprocess.run(["kind", "delete", "cluster", "--name", self.cluster_name],
                           capture_output=True, text=True, timeout=120)
        else:
            subprocess.run(["helm", "uninstall", "range", "--namespace", self.namespace],
                           capture_output=True, text=True, timeout=120)
