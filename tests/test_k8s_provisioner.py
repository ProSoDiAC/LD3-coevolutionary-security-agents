"""Provisioning helpers stay side-effect free unless invoked."""

from coevsec.environment.k8s.provisioner import tooling_available, CHART_DIR, KIND_CONFIG


def test_chart_and_kind_config_exist():
    assert CHART_DIR.exists()
    assert (CHART_DIR / "Chart.yaml").exists()
    assert (CHART_DIR / "templates" / "gateway.yaml").exists()
    assert (CHART_DIR / "templates" / "datastore.yaml").exists()
    assert KIND_CONFIG.exists()


def test_tooling_available_reports_tuple():
    ok, missing = tooling_available()
    assert isinstance(ok, bool)
    assert isinstance(missing, list)
    # kind is not required to be installed for the rest of the framework.
    if not ok:
        assert "kind" in missing or "kubectl" in missing or "helm" in missing


def test_isolation_policies_exclude_isolated_pods():
    """NetworkPolicies are additive: isolation must exclude pods from the allow
    policy and deny them explicitly, not add a deny policy next to an allow."""
    from pathlib import Path

    import yaml

    chart = Path(__file__).resolve().parents[1] / "src/coevsec/environment/k8s/chart/templates"
    docs = [d for d in yaml.safe_load_all((chart / "networkpolicy.yaml").read_text()) if d]
    allow = next(d for d in docs if d["metadata"]["name"] == "default-allow-in-namespace")
    deny = next(d for d in docs if d["metadata"]["name"] == "deny-isolated")
    excl = {"key": "coevsec/isolated", "operator": "NotIn", "values": ["true"]}
    assert excl in allow["spec"]["podSelector"]["matchExpressions"]
    assert deny["spec"]["podSelector"]["matchLabels"] == {"coevsec/isolated": "true"}
    assert "ingress" not in deny["spec"] and "egress" not in deny["spec"]
