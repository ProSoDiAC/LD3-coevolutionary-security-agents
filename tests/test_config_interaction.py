"""Config loading, taxonomy, interaction graph, coalitions, mock LLM."""

from __future__ import annotations

from pathlib import Path

from coevsec.core.config import ExperimentConfig
from coevsec.core.taxonomy import is_emergent
from coevsec.core.types import AdaptationLevel
from coevsec.interaction import InteractionGraph, detect_coalitions
from coevsec.llm import make_backend
from coevsec.llm.base import LLMBackend


ROOT = Path(__file__).resolve().parents[1]


def test_e1_config_loads():
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "examples" / "1v1_static.yaml")
    assert cfg.name == "example_1v1_static"
    specs = cfg.expand_agents()
    assert len(specs) == 2
    assert specs[0]["group"].adaptation == AdaptationLevel.STATIC
    assert cfg.config_hash()


def test_e8_expands_twenty_agents():
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "examples" / "population_10v10.yaml")
    assert len(cfg.expand_agents()) == 20
    assert cfg.measure_esr is True


def test_emergence_definition():
    assert is_emergent("deception", {"reconnaissance"})
    assert not is_emergent("reconnaissance", {"reconnaissance"})


def test_graph_static_fully_connected():
    g = InteractionGraph(topology="static")
    g.add_agent("a", "attacker")
    g.add_agent("b", "defender")
    g.initialise()
    assert g.can_send("a", "b", "direct")
    assert g.can_send("a", "b", "graph")
    snap = g.snapshot(0)
    assert snap["edges"] == 2


def test_graph_role_ring():
    g = InteractionGraph(topology="graph")
    for i in range(4):
        g.add_agent(f"a{i}", "attacker")
    g.initialise()
    assert g.can_send("a0", "a1", "graph")
    assert not g.can_send("a0", "a2", "graph")  # ring of 4: opposite node is not a neighbour


def test_coalition_requires_reciprocal_same_role():
    roles = {"a0": "attacker", "a1": "attacker", "d0": "defender"}
    counts = {("a0", "a1"): 3, ("a1", "a0"): 3, ("a0", "d0"): 5, ("d0", "a0"): 5}
    coals = detect_coalitions(counts, roles, min_reciprocal=2)
    assert len(coals) == 1
    assert coals[0] == {"a0", "a1"}


def test_mock_backend_emits_tool_schema():
    from coevsec.core.config import LLMConfig

    backend = make_backend(LLMConfig(provider="mock", model="mock"))
    assert isinstance(backend, LLMBackend)
    schema = {
        "type": "object",
        "properties": {"tool": {"type": "string", "enum": ["recon", "wait"]}},
        "required": ["tool"],
    }
    resp = backend.complete("sys", "user", json_schema=schema)
    assert resp.tool_call is not None
    assert resp.tool_call.tool in {"recon", "wait"}
    assert resp.total_tokens > 0
