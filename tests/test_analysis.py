"""Analysis engine: metric table, dynamics classification, emergence tagging."""

from __future__ import annotations

import json
from pathlib import Path

from coevsec.analysis.engine import analyze_run


def test_analyze_run_writes_report(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    episodes = [
        {
            "episode": 0, "attack_success": False, "detected": True,
            "attacker_cost": 4, "defender_cost": 3, "detection_cost": 2,
            "attacker_strategy": {"stealth": 0.3},
            "defender_strategy": {"vigilance": 0.4},
            "coalitions": [],
        },
        {
            "episode": 1, "attack_success": True, "detected": True,
            "attacker_cost": 5, "defender_cost": 4, "detection_cost": 2,
            "attacker_strategy": {"stealth": 0.5},
            "defender_strategy": {"vigilance": 0.6},
            "coalitions": [["attacker_00", "attacker_01"]],
        },
        {
            "episode": 2, "attack_success": True, "detected": False,
            "attacker_cost": 5, "defender_cost": 4, "detection_cost": 1,
            "attacker_strategy": {"stealth": 0.55},
            "defender_strategy": {"vigilance": 0.7},
            "coalitions": [["attacker_00", "attacker_01"]],
        },
        {
            "episode": 3, "attack_success": False, "detected": True,
            "attacker_cost": 3, "defender_cost": 5, "detection_cost": 3,
            "attacker_strategy": {"stealth": 0.7},
            "defender_strategy": {"vigilance": 0.85},
            "coalitions": [],
        },
    ]
    (run / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes))
    traj = [
        {"role": "attacker", "action": "recon"},
        {"role": "attacker", "action": "wait"},
        {"role": "defender", "action": "deploy_decoy"},
        {"role": "defender", "action": "monitor"},
    ]
    (run / "trajectories.jsonl").write_text("\n".join(json.dumps(t) for t in traj))
    (run / "config.json").write_text(json.dumps({
        "agents": [{"encoded_behaviours": ["reconnaissance", "adaptive_monitoring"]}]
    }))

    report = analyze_run(run)
    assert report["n_episodes"] == 4
    assert "metric_table" in report
    assert "strategy_dynamics" in report
    assert report["strategy_dynamics"]["classification"] in {
        "stable_equilibrium", "non_convergent", "converging", "insufficient_data",
    }
    assert "param_late_delta" in report["strategy_dynamics"]
    assert "deception" in report["emergence"]["candidate_emergent"]
    assert (run / "analysis" / "report.json").exists()
    assert report["coalition_graph"]["episodes_with_coalitions"] == 2
