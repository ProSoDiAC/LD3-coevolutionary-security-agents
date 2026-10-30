"""Regression test for the cross-play eval loop no longer re-loading the
checkpoint every episode (which froze PERSISTENT and EPISODIC into
identical behaviour for the length of the eval window)."""

from __future__ import annotations

from coevsec.agents.attacker.policy import AttackerHeuristicPolicy
from coevsec.core.config import ExperimentConfig
from coevsec.experiments.crossplay import run_with_checkpoints


def _cfg(attacker_adaptation: str) -> ExperimentConfig:
    raw = {
        "name": "xplay_test",
        "seed": 1,
        "episodes": 1,
        "max_turns": 10,
        "environment": {"backend": "sim", "hosts": 2, "vulnerabilities_per_host": 2},
        "agents": [
            {"id_prefix": "attacker", "role": "attacker", "adaptation": attacker_adaptation,
             "policy": {"kind": "heuristic", "stealth": 0.3}},
            {"id_prefix": "defender", "role": "defender", "adaptation": "static",
             "policy": {"kind": "heuristic", "stealth": 0.4}},
        ],
    }
    return ExperimentConfig.model_validate(raw)


def test_checkpoint_loaded_once_not_per_episode(monkeypatch):
    calls = []
    original = AttackerHeuristicPolicy.load_checkpoint

    def counting_load(self, state):
        calls.append(state)
        return original(self, state)

    monkeypatch.setattr(AttackerHeuristicPolicy, "load_checkpoint", counting_load)
    run_with_checkpoints(
        _cfg("persistent"), seed=1,
        att_checkpoint={"base_stealth": 0.7, "stealth": 0.7, "aggression": 0.6},
        def_checkpoint=None, eval_episodes=4,
    )
    assert len(calls) == 1


def test_persistent_checkpoint_keeps_adapting_across_eval_episodes(monkeypatch):
    """end_episode() must run between eval episodes so a PERSISTENT
    checkpoint's base_stealth can keep moving, unlike a frozen replay."""
    seen_base_stealth = []
    original_end_episode = AttackerHeuristicPolicy.end_episode

    def recording_end_episode(self, summary):
        original_end_episode(self, summary)
        seen_base_stealth.append(self.base_stealth)

    monkeypatch.setattr(AttackerHeuristicPolicy, "end_episode", recording_end_episode)
    run_with_checkpoints(
        _cfg("persistent"), seed=1,
        att_checkpoint={"base_stealth": 0.5, "stealth": 0.5, "aggression": 0.6},
        def_checkpoint=None, eval_episodes=5,
    )
    assert len(seen_base_stealth) == 5
    # At least one PERSISTENT update must have moved base_stealth away from
    # the loaded checkpoint value -- a frozen replay would keep it at 0.5.
    assert any(abs(v - 0.5) > 1e-9 for v in seen_base_stealth)
