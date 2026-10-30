"""Unit tests for core metrics, including CEP and ESR."""

from coevsec.metrics import (
    adaptation_gain,
    attack_success_rate,
    behavioral_novelty,
    coalition_rate,
    coevolutionary_pressure,
    deception_rate,
    detection_rate,
    emergent_security_risk,
    mean_cost,
    recovery_time,
    strategy_diversity,
)


def test_asr_and_dr():
    eps = [
        {"attempted_attack": True, "attack_success": True, "detected": True},
        {"attempted_attack": True, "attack_success": False, "detected": True},
        {"attempted_attack": True, "attack_success": True, "detected": False},
    ]
    assert attack_success_rate(eps) == 2 / 3
    assert detection_rate(eps) == 2 / 3


def test_costs_and_recovery():
    eps = [
        {"attacker_cost": 2.0, "defender_cost": 1.0, "detection_cost": 0.5, "recovery_time": 3},
        {"attacker_cost": 4.0, "defender_cost": 3.0, "detection_cost": 1.5, "recovery_time": 1},
    ]
    assert mean_cost(eps, "attacker_cost") == 3.0
    assert recovery_time(eps) == 2.0


def test_adaptation_gain():
    assert abs(adaptation_gain(0.8, 0.5) - 0.3) < 1e-9


def test_coalition_and_deception():
    eps = [
        {"coalitions": [["a", "b"]], "deceptive_actions": 1, "total_actions": 10},
        {"coalitions": [], "deceptive_actions": 1, "total_actions": 10},
    ]
    assert coalition_rate(eps) == 0.5
    assert deception_rate(eps) == 0.1


def test_strategy_diversity_and_cep():
    sigs_a = [{"stealth": 0.2}, {"stealth": 0.5}, {"stealth": 0.9}]
    sigs_d = [{"vigilance": 0.1}, {"vigilance": 0.4}, {"vigilance": 0.8}]
    assert strategy_diversity(sigs_a) > 0
    cep = coevolutionary_pressure(sigs_a, sigs_d)
    assert cep > 0
    static = [{"stealth": 0.3}] * 5
    assert coevolutionary_pressure(static, static) == 0.0


def test_novelty_and_esr():
    behaviours = [{"recon"}, {"recon"}, {"stealth"}, {"stealth", "deception"}]
    assert behavioral_novelty(behaviours) == 0.75
    assert emergent_security_risk(0.4, 0.2) == 2.0
    assert emergent_security_risk(0.0, 0.0) == 1.0
    assert emergent_security_risk(0.1, 0.0) == float("inf")
