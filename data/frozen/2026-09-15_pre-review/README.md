# Frozen pre-review results (2026-09-15)

Aggregates produced by the pre-review code (September 2026), before the
review. They are kept only as the reference for the frozen vs. regenerated
comparison (`scripts/compare_frozen.py` writes
`data/paper_runs/frozen_comparison.json`). They are not used for any current
result.

| File | Content |
|---|---|
| `ladder_iso.json` | 3x3 heuristic ladder, 6 seeds (reproduced exactly by the pre-review code during the review) |
| `detector_ablation.json` | Detector and pacing ablations on L1/L0 and L0/L0 |
| `crossplay.json` | Checkpoint cross-play matrix |
| `campaign_v2.json` | Language-model campaign, B3 to B5 aggregates |
| `cot_pilot_summary.json` | Reasoning-field pilot |

These results are superseded because of defects in that code, described in
`docs/REVIEW_REPORT.md`:
- overlapping episode seeds across runs;
- an adaptive attacker stranded at the single gateway after one failed
  exploit;
- a hybrid controller that overrode valid model choices and never hinted
  reconnaissance.

The remaining files of the original bundle were removed: the legacy ladder
(identical to the isomorphic one), mixed-version trajectories, Kubernetes
checks, the manifest and derived statistics. `run_dir` fields refer to the
original machine.
