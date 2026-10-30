# coevsec: when does memory help adaptive cyber agents?

`coevsec` is a research framework for evaluating attacker and defender agents
in repeated cyber interaction. It asks **when persistent memory helps an
adaptive agent, when it is useless, and when it is harmful**, treating the
adaptation regime of each side as a controlled factor. It accompanies the
ITASEC27 paper *When Does Memory Help Adaptive Cyber Agents? A Controlled
Two-Sided Evaluation of LLM Attackers and Defenders* (Italian Conference on
Cybersecurity, Siena, 2027). The ITASEC27 manuscript and its submission files
are maintained separately from this public code repository.

## What the paper does, in plain words

Attackers and defenders in a simulated network fight over many episodes. Each
can be given a persistent memory: a record of what happened in earlier
episodes that it reads before acting. The common assumption is that memory
makes an agent better. The paper tests that assumption with controlled
experiments and finds that the sign of the effect depends on the role, the
environment and the kind of agent.

Two kinds of agents are compared. Heuristic agents carry a few scalar
parameters. Hybrid agents are driven by a language model (qwen2.5:14b) with a
heuristic fallback, and their memory is a reflection written after each
episode that names the host which recurs. The main experiment is a 2x2 crossing
attacker memory with defender memory, 20 seeds per cell, 30 episodes per run,
on a detector arena with headroom (not saturated), repeated under two
environment regimes: a recurring latent structure and independent episodes.

## The protocol in one paragraph

Each role runs at one of three adaptation levels (**L0** static, **L1**
episodic, **L2** persistent). These are crossed, at fixed policy class, with
the **environment regime** (independent episodes, a recurring latent
structure, or a mid-run drift) and with a detector **arena** that is either
saturated or has headroom. Runs use disjoint episode seeds so seed-level
aggregates are independent and cells sharing a seed are paired, and the
detection rate is decomposed into the **engagement rate** and the **detection
rate given engagement**, so evasion is not confused with abstention. Hybrid
language-model agents report the share of decisions taken by their heuristic
fallback. Effects are seed-paired differences with Benjamini-Hochberg
adjustment within each family of tests.

## Current results (see the manuscript for statistics)

Language-model 2x2 (DR = detection rate, ASR = attack success rate; all
p-values BH-adjusted; the contrast is the full persistent-memory package, L2,
against a memoryless L0 agent):

| Effect | Recurring regime | Independent episodes |
|---|---|---|
| Defender memory on DR | +0.056 (p=.002) | +0.083 (p<.001) |
| Defender memory on ASR | -0.069 (p=.002) | -0.061 (p<.001) |
| Attacker memory on its own ASR | -0.182 (p<.001) | -0.174 (p<.001) |
| Interaction on DR | +0.018 (n.s.) | +0.063 (p=.021) |

- **Memory is not generically positive.** It helps the defender and harms the
  attacker that uses it, and the effects persist when the vulnerability
  profile no longer recurs.
- **Defender pathway measured.** The memory defender monitors or investigates
  the recurring service host in 93% of episodes, against 0% without memory,
  and the gain comes from less evasion with engagement flat.
- **Attacker pathway not explained.** The memory attacker does not become more
  cautious; it waits slightly less (-0.034). The harm is reported without a
  claimed isolated mechanism.
- **Heuristic agents.** With a scalar-parameter policy, memory never helps in
  any regime or arena, which points to policy capacity as a precondition (an
  interpretation, since moving to the language-model agent changes several
  things at once).
- **Interaction:** no evidence of one in the recurring regime; a positive
  detection interaction appears under independent episodes.
- **Two cautions.** The engagement decomposition exposes a spurious evasion
  effect of about 0.47 from an attacker that merely stops engaging, and
  provenance accounting shows a heuristic fallback executes a large share of
  each hybrid agent's steps, so outcomes belong to the composed agent.

Limits: one language-model family and one simulated arena; the service host
named by the reflection is stable across episodes in both regimes, so whether
a stable host is enough, and whether the memory component alone (apart from
the package) drives the effect, are future work.

## Repository layout

| Path | Content |
|---|---|
| `src/coevsec/` | Python package: agents, simulator, optional Kubernetes backend, metrics, analysis, telemetry |
| `scripts/` | Study drivers, data export and artifact generation (see below) |
| `configs/examples/` | Example configurations for single runs (1v1, 10v10 population, Kubernetes backend); the reported studies generate their configurations in the study scripts |
| `data/paper_runs/` | Versioned data behind every reported number (aggregates, per-episode records, manifest) |
| `data/frozen/` | Pre-review aggregates kept only to document the frozen vs. regenerated comparison |
| `repro/` | Reproduction guide, environment lock, gate rules, exact prompts |
| `tests/` | Unit and regression tests |

The manuscript bundle is maintained separately. Every reported table and
number can be regenerated from `data/paper_runs/` with
`scripts/make_paper_artifacts.py`; its output directory is configurable with
`--paper`.

## Setup

```bash
uv venv
uv pip install -e ".[dev,analysis]"
.venv/bin/python -m pytest -q
```

Ollama is needed only for the language-model campaign; copy `.env.example`
to `.env` and set `COEVSEC_OLLAMA_BASE_URL`. To check that the shipped data and
tables are consistent without re-running any study:

```bash
.venv/bin/python scripts/make_paper_artifacts.py --data data/paper_runs --paper paper
git status --short paper data   # no change expected
```

## Running experiments

A single experiment is a YAML file (run inside the activated virtual environment):

```bash
coevsec run --config configs/examples/1v1_static.yaml
coevsec analyze --run runs/<name>_<hash>
```

The studies reported in the manuscript are driven by scripts:

| Script | Study |
|---|---|
| `scripts/regime_study.py`, `analyze_regime.py` | Adaptation x environment-regime x arena factorial and its exact-inference analysis |
| `scripts/llm_campaign.py`, `analyze_llm2x2.py` | 2x2 attacker-memory x defender-memory language-model campaign and analysis |
| `scripts/ladder_ci.py` | Saturated-arena 3x3 ladder (`--attacker-param abandon_failed_hosts=true` for the abandonment variant) |
| `scripts/detector_ablation.py` | Detector, defender-channel, pacing and abandonment ablations |
| `scripts/crossplay_matrix.py` | Checkpoint cross-play |
| `scripts/k8s_validate.py` | Kubernetes backend: enforcement probe and simulator comparison |
| `scripts/campaign_2day.py` | Saturated-arena language-model control campaign (Ollama) |
| `scripts/export_paper_data.py` | Export runs to `data/paper_runs/` with digests |
| `scripts/make_paper_artifacts.py` | Statistics, LaTeX tables, figure data and result macros |
| `scripts/compare_frozen.py` | Frozen (pre-review) vs. regenerated results |

See [`repro/REPRODUCE.md`](repro/REPRODUCE.md) for the exact command
sequence, seeding and runtimes. The Kubernetes backend
(`scripts/k8s_validate.py`, requires Docker, `kind`, `kubectl` and `helm`)
enforces the defender's isolations on a provisioned kind cluster through
network policies while game logic stays in the simulator; on the ladder cells
it reproduces every simulated episode exactly.

## Citation

```bibtex
@software{dagati_coevsec,
  author = {D'Agati, Luca and Tricomi, Giuseppe and Mirto, Fabio Orazio and Longo, Francesco and Merlino, Giovanni},
  title  = {coevsec: When Does Memory Help Adaptive Cyber Agents?},
  year   = {2026},
  url    = {https://github.com/ProSoDiAC/LD3-coevolutionary-security-agents}
}
```

## License

MIT, see [`LICENSE`](LICENSE). Never commit `.env`; raw experiment outputs
under `runs/` are git-ignored and may contain model prompts and trajectories.
