# Reproducing the reported results

Every table, figure and number in the manuscript is generated from the
versioned data in `data/paper_runs/`, and that data is produced by the
scripts below. The chain is:

```
install -> run studies (runs/review/) -> export (data/paper_runs/) -> artifacts (tables, figures, macros) -> LaTeX
```

## 1. Install

Python 3.11+ and [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv
uv pip install -e ".[dev,analysis]"
.venv/bin/python -m pytest -q          # 75 tests, ~2 s
```

`repro/environment.lock` pins the exact versions used (Python 3.12.12).

## 2. Heuristic studies (CPU only, about 15 minutes in total)

```bash
PY=.venv/bin/python
$PY scripts/ladder_ci.py --seeds 100,110,120,130,140,150 --episodes 40 --out runs/review/ladder_iso.json
$PY scripts/ladder_ci.py --seeds 100,110,120,130,140,150 --episodes 40 \
    --attacker-param abandon_failed_hosts=true --out runs/review/ladder_iso_abandon.json
$PY scripts/ladder_ci.py --seeds $(seq -s, 100 10 290) --episodes 40 --out runs/review/ladder_iso_20seeds.json
$PY scripts/detector_ablation.py --seeds 100,110,120,130,140,150 --episodes 40 --out runs/review/detector.json
$PY scripts/crossplay_matrix.py --seeds 100,110,120,130,140,150 --episodes 40 \
    --checkpoint-ep 20 --eval-episodes 20 --out runs/review/crossplay.json
# Adaptation x environment-regime x arena factorial (1080 runs, a few minutes)
$PY scripts/regime_study.py --seeds 100-290 --episodes 40 --workers 8
$PY scripts/analyze_regime.py
```

These runs are deterministic: re-running them reproduces the exported
episode records exactly. Compare `episodes_canonical_sha256` in
`data/paper_runs/manifest.json`, a digest over records re-serialised with
sorted keys. Runs produced before the writer switched to sorted keys can
differ in key order only.

## 3. Language-model campaign (GPU, Ollama)

Requires an Ollama server with `qwen2.5:14b` (digest
`7cdf5a0187d5…22f6`, Q4_K_M, Ollama 0.17.7 in our runs).

```bash
export COEVSEC_OLLAMA_BASE_URL=http://localhost:11434
# Primary 2x2 (attacker memory x defender memory) on headroom + recurring,
# 4 cells x 20 seeds = 80 runs of 30 episodes; checkpointed and resumable.
$PY scripts/llm_campaign.py --seeds 100-290 --episodes 30 --workers 1
$PY scripts/analyze_llm2x2.py            # cells, effects (BH-adjusted), interaction CIs
$PY scripts/analyze_llm_mechanisms.py    # behavioural mechanisms from the trajectories
# The same 2x2 under independent episodes (adds 80 runs; reported in the paper).
# On a GPU with <12 GB free, COEVSEC_OLLAMA_NUM_GPU=40 offloads ~20% of layers to CPU.
$PY scripts/llm_campaign.py --seeds 100-290 --episodes 30 --workers 1 --iid-full
# Saturated-arena control campaign (symmetric B3-B5, asymmetric, attribution,
# reasoning pilot), 6 seeds.
$PY scripts/campaign_2day.py --out-dir runs/review/campaign --workers 2
```

Phases: `1_hybrid_ci` (B3 to B5, both roles hybrid, 6 seeds x 30 episodes),
`7_cot_pilot` (reasoning field on/off), `2b_ablation_modes` (model only vs
heuristic only) and `1b_asymmetric`. The campaign is checkpointed
(`suite_state.json`) and skips completed jobs when restarted; `--phase` runs a
subset. With an idle L40 a primary run takes about 30 minutes; the whole
campaign takes roughly a day. Sampling is seeded per call, but GPU kernels are
not guaranteed to be bitwise deterministic, so re-runs can differ slightly in
individual decisions.

Seeds per phase: B3 to B5 use 520, 530, ..., 570; the asymmetric cells use
seed + 1 (hybrid attacker) and seed + 2 (hybrid defender); the reasoning pilot
uses seed + 3 for both arms (paired); the attribution ablation uses 530, 550,
570 for the model-only arm and 531, 551, 571 for the heuristic-only arm, so
those two arms are not paired.

## 3b. Kubernetes backend (optional, about 10 minutes)

Requires Docker, `kind` (v0.33.0 used), `kubectl` (v1.37.1) and `helm`
(v3.22.0). The script creates a kind cluster named `coevsec` if missing,
checks that isolation is enforced by the cluster's network policies, then
runs three ladder cells on both backends and compares the episode records:

```bash
$PY scripts/k8s_validate.py --seeds 100,110,120,130,140,150 --episodes 40
```

On LXC hosts (e.g. Proxmox containers) the kind kubelet fails with
`open /dev/kmsg: no such file or directory`. Create the device link
(`ln -s /dev/console /dev/kmsg`) and create the cluster with `/dev/kmsg`
mounted into the nodes, adding to each node of
`src/coevsec/environment/k8s/kind-cluster.yaml`:

```yaml
    extraMounts:
      - {hostPath: /dev/kmsg, containerPath: /dev/kmsg}
```

## 4. Export and artifacts

```bash
$PY scripts/export_paper_data.py --runs runs/review --out data/paper_runs
$PY scripts/make_paper_artifacts.py --data data/paper_runs --paper paper
$PY scripts/compare_frozen.py          # frozen vs. regenerated table
```

`make_paper_artifacts.py` writes `data/paper_runs/paper_stats.json` and, under
the selected `--paper` directory, every table, figure data file and
`results_macros.tex`. The manuscript bundle is maintained separately from this
repository; when placed at `paper/`, it uses these generated artifacts.

## 5. Manuscript

The ITASEC27 CEUR-WS manuscript source is distributed separately. After
placing that bundle in `paper/`, regenerate its tables and macros with:

```bash
PY=.venv/bin/python
$PY scripts/make_paper_artifacts.py --data data/paper_runs --paper paper
```

Build `paper/main.tex` with the CEUR-WS `ceurart` class and the LaTeX packages
listed in its preamble; these TeX dependencies are not vendored in this repo.

## Seeding

| Stream | Seed |
|---|---|
| Episode environment | `1000 * run_seed + episode` (disjoint across runs) |
| Agent policy RNG | `sha256(f"{run_seed}:{agent_id}")[:8]` |
| LLM sampling (per call) | stream seeded by `sha256(f"llm:{run_seed}:{agent_id}")[:8]` |

## Data layout

| Path | Content |
|---|---|
| `data/paper_runs/*.json` | Aggregates of each study (ladder, replication, abandonment variant, ablations, cross-play, Kubernetes check, campaign state) |
| `data/paper_runs/episodes/*.jsonl.gz` | One record per episode of every run |
| `data/paper_runs/manifest.json` | Code content digest, file digests, per-run config, episode and executed-action digests, action counts |
| `data/paper_runs/paper_stats.json` | Every statistic used in the manuscript |
| `data/paper_runs/frozen_comparison.json` | Frozen vs. regenerated key results |
| `data/frozen/2026-09-15_pre-review/` | Pre-review aggregates used only for that comparison (see its README) |
