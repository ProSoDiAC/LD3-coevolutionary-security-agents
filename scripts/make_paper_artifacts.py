#!/usr/bin/env python3
"""Statistics, LaTeX tables, figure data and result macros for the manuscript.

Reads only the exported data in ``data/paper_runs`` (see
``scripts/export_paper_data.py``) and writes:

* ``data/paper_runs/paper_stats.json``: every number used in the manuscript;
* ``<paper>/figures/generated/*.tex|*.dat``: tables, figure bodies and
  ``results_macros.tex`` (``\\res{key}`` expands to a value from paper_stats).

Usage:
  python scripts/make_paper_artifacts.py [--data data/paper_runs] [--paper paper]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

LEVEL = {"static": "L0", "adaptive": "L1", "persistent": "L2"}
CELLS = [
    "static_static", "static_adaptive", "static_persistent",
    "adaptive_static", "adaptive_adaptive", "adaptive_persistent",
    "persistent_static", "persistent_adaptive", "persistent_persistent",
]
SHORT = {"static": "S", "adaptive": "A", "persistent": "P"}
LLM_CELLS = [("b3_static", "B3", "L0"), ("b4_episodic", "B4", "L1"), ("b5_persistent", "B5", "L2")]


def short(cell: str) -> str:
    a, d = cell.split("_")
    return f"{SHORT[a]}/{SHORT[d]}"


def levels(cell: str) -> str:
    a, d = cell.split("_")
    return f"{LEVEL[a]}/{LEVEL[d]}"


# ----------------------------------------------------------------- statistics
def mean_ci(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci95": float("nan")}
    m = sum(xs) / n
    if n == 1:
        return {"n": 1, "mean": m, "ci95": 0.0}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return {"n": n, "mean": m, "sd": sd, "ci95": float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)}


def boot_ci(diffs: list[float], n_boot: int = 10000) -> tuple[float, float]:
    rng = random.Random(42)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def paired(a: list[float], b: list[float]) -> dict:
    """Seed-paired t-test, d_z = mean(diff)/sd(diff) and a bootstrap CI."""
    assert len(a) == len(b) and len(a) >= 2, (len(a), len(b))
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    if sd < 1e-12:
        # Identical differences on every seed: the t statistic is undefined.
        # With a constant non-zero difference the exact two-sided sign-flip
        # permutation p-value is 2 / 2^n.
        t, p, dz = (0.0, 1.0, 0.0) if abs(m) < 1e-12 else (
            math.copysign(math.inf, m), 2.0 / 2 ** n, math.copysign(math.inf, m))
    else:
        res = stats.ttest_rel(a, b)
        t, p, dz = float(res.statistic), float(res.pvalue), m / sd
    lo, hi = boot_ci(d)
    return {"n": n, "mean_a": sum(a) / n, "mean_b": sum(b) / n, "diff": m, "t": t, "p": p,
            "d_z": dz, "boot_lo": lo, "boot_hi": hi}


def bh(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, prev = [1.0] * m, 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        adj[i] = min(prev, 1.0)
    return adj


# -------------------------------------------------------------------- loading
def load(data: Path, label: str) -> dict | None:
    p = data / f"{label}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_episodes(data: Path, label: str) -> dict[str, list[dict]]:
    p = data / "episodes" / f"{label}.jsonl.gz"
    out: dict[str, list[dict]] = defaultdict(list)
    if p.exists():
        with gzip.open(p, "rt") as fh:
            for line in fh:
                rec = json.loads(line)
                out[rec["run"]].append(rec["episode"])
    return out


def run_metrics(eps: list[dict]) -> dict:
    n = len(eps)
    eng = [e for e in eps if e.get("foothold_gained")]
    heur = sum((e.get("hybrid_gate_stats") or {}).get("attacker", {}).get("heuristic_steps", 0) for e in eps)
    llm = sum((e.get("hybrid_gate_stats") or {}).get("attacker", {}).get("llm_steps", 0) for e in eps)
    return {
        "asr": sum(bool(e.get("attack_success")) for e in eps) / n,
        "dr": sum(bool(e.get("detected")) for e in eps) / n,
        "engagement": len(eng) / n,
        "dr_engaged": (sum(bool(e.get("detected")) for e in eng) / len(eng)) if eng else float("nan"),
        "gate_frac": heur / (heur + llm) if heur + llm else float("nan"),
        # Share of detected episodes whose first detection was automatic, and
        # share of successful attacks that were also detected.
        "src_auto": (sum(e.get("first_detection_reason") == "automatic" for e in eps)
                     / max(1, sum(bool(e.get("detected")) for e in eps))),
        "succ_detected": (sum(bool(e.get("attack_success")) and bool(e.get("detected")) for e in eps)
                          / sum(bool(e.get("attack_success")) for e in eps))
        if any(e.get("attack_success") for e in eps) else float("nan"),
        "episodes": n,
    }


# -------------------------------------------------------------------- formats
def f3(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def f2(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def fci(c: dict) -> str:
    if c["n"] == 0 or math.isnan(c["mean"]):
        return "n/a"
    return f"${c['mean']:.3f}{{\\pm}}{c['ci95']:.3f}$"


def fp(p: float) -> str:
    if p is None or math.isnan(p):
        return "n/a"
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def fsigned(x: float) -> str:
    return f"{x:+.3f}".replace("+", "+") if abs(x) >= 0.0005 else "0.000"


# ------------------------------------------------------------------- analysis
class Results:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.stats: dict = {}

    def put(self, key: str, value: str) -> None:
        self.values[key] = value


def ladder_block(res: Results, lad: dict, tag: str) -> dict:
    cells = lad["cells"]
    out = {}
    for cell in CELLS:
        runs = cells[cell]["runs"]
        row = {k: mean_ci([r.get(k) for r in runs]) for k in
               ("asr", "dr", "engagement", "dr_engaged", "cep", "cep_adj")}
        out[cell] = row
        for k, c in row.items():
            res.put(f"{tag}.{short(cell)}.{k}", f3(c["mean"]))
            res.put(f"{tag}.{short(cell)}.{k}.ci", f3(c["ci95"]))
    return out


def ladder_series(lad: dict, cell: str, key: str, seeds: list[int] | None = None) -> list[float]:
    by = {r["seed"]: r for r in lad["cells"][cell]["runs"]}
    seeds = sorted(by) if seeds is None else seeds
    assert set(seeds) <= set(by), (cell, sorted(set(seeds) - set(by)))
    return [float(by[s][key]) for s in seeds]


# Confirmatory family (BH within it); E1' is a mechanism check reported
# without adjustment because it shares its differences with H1'.
LADDER_TESTS = [
    ("H1", "adaptive_static", "static_static", "dr", "L1 attacker vs.\\ L0 (DR)"),
    ("H1'", "persistent_static", "static_static", "dr", "L2 attacker vs.\\ L0 (DR)"),
    ("H2", "static_adaptive", "static_static", "dr", "L1 defender vs.\\ L0 (DR)"),
    ("H2'", "static_persistent", "static_static", "dr", "L2 defender vs.\\ L0 (DR)"),
    ("E1'", "persistent_static", "static_static", "engagement", "L2 attacker vs.\\ L0 (engagement)"),
    ("C", "persistent_persistent", "static_static", "cep", "L2/L2 vs.\\ L0/L0 (CEP)"),
]


def ladder_tests(lad: dict, seeds: list[int] | None = None) -> list[dict]:
    out = []
    for hid, a, b, key, label in LADDER_TESTS:
        r = paired(ladder_series(lad, a, key, seeds), ladder_series(lad, b, key, seeds))
        out.append({"id": hid, "label": label, "a": a, "b": b, "metric": key,
                    "family": "mechanism" if hid == "E1'" else "ladder", **r})
    return out


def bh_by_family(tests: list[dict]) -> list[float]:
    """BH within each family; mechanism checks keep their raw p-value."""
    q = [t["p"] for t in tests]
    for fam in {t["family"] for t in tests} - {"mechanism"}:
        idx = [i for i, t in enumerate(tests) if t["family"] == fam]
        for i, v in zip(idx, bh([tests[i]["p"] for i in idx])):
            q[i] = v
    return q


def campaign_runs(data: Path) -> dict[str, dict]:
    """job_id -> per-run metrics recomputed from the episode records."""
    camp = load(data, "campaign")
    if camp is None:
        return {}
    eps = load_episodes(data, "campaign")
    manifest_runs = json.loads((data / "manifest.json").read_text())["runs"].get("campaign", {})
    out = {}
    for job, r in (camp.get("raw") or camp.get("results") or {}).items():
        if r.get("status") != "ok" or not r.get("run_dir"):
            continue
        name = Path(r["run_dir"]).name
        if name in eps:
            m = run_metrics(eps[name])
            gate = (r.get("summary") or {}).get("hybrid_gate_stats") or {}
            if gate:
                m["gate_frac"] = gate.get("attacker_gate_frac", m["gate_frac"])
                m["def_gate_frac"] = gate.get("defender_gate_frac")
            m["cep"] = (r.get("summary") or {}).get("coevolutionary_pressure")
            # The seed actually run (job names carry the base seed, but some
            # phases offset it; pairing must use the configured seed).
            m["seed"] = ((manifest_runs.get(name) or {}).get("config") or {}).get("seed")
            out[job] = m
    return out


def by_seed(runs: dict[str, dict], prefix: str) -> dict[int, dict]:
    out = {}
    for job, m in runs.items():
        if job.startswith(prefix + "_s"):
            seed = m.get("seed")
            out[int(seed) if seed is not None else int(job.rsplit("_s", 1)[1])] = m
    return out


def llm_pair(runs, pa, pb, key) -> dict | None:
    a, b = by_seed(runs, pa), by_seed(runs, pb)
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    return paired([a[s][key] for s in seeds], [b[s][key] for s in seeds])


# --------------------------------------------------------------------- tables
def write(gen: Path, name: str, text: str) -> None:
    (gen / name).write_text(text)


def table(caption: str, label: str, colspec: str, header: str, rows: list[str],
          env: str = "table", size: str = "\\footnotesize", tabx: bool = False) -> str:
    begin = f"\\begin{{tabularx}}{{\\columnwidth}}{{{colspec}}}" if tabx else f"\\begin{{tabular}}{{{colspec}}}"
    end = "\\end{tabularx}" if tabx else "\\end{tabular}"
    body = f"{begin}\n\\toprule\n{header} \\\\\n\\midrule\n" + "\n".join(rows) + f"\n\\bottomrule\n{end}"
    # tabularx is already sized to \columnwidth; a natural-width tabular can be
    # wider than the text block (elsarticle's single column is narrower than
    # IEEEtran's two-column span), so cap it at \linewidth, shrinking only when
    # it would otherwise overflow.
    if not tabx:
        body = "\\begin{adjustbox}{max width=\\linewidth}\n" + body + "\n\\end{adjustbox}"
    return (
        f"\\begin{{{env}}}[!t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\n"
        f"\\setlength{{\\tabcolsep}}{{3pt}}\n{size}\n{body}\n\\end{{{env}}}\n"
    )


def ladder_table(block: dict, n_seeds: int, episodes: int) -> str:
    rows = []
    for cell in CELLS:
        c = block[cell]
        rows.append(
            f"{levels(cell)} & {fci(c['asr'])} & {fci(c['dr'])} & {fci(c['engagement'])} & "
            f"${c['dr_engaged']['mean']:.3f}$ & {fci(c['cep'])} \\\\"
        )
    return table(
        f"Isomorphic adaptation ladder (attacker/defender level; mean $\\pm$ 95\\% CI over "
        f"{n_seeds} seeds, {episodes} episodes each). Eng.: episodes with at least one foothold; "
        f"DR$\\mid$Eng.: detection rate among engaged episodes (seed mean).",
        "tab:ladder", "@{}lccccc@{}",
        "\\textbf{Cell} & \\textbf{ASR} & \\textbf{DR} & \\textbf{Eng.} & \\textbf{DR$\\mid$Eng.} & \\textbf{CEP}",
        rows, env="table*", size="\\footnotesize",
    )


def stats_table(tests: list[dict], p_adj: list[float], n: int) -> str:
    rows = []
    for t, q in zip(tests, p_adj):
        undefined = math.isinf(t["t"]) or (t["diff"] == 0 and t["boot_lo"] == t["boot_hi"])
        tv = "n/a" if undefined else f"${t['t']:.2f}$"
        dz = "n/a" if undefined else f"${t['d_z']:.2f}$"
        rows.append(
            f"{t['id']} & {t['label']} & ${t['diff']:+.3f}$ & "
            f"$[{t['boot_lo']:+.3f}, {t['boot_hi']:+.3f}]$ & {tv} & {fp(t['p'])} & {fp(q)} & {dz} \\\\"
        )
    return table(
        "Seed-paired contrasts (six seeds for every row). "
        "$\\Delta$: first condition minus second; CI: bootstrap 95\\% interval of the mean paired "
        "difference; $p_{\\mathrm{BH}}$: Benjamini--Hochberg adjustment within each family "
        "(ladder H1 to C; language-model H3, L1, L2; reasoning pilot R1, R2); E1$'$ is a "
        "mechanism check without adjustment; "
        "$d_z=\\bar{\\Delta}/s_{\\Delta}$.",
        "tab:stats-tests", "@{}llrrrrrr@{}",
        "\\textbf{ID} & \\textbf{Contrast} & $\\Delta$ & \\textbf{CI} & $t$ & $p$ & $p_{\\mathrm{BH}}$ & $d_z$",
        rows, env="table*", size="\\footnotesize",
    )


ABL_NAMES = {
    "baseline": "Baseline",
    "decay_off": "No suspicion decay",
    "no_decay_foothold": "No decay on compromised hosts",
    "lambda_low": "Detector rate $\\lambda=0.06$",
    "lambda_very_low": "Detector rate $\\lambda=0.03$",
    "lambda_high": "Detector rate $\\lambda=0.24$",
    "no_investigate": "No defender investigation",
    "no_decoy": "No decoys",
    "no_investigate_no_decoy": "No investigation, no decoys",
    "no_stealth_pacing": "No stealth pacing",
    "abandon_failed_hosts": "Abandon failed hosts",
}
ABL_CELLS = ["static_static", "adaptive_static", "persistent_static", "static_adaptive", "static_persistent"]


def ablation_block(res: Results, abl: dict) -> dict:
    out = {}
    for cell in ABL_CELLS:
        for cond, v in abl["cells"][cell].items():
            runs = v["runs"]
            d = {k: mean_ci([r.get(k) for r in runs]) for k in ("dr", "asr", "engagement", "dr_engaged")}
            out[(cell, cond)] = d
            for k, c in d.items():
                res.put(f"abl.{cond}.{short(cell)}.{k}", f3(c["mean"]))
    return out


def ablation_table(block: dict) -> str:
    rows = []
    for cond in ABL_NAMES:
        if ("static_static", cond) not in block:
            continue
        cells = []
        for cell in ABL_CELLS:
            c = block[(cell, cond)]
            cells.append(f"${c['dr']['mean']:.3f}$")
        eng = block[("adaptive_static", cond)]["engagement"]["mean"]
        dre = block[("adaptive_static", cond)]["dr_engaged"]["mean"]
        rows.append(f"{ABL_NAMES[cond]} & " + " & ".join(cells) + f" & ${eng:.3f}$ & ${dre:.3f}$ \\\\")
    return table(
        "Detector, pacing and engagement ablations (mean DR over six seeds, 40 episodes; "
        "attacker/defender levels). The last two columns decompose the L1/L0 cell: engagement "
        "rate and DR among engaged episodes.",
        "tab:ablation", "@{}lccccccc@{}",
        "\\textbf{Condition} & \\textbf{L0/L0} & \\textbf{L1/L0} & \\textbf{L2/L0} & \\textbf{L0/L1} & "
        "\\textbf{L0/L2} & \\textbf{Eng.\\ L1/L0} & \\textbf{DR$\\mid$Eng.\\ L1/L0}",
        rows, env="table*", size="\\footnotesize",
    )


def crossplay_table(res: Results, xp: dict) -> str:
    cells = xp["cells"]
    rows = []
    for a in cells:
        for d in cells:
            v = xp["results"][a][d]
            eng = v.get("engagement_by_seed")
            e = f"${sum(eng) / len(eng):.3f}$" if eng else "n/a"
            rows.append(f"{levels(a)} & {levels(d)} & ${v['dr_mean']:.3f}$ & ${v['asr_mean']:.3f}$ & {e} \\\\")
            res.put(f"xp.{short(a)}.{short(d)}.dr", f3(v["dr_mean"]))
            res.put(f"xp.{short(a)}.{short(d)}.asr", f3(v["asr_mean"]))
    drs = [xp["results"][a][d]["dr_mean"] for a in cells for d in cells]
    asrs = [xp["results"][a][d]["asr_mean"] for a in cells for d in cells]
    res.put("xp.min.dr", f3(min(drs)))
    res.put("xp.max.dr", f3(max(drs)))
    res.put("xp.min.asr", f3(min(asrs)))
    pp_def = [xp["results"][a]["persistent_persistent"]["asr_mean"] for a in cells]
    other_def = [xp["results"][a][d]["asr_mean"] for a in cells for d in cells if d != "persistent_persistent"]
    res.put("xp.ppdef.asr", f3(sum(pp_def) / len(pp_def)))
    res.put("xp.otherdef.asr", f3(sum(other_def) / len(other_def)))
    res.put("xp.max.asr", f3(max(asrs)))
    n, k = len(xp["seeds"]), xp["eval_episodes"]
    return table(
        f"Cross-play checkpoint swap after episode~{xp['checkpoint_episode']} ({n} seeds, {k} evaluation "
        "episodes per pairing). Each checkpoint is replayed at the adaptation level of its cell of origin; "
        "attacker checkpoint from the first cell, defender checkpoint from the second.",
        "tab:crossplay", "@{}llccc@{}",
        "\\textbf{Attacker from} & \\textbf{Defender from} & \\textbf{DR} & \\textbf{ASR} & \\textbf{Eng.}",
        rows,
    )


def dynamics_table(lad: dict) -> str:
    rows = []
    for cell in CELLS:
        cnt = defaultdict(int)
        for r in lad["cells"][cell]["runs"]:
            cnt[r.get("dynamics_class") or "unknown"] += 1
        rows.append(f"{levels(cell)} & {cnt['stable_equilibrium']} & {cnt['converging']} & "
                    f"{cnt['non_convergent']} \\\\")
    return table(
        "Inter-episode dynamics of the carried-over parameters (base stealth and vigilance): "
        "number of runs per class (six seeds).",
        "tab:dynamics", "@{}lccc@{}",
        "\\textbf{Cell} & \\textbf{Stable} & \\textbf{Converging} & \\textbf{Non-convergent}", rows,
    )


def cross_lagged_table(lad: dict) -> str:
    rows = []
    for cell in CELLS:
        runs = lad["cells"][cell]["runs"]
        parts = []
        for key in ("cross_lagged_def_to_att", "cross_lagged_att_to_def"):
            xs = [r[key] for r in runs if r.get(key) is not None and not math.isnan(r[key])]
            parts.append(f"${sum(xs) / len(xs):+.2f}$ [${min(xs):+.2f}$, ${max(xs):+.2f}$]" if xs else "n/a")
        rows.append(f"{levels(cell)} & {parts[0]} & {parts[1]} \\\\")
    return table(
        "Cross-lagged correlations of episode-to-episode signature change: seed mean [min, max].",
        "tab:cross-lagged", "@{}lcc@{}", "\\textbf{Cell} & $r_{D\\to A}$ & $r_{A\\to D}$", rows,
    )


def replication_table(t20: list[dict], p20: list[float], tnew: list[dict], pnew: list[float],
                      n_new: int) -> str:
    rows = []
    for t, q, u, qn in zip(t20, p20, tnew, pnew):
        fam = t["family"] == "mechanism"
        rows.append(f"{t['id']} & {t['label']} & ${t['diff']:+.3f}$ & {fp(t['p'])} & "
                    f"{'n/a' if fam else fp(q)} & ${u['diff']:+.3f}$ & {fp(u['p'])} \\\\")
    return table(
        f"Twenty-seed replication of the ladder contrasts: all twenty seeds (including the six "
        f"primary ones; BH within the ladder family) and the {n_new} new seeds only.",
        "tab:replication", "@{}llrrrrr@{}",
        "\\textbf{ID} & \\textbf{Contrast} & $\\Delta_{20}$ & $p_{20}$ & $p_{\\mathrm{BH},20}$ & "
        f"$\\Delta_{{{n_new}}}$ & $p_{{{n_new}}}$",
        rows, env="table*",
    )


# -------------------------------------------------------------------- figures
def ladder_figure(gen: Path, block: dict, n_seeds: int) -> None:
    grid = {"static": 1, "adaptive": 2, "persistent": 3}
    for di, d in enumerate(("static", "adaptive", "persistent")):
        lines = ["att dr dr_ci eng eng_ci asr asr_ci"]
        for a in ("static", "adaptive", "persistent"):
            c = block[f"{a}_{d}"]
            lines.append(f"{grid[a]} {c['dr']['mean']:.6f} {c['dr']['ci95']:.6f} "
                         f"{c['engagement']['mean']:.6f} {c['engagement']['ci95']:.6f} "
                         f"{c['asr']['mean']:.6f} {c['asr']['ci95']:.6f}")
        write(gen, f"ladder_def{di}.dat", "\n".join(lines) + "\n")


def abandonment_figure(gen: Path, lad: dict, lad_ab: dict) -> None:
    lines = ["idx dr eng dr_ab eng_ab"]
    for i, cell in enumerate(["static_static", "adaptive_static", "persistent_static", "persistent_persistent"]):
        c, cab = lad["cells"][cell], lad_ab["cells"][cell]
        m = lambda cc, k: mean_ci([r[k] for r in cc["runs"]])["mean"]  # noqa: E731
        lines.append(f"{i} {m(c, 'dr'):.6f} {m(c, 'engagement'):.6f} {m(cab, 'dr'):.6f} {m(cab, 'engagement'):.6f}")
    write(gen, "abandonment.dat", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/paper_runs")
    ap.add_argument("--paper", default="paper")
    args = ap.parse_args()
    data = ROOT / args.data
    gen = ROOT / args.paper / "figures" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    res = Results()

    # Heuristic ladder (primary).
    lad = load(data, "ladder_iso")
    n_seeds, episodes = len(lad["seeds"]), lad["episodes"]
    res.put("ladder.seeds", str(n_seeds))
    res.put("ladder.episodes", str(episodes))
    block = ladder_block(res, lad, "lad")
    tests = ladder_tests(lad)

    # Language-model campaign.
    runs = campaign_runs(data)
    llm_tests = []
    for hid, pa, pb, key, label in [
        ("H3", "p1_hybrid_b5_persistent", "p1_hybrid_b4_episodic", "asr", "B5 vs.\\ B4 (ASR)"),
        ("M1", "p1_hybrid_b4_episodic", "p1_hybrid_b3_static", "asr", "B4 vs.\\ B3 (ASR)"),
        ("M2", "p1_hybrid_b4_episodic", "p1_hybrid_b3_static", "dr", "B4 vs.\\ B3 (DR)"),
        ("R1", "p7_cot_on", "p7_cot_off", "gate_frac", "Reasoning on vs.\\ off (gate fraction)"),
        ("R2", "p7_cot_on", "p7_cot_off", "asr", "Reasoning on vs.\\ off (ASR)"),
    ]:
        r = llm_pair(runs, pa, pb, key)
        if r is not None:
            llm_tests.append({"id": hid, "label": label, "a": pa, "b": pb, "metric": key,
                              "family": "pilot" if hid.startswith("R") else "llm", **r})
    all_tests = tests + llm_tests
    p_adj = bh_by_family(all_tests)
    for t, q in zip(all_tests, p_adj):
        t["p_bh"] = q
        k = t["id"].replace("'", "p")
        res.put(f"test.{k}.diff", f"{t['diff']:+.3f}")
        res.put(f"test.{k}.absdiff", f"{abs(t['diff']):.3f}")
        res.put(f"test.{k}.p", "<0.001" if t["p"] < 0.001 else f"{t['p']:.3f}")
        res.put(f"test.{k}.peq", "<0.001" if t["p"] < 0.001 else f"={t['p']:.3f}")
        res.put(f"test.{k}.pbheq", "<0.001" if q < 0.001 else f"={q:.3f}")
        res.put(f"test.{k}.pbh", "<0.001" if q < 0.001 else f"{q:.3f}")
        res.put(f"test.{k}.t", "n/a" if math.isinf(t["t"]) else f"{t['t']:.2f}")
        res.put(f"test.{k}.dz", "n/a" if math.isinf(t["d_z"]) else f"{t['d_z']:.2f}")
        res.put(f"test.{k}.lo", f"{t['boot_lo']:+.3f}")
        res.put(f"test.{k}.hi", f"{t['boot_hi']:+.3f}")
    res.put("tests.n", str(len(all_tests)))
    write(gen, "ladder_table.tex", ladder_table(block, n_seeds, episodes))
    write(gen, "stats_tests.tex", stats_table(all_tests, p_adj, n_seeds))
    ladder_figure(gen, block, n_seeds)
    write(gen, "dynamics_table.tex", dynamics_table(lad))
    write(gen, "cross_lagged.tex", cross_lagged_table(lad))
    cep_rows = [f"{levels(c)} & {fci(block[c]['cep_adj'])} \\\\" for c in CELLS]
    write(gen, "cep_adj_table.tex", table(
        "Baseline-adjusted co-evolutionary pressure (same-seed L0/L0 subtracted; mean $\\pm$ 95\\% CI).",
        "tab:cep-adj", "@{}lc@{}", "\\textbf{Cell} & $\\mathrm{CEP}_{\\mathrm{adj}}$", cep_rows))
    res.stats["ladder"] = {"cells": {c: block[c] for c in CELLS}, "tests": tests}

    # Mechanism statistics from the exported data: first-detection source,
    # defender isolations, behavioural identity (action digests).
    lad_eps = load_episodes(data, "ladder_iso")
    manifest = json.loads((data / "manifest.json").read_text())
    digests = {name: m.get("actions_sha256") for name, m in manifest["runs"].get("ladder_iso", {}).items()}
    seed_of = {r["run_dir"].rsplit("/", 1)[-1]: (c, r["seed"]) for c in CELLS for r in lad["cells"][c]["runs"]}
    base_digest = {sd: digests.get(n) for n, (c, sd) in seed_of.items() if c == "static_static"}
    mech = {}
    for name, (cell, sd) in seed_of.items():
        m = mech.setdefault(cell, {"src": defaultdict(int), "ds_iso": 0, "gw_iso": 0, "n": 0, "ident": 0})
        for e in lad_eps.get(name, []):
            m["n"] += 1
            m["src"][e.get("first_detection_reason") or "undetected"] += 1
            iso = e.get("isolated_hosts") or []
            m["ds_iso"] += "datastore" in iso
            m["gw_iso"] += "gw" in iso
        m["ident"] += digests.get(name) is not None and digests.get(name) == base_digest.get(sd)
    for cell, m in mech.items():
        det = sum(v for k, v in m["src"].items() if k != "undetected")
        for k in ("investigation", "automatic", "decoy"):
            res.put(f"src.{short(cell)}.{k}", f"{100 * m['src'][k] / det:.0f}" if det else "n/a")
        res.put(f"ident.{short(cell)}", str(m["ident"]))
        res.put(f"iso.{short(cell)}.datastore", f"{100 * m['ds_iso'] / m['n']:.0f}")
        res.put(f"iso.{short(cell)}.gateway", f"{100 * m['gw_iso'] / m['n']:.0f}")
    res.stats["mechanism"] = {c: {**m, "src": dict(m["src"])} for c, m in mech.items()}
    # L2 carried-over parameters: first episode at their bound (stealth 1.0,
    # vigilance floor 0.1) and share of the following episodes spent there.
    first_hit: dict[str, list[int]] = defaultdict(list)
    at_bound: dict[str, list[float]] = defaultdict(list)
    for run, eps in lad_eps.items():
        eps = sorted(eps, key=lambda e: e["episode"])
        for role, key, bound, lvl in (("attacker", "base_stealth", 1.0, "_persistent_s"),
                                      ("defender", "base_vigilance", 0.1, "persistent_s")):
            is_l2 = run.split("_")[1 if role == "attacker" else 2] == "persistent"
            if not is_l2:
                continue
            vals = [(e.get(f"{role}_carried_params") or {}).get(key) for e in eps]
            hit = next((i for i, x in enumerate(vals) if x is not None and abs(x - bound) < 1e-9), None)
            if hit is None:
                continue
            first_hit[role].append(hit + 1)
            rest = vals[hit + 1:]
            if not rest:
                continue
            at_bound[role].append(sum(abs(x - bound) < 1e-9 for x in rest) / len(rest))
    # Representative L2/L2 run (lowest seed) for the carried-parameter figure.
    pp = sorted(r for r in lad_eps if r.startswith("ci_persistent_persistent_s"))
    if pp:
        eps = sorted(lad_eps[pp[0]], key=lambda e: e["episode"])
        lines = ["ep stealth vigilance detected engaged"]
        for e in eps:
            lines.append(f"{e['episode'] + 1} {e['attacker_carried_params']['base_stealth']:.3f} "
                         f"{e['defender_carried_params']['base_vigilance']:.3f} "
                         f"{int(bool(e['detected']))} {int(bool(e.get('foothold_gained')))}")
        write(gen, "carried_params.dat", "\n".join(lines) + "\n")
        res.put("fig.carried.seed", pp[0].split("_s")[-1].split("_")[0])
    for role in ("attacker", "defender"):
        xs = sorted(first_hit.get(role, []))
        res.put(f"sat.{role}.runs", str(len(xs)))
        res.put(f"sat.{role}.max", str(max(xs)) if xs else "n/a")
        res.put(f"sat.{role}.median", f"{(xs[(len(xs) - 1) // 2] + xs[len(xs) // 2]) / 2:g}" if xs else "n/a")
        fr = at_bound.get(role, [])
        res.put(f"sat.{role}.frac", f"{100 * sum(fr) / len(fr):.0f}" if fr else "n/a")
    res.stats["saturation"] = {"first_hit": dict(first_hit), "frac_at_bound": dict(at_bound)}

    # Abandonment variant.
    lad_ab = load(data, "ladder_iso_abandon")
    if lad_ab:
        block_ab = ladder_block(res, lad_ab, "ab")
        abandonment_figure(gen, lad, lad_ab)
        t_ab = paired(ladder_series(lad_ab, "adaptive_static", "dr"), ladder_series(lad_ab, "static_static", "dr"))
        res.put("ab.H1.diff", f"{t_ab['diff']:+.3f}")
        res.put("ab.H1.absdiff", f"{abs(t_ab['diff']):.3f}")
        res.stats["ladder_abandon"] = {"cells": {c: block_ab[c] for c in CELLS}, "H1": t_ab}

    # Twenty-seed replication.
    lad20 = load(data, "ladder_iso_20seeds")
    if lad20:
        t20 = ladder_tests(lad20)
        p20 = bh_by_family(t20)
        new_seeds = sorted(set(lad20["seeds"]) - set(lad["seeds"]))
        tnew = ladder_tests(lad20, new_seeds)
        pnew = bh_by_family(tnew)
        res.put("repnew.seeds", str(len(new_seeds)))
        for t, q in zip(tnew, pnew):
            k = t["id"].replace("'", "p")
            res.put(f"repnew.{k}.diff", f"{t['diff']:+.3f}")
            res.put(f"repnew.{k}.absdiff", f"{abs(t['diff']):.3f}")
            res.put(f"repnew.{k}.peq", "<0.001" if t["p"] < 0.001 else f"={t['p']:.3f}")
            res.put(f"repnew.{k}.pbheq", "<0.001" if q < 0.001 else f"={q:.3f}")
        ncls = defaultdict(int)
        for c in CELLS:
            for r in lad20["cells"][c]["runs"]:
                ncls[r.get("dynamics_class")] += 1
        res.put("rep.nonconv", str(ncls["non_convergent"]))
        res.put("rep.runs", str(sum(ncls.values())))
        write(gen, "replication.tex", replication_table(t20, p20, tnew, pnew, len(new_seeds)))
        res.put("rep.seeds", str(len(lad20["seeds"])))
        for t, q in zip(t20, p20):
            k = t["id"].replace("'", "p")
            res.put(f"rep.{k}.diff", f"{t['diff']:+.3f}")
            res.put(f"rep.{k}.absdiff", f"{abs(t['diff']):.3f}")
            res.put(f"rep.{k}.p", "<0.001" if t["p"] < 0.001 else f"{t['p']:.3f}")
            res.put(f"rep.{k}.peq", "<0.001" if t["p"] < 0.001 else f"={t['p']:.3f}")
            res.put(f"rep.{k}.pbh", "<0.001" if q < 0.001 else f"{q:.3f}")
            res.put(f"rep.{k}.pbheq", "<0.001" if q < 0.001 else f"={q:.3f}")
        b20 = ladder_block(res, lad20, "rep")
        res.stats["ladder_20seeds"] = {"cells": {c: b20[c] for c in CELLS}, "tests": t20, "p_bh": p20}

    # Ablations and cross-play.
    abl = load(data, "detector_ablation")
    if abl:
        ab_block = ablation_block(res, abl)
        write(gen, "ablation_table.tex", ablation_table(ab_block))
        res.stats["ablation"] = {f"{c}|{k}": v for (c, k), v in ab_block.items()}
    xp = load(data, "crossplay")
    if xp:
        write(gen, "crossplay.tex", crossplay_table(res, xp))
        res.stats["crossplay"] = xp["results"]

    # Kubernetes backend check.
    k8s = load(data, "k8s_validate")
    if k8s:
        rows, tot = [], defaultdict(int)
        for cell, v in k8s["cells"].items():
            iso = v["isolations"]
            for k, n in iso.items():
                tot[k] += n
            rows.append(f"{levels(cell)} & ${v['asr_sim']:.3f}$ & ${v['asr_k8s']:.3f}$ & "
                        f"${v['dr_sim']:.3f}$ & ${v['dr_k8s']:.3f}$ & "
                        f"{sum(r['episodes_identical'] for r in v['runs'])}/{len(v['runs'])} & "
                        f"{iso['hosts_applied']}/{iso['hosts']} \\\\")
        write(gen, "k8s_table.tex", table(
            f"Simulator vs.\\ Kubernetes backend ({len(k8s['seeds'])} seeds, {k8s['episodes']} episodes): "
            "mean ASR and DR, runs whose episode records are identical, and isolations of real hosts "
            "applied on the cluster.",
            "tab:k8s", "@{}lcccccc@{}",
            "\\textbf{Cell} & \\textbf{ASR sim} & \\textbf{ASR k8s} & \\textbf{DR sim} & "
            "\\textbf{DR k8s} & \\textbf{Identical} & \\textbf{Isolations}", rows))
        res.put("k8s.enforced", "yes" if k8s["enforcement"]["enforced"] else "no")
        res.put("k8s.hosts", str(tot["hosts"]))
        res.put("k8s.hosts_applied", str(tot["hosts_applied"]))
        res.put("k8s.decoys", str(tot["decoys"]))
        res.put("k8s.runs", str(sum(len(v["runs"]) for v in k8s["cells"].values())))
        res.put("k8s.identical", str(sum(r["episodes_identical"] for v in k8s["cells"].values() for r in v["runs"])))
        res.stats["k8s"] = {"enforcement": k8s["enforcement"], "isolations": dict(tot)}

    # Language-model tables.
    if runs:
        llm = {}
        rows = []
        for label, name, lvl in LLM_CELLS:
            ms = list(by_seed(runs, f"p1_hybrid_{label}").values())
            if not ms:
                continue
            c = {k: mean_ci([m[k] for m in ms]) for k in
                 ("asr", "dr", "engagement", "dr_engaged", "gate_frac", "src_auto", "succ_detected")}
            llm[name] = c
            rows.append(f"{name} ({lvl}/{lvl}) & {fci(c['asr'])} & {fci(c['dr'])} & {fci(c['engagement'])} & "
                        f"${c['dr_engaged']['mean']:.3f}$ & {fci(c['gate_frac'])} \\\\")
            for k, v in c.items():
                res.put(f"llm.{name}.{k}", f3(v["mean"]))
                res.put(f"llm.{name}.{k}.ci", f3(v["ci95"]))
            res.put(f"llm.{name}.n", str(c["asr"]["n"]))
        # Action mix from the manifest's per-run action counts.
        man_runs = json.loads((data / "manifest.json").read_text())["runs"].get("campaign", {})
        for label, name, _ in LLM_CELLS:
            att, dfn = defaultdict(int), defaultdict(int)
            for rn, meta in man_runs.items():
                if not rn.startswith(f"p1_hybrid_{label}_s"):
                    continue
                for origin, tools in ((meta.get("action_counts") or {}).get("attacker") or {}).items():
                    for tool, n in tools.items():
                        att[(origin, tool)] += n
                for origin, tools in ((meta.get("action_counts") or {}).get("defender") or {}).items():
                    for tool, n in tools.items():
                        dfn[tool] += n
            ta, td = sum(att.values()), sum(dfn.values())
            if ta:
                res.put(f"mix.{name}.fallback_wait", f"{100 * att[('fallback', 'wait')] / ta:.0f}")
                res.put(f"mix.{name}.model_wait", f"{100 * att[('policy', 'wait')] / ta:.0f}")
            if td:
                res.put(f"mix.{name}.def_monitor", f"{100 * dfn['monitor'] / td:.0f}")
                res.put(f"mix.{name}.def_isolate", f"{100 * dfn['isolate'] / td:.1f}")
                res.put(f"mix.{name}.def_investigate", f"{100 * dfn['investigate'] / td:.0f}")
        write(gen, "llm_table.tex", table(
            "Hybrid language-model cells, both roles hybrid at the same level (mean $\\pm$ 95\\% CI over "
            "six seeds, 30 episodes). Gate: fraction of attacker steps executed by the heuristic fallback.",
            "tab:llm", "@{}lccccc@{}",
            "\\textbf{Cell} & \\textbf{ASR} & \\textbf{DR} & \\textbf{Eng.} & \\textbf{DR$\\mid$Eng.} & \\textbf{Gate}",
            rows, env="table*"))
        # Asymmetric cells and ablations.
        rows = []
        for prefix, desc in [("p1b_hybatt_heurdef", "Hybrid attacker at {lvl}, heuristic L0 defender"),
                             ("p1b_heuratt_hybdef", "Heuristic L0 attacker, hybrid defender at {lvl}")]:
            for label, name, lvl in LLM_CELLS:
                ms = list(by_seed(runs, f"{prefix}_{label}").values())
                if not ms:
                    continue
                c = {k: mean_ci([m[k] for m in ms]) for k in ("asr", "dr", "engagement", "gate_frac")}
                tag = "ha" if "hybatt" in prefix else "hd"
                for k, v in c.items():
                    res.put(f"asym.{tag}.{name}.{k}", f3(v["mean"]))
                rows.append(f"{desc.format(lvl=lvl)} & {c['asr']['n']} & {fci(c['asr'])} & {fci(c['dr'])} & "
                            f"{fci(c['engagement'])} \\\\")
        for prefix, desc in [("p2b_llm_only_b4", "Language model only, no gate (both L1)"),
                             ("p2b_gate_only_b4", "Heuristic only (both L1)")]:
            ms = list(by_seed(runs, prefix).values())
            if ms:
                c = {k: mean_ci([m[k] for m in ms]) for k in ("asr", "dr", "engagement")}
                tag = "llmonly" if "llm_only" in prefix else "gateonly"
                for k, v in c.items():
                    res.put(f"abl2.{tag}.{k}", f3(v["mean"]))
                rows.append(f"{desc} & {c['asr']['n']} & {fci(c['asr'])} & {fci(c['dr'])} & {fci(c['engagement'])} \\\\")
        if rows:
            write(gen, "llm_asym_table.tex", table(
                "Asymmetric and attribution cells of the language-model campaign (mean $\\pm$ 95\\% CI; "
                "$n$ seeds; 30 episodes for asymmetric cells, 20 for the attribution ablation).",
                "tab:llm-asym", "@{}Xcccc@{}",
                "\\textbf{Configuration} & $n$ & \\textbf{ASR} & \\textbf{DR} & \\textbf{Eng.}",
                rows, env="table*", tabx=False).replace("@{}Xcccc@{}", "@{}lcccc@{}"))
        # Hybrid B4 restricted to the model-only arm's seeds (paired comparison).
        lo, hy = by_seed(runs, "p2b_llm_only_b4"), by_seed(runs, "p1_hybrid_b4_episodic")
        common = sorted(set(lo) & set(hy))
        if common:
            # Same seeds and, since the model-only arm has 20 episodes, the
            # same first 20 episode environments of each hybrid run.
            n_lo = min(lo[s_]["episodes"] for s_ in common)
            eps_all = load_episodes(data, "campaign")
            camp_raw = (load(data, "campaign").get("results") or {})
            sub_metrics = []
            for s_ in common:
                job = next(j for j, m in runs.items() if m is hy[s_])
                name = Path(camp_raw[job]["run_dir"]).name
                eps = sorted(eps_all[name], key=lambda e: e["episode"])[:n_lo]
                sub_metrics.append(run_metrics(eps))
            for k in ("asr", "dr"):
                res.put(f"abl2.hybrid_same_seeds.{k}", f3(sum(m[k] for m in sub_metrics) / len(sub_metrics)))
            res.put("abl2.n", str(len(common)))
            res.put("abl2.episodes", str(n_lo))
        cot = {}
        for tag in ("on", "off"):
            ms = list(by_seed(runs, f"p7_cot_{tag}").values())
            if ms:
                cot[tag] = {k: mean_ci([m[k] for m in ms]) for k in ("asr", "dr", "engagement", "gate_frac")}
                for k, v in cot[tag].items():
                    res.put(f"cot.{tag}.{k}", f3(v["mean"]))
        res.stats["llm"] = {"phase1": llm, "cot": cot, "runs": runs}
    res.stats["tests_all"] = all_tests

    # Heuristic adaptation x environment-regime x arena study.
    reg_path = data / "regime_stats.json"
    if reg_path.exists():
        reg = json.loads(reg_path.read_text())
        arenas = list(reg["results"])
        regimes = list(reg["results"][arenas[0]])
        cid_label = {"att_mem": "Attacker L2 vs L0", "def_mem": "Defender L2 vs L0",
                     "both_mem": "L2/L2 vs L0/L0"}
        rows = []
        for ar in arenas:
            for rg in regimes:
                cell = reg["results"][ar][rg]
                res.put(f"reg.{ar}.{rg}.dr00", f3(cell["L0_L0_dr"]))
                first = True
                for tt in cell["contrasts"]:
                    cid = tt["id"]
                    for m in ("dr", "asr", "er"):
                        res.put(f"reg.{ar}.{rg}.{cid}.{m}", f"{tt[m]['diff']:+.3f}")
                        res.put(f"reg.{ar}.{rg}.{cid}.{m}.abs", f"{abs(tt[m]['diff']):.3f}")
                    res.put(f"reg.{ar}.{rg}.{cid}.dr.p", fp(tt["dr"]["p"]).strip("$"))
                    frac = tt["dr_from_engagement"]
                    res.put(f"reg.{ar}.{rg}.{cid}.engfrac",
                            "n/a" if frac is None else f"{frac:.2f}")
                    rows.append(
                        f"{ar if first else ''} & {rg if first else ''} & {cid_label[cid]} & "
                        f"${tt['dr']['diff']:+.3f}$ & {fp(tt['dr']['p'])} & "
                        f"${tt['asr']['diff']:+.3f}$ & ${tt['er']['diff']:+.3f}$ \\\\")
                    first = False
                rows.append("\\midrule")
        if rows and rows[-1] == "\\midrule":
            rows.pop()
        write(gen, "regime_table.tex", table(
            "Heuristic memory contrasts across environment regime and detector arena "
            f"({len(reg['seeds'])} seeds, {reg['episodes']} episodes). $\\Delta$ is the "
            "seed-paired mean difference from the L0/L0 baseline of the same arena and "
            "regime; $p$ is exact sign-flip for constant differences, else paired $t$.",
            "tab:regime", "@{}lllrrrr@{}",
            "\\textbf{Arena} & \\textbf{Regime} & \\textbf{Contrast} & $\\Delta$DR & $p$ & "
            "$\\Delta$ASR & $\\Delta$ER", rows, env="table*"))
        res.stats["regime"] = reg["results"]

    # 2x2 language-model campaign (attacker memory x defender memory).
    l2_path = data / "llm2x2_stats.json"
    if l2_path.exists():
        l2 = json.loads(l2_path.read_text())
        rg = "recurring"
        if rg in l2["cells"]:
            cells = l2["cells"][rg]
            names = {"A0D0": "L0/L0", "A0D1": "L0/L2", "A1D0": "L2/L0", "A1D1": "L2/L2"}
            rows = []
            for c in ("A0D0", "A0D1", "A1D0", "A1D1"):
                cc = cells[c]
                for k in ("asr", "dr", "er", "dr_engaged", "gate_frac", "gate_frac_def"):
                    if k in cc:
                        res.put(f"llm2.{c}.{k}", f3(cc[k]["mean"]))
                        res.put(f"llm2.{c}.{k}.ci", f3(cc[k]["ci95"]))
                gd = fci(cc["gate_frac_def"]) if "gate_frac_def" in cc else "--"
                rows.append(f"{names[c]} & {fci(cc['asr'])} & {fci(cc['dr'])} & {fci(cc['er'])} & "
                            f"${cc['dr_engaged']['mean']:.3f}$ & {fci(cc['gate_frac'])} & {gd} \\\\")
            write(gen, "llm2x2_table.tex", table(
                "Hybrid language-model 2$\\times$2 on the headroom arena, recurring regime "
                "(attacker/defender memory level; mean $\\pm$ 95\\% CI over 20 seeds, 30 "
                "episodes). Gate(A) and Gate(D): fraction of the attacker's and defender's "
                "steps taken by the fallback, excluding the forced initial reconnaissance.",
                "tab:llm2x2", "@{}lcccccc@{}",
                "\\textbf{Cell} & \\textbf{ASR} & \\textbf{DR} & \\textbf{ER} & "
                "\\textbf{DR$\\mid$E} & \\textbf{Gate(A)} & \\textbf{Gate(D)}", rows))
            for metric in ("asr", "dr"):
                for eff in ("attacker_memory", "defender_memory", "interaction"):
                    e = l2["effects"][rg][metric][eff]
                    tag = {"attacker_memory": "attmem", "defender_memory": "defmem",
                           "interaction": "inter"}[eff]
                    res.put(f"llm2.{tag}.{metric}.diff", f"{e['diff']:+.3f}")
                    res.put(f"llm2.{tag}.{metric}.absdiff", f"{abs(e['diff']):.3f}")
                    res.put(f"llm2.{tag}.{metric}.p", fp(e["p"]).strip("$"))
                    res.put(f"llm2.{tag}.{metric}.peq",
                            "<0.001" if e["p"] < 0.001 else f"={e['p']:.3f}")
                    if e.get("p_bh") is not None:
                        res.put(f"llm2.{tag}.{metric}.pbh",
                                "<0.001" if e["p_bh"] < 0.001 else f"{e['p_bh']:.3f}")
                        res.put(f"llm2.{tag}.{metric}.pbheq",
                                "<0.001" if e["p_bh"] < 0.001 else f"={e['p_bh']:.3f}")
                    if "boot_lo" in e:
                        res.put(f"llm2.{tag}.{metric}.ci",
                                f"[{e['boot_lo']:+.3f}, {e['boot_hi']:+.3f}]")
            res.stats["llm2x2"] = l2
        if "iid" in l2["cells"]:
            ic = l2["cells"]["iid"]
            names = {"A0D0": "L0/L0", "A0D1": "L0/L2", "A1D0": "L2/L0", "A1D1": "L2/L2"}
            irows = []
            for c in ("A0D0", "A0D1", "A1D0", "A1D1"):
                for k in ("asr", "dr"):
                    res.put(f"llm2i.{c}.{k}", f3(ic[c][k]["mean"]))
                irows.append(f"{names[c]} & {fci(ic[c]['asr'])} & {fci(ic[c]['dr'])} & "
                             f"{fci(ic[c]['er'])} & ${ic[c]['dr_engaged']['mean']:.3f}$ \\\\")
            write(gen, "llm2x2_iid_table.tex", table(
                "The same $2{\\times}2$ under independent episodes (iid regime; mean "
                "$\\pm$ 95\\% CI over 20 seeds, 30 episodes).",
                "tab:llm2x2iid", "@{}lcccc@{}",
                "\\textbf{Cell} & \\textbf{ASR} & \\textbf{DR} & \\textbf{ER} & "
                "\\textbf{DR$\\mid$E}", irows))
            for metric in ("asr", "dr"):
                for eff, tag in (("attacker_memory", "attmem"), ("defender_memory", "defmem"),
                                 ("interaction", "inter")):
                    e = l2["effects"]["iid"][metric][eff]
                    res.put(f"llm2i.{tag}.{metric}.diff", f"{e['diff']:+.3f}")
                    res.put(f"llm2i.{tag}.{metric}.absdiff", f"{abs(e['diff']):.3f}")
                    if e.get("p_bh") is not None:
                        res.put(f"llm2i.{tag}.{metric}.pbheq",
                                "<0.001" if e["p_bh"] < 0.001 else f"={e['p_bh']:.3f}")
                    if "boot_lo" in e:
                        res.put(f"llm2i.{tag}.{metric}.ci",
                                f"[{e['boot_lo']:+.3f}, {e['boot_hi']:+.3f}]")

    # Behavioural mechanisms behind the 2x2 effects (measured from trajectories).
    mech_path = data / "llm_mechanisms.json"
    if mech_path.exists():
        mech = json.loads(mech_path.read_text())
        rg = "recurring"
        if rg in mech.get("cells", {}):
            mc = mech["cells"][rg]
            names = {"A0D0": "L0/L0", "A0D1": "L0/L2", "A1D0": "L2/L0", "A1D1": "L2/L2"}
            mrows = []
            for c in ("A0D0", "A0D1", "A1D0", "A1D1"):
                cc = mc[c]
                for k in ("def_watch_turn", "def_watch_frac", "att_wait_frac", "att_deceptive"):
                    res.put(f"mech.{c}.{k}", f3(cc[k]) if not math.isnan(cc[k]) else "n/a")
                wt = f"{cc['def_watch_turn']:.2f}" if not math.isnan(cc["def_watch_turn"]) else "--"
                mrows.append(f"{names[c]} & {wt} & {cc['def_watch_frac']:.2f} & "
                             f"{cc['att_wait_frac']:.3f} & {cc['att_deceptive']:.2f} \\\\")
            write(gen, "mechanisms_table.tex", table(
                "Behavioural mechanisms in the 2$\\times$2 (headroom, recurring; means over "
                "20 seeds). Watch turn and watch rate: when and how often the defender first "
                "monitors or investigates the recurring service host (``--'' when it never "
                "does). Wait fraction and deceptive actions describe the attacker's pacing.",
                "tab:mechanisms", "@{}lcccc@{}",
                "\\textbf{Cell} & \\textbf{Watch turn} & \\textbf{Watch rate} & "
                "\\textbf{Wait frac.} & \\textbf{Deceptive}", mrows, env="table"))
            for k in ("def_watch_turn", "def_watch_frac", "att_wait_frac", "att_deceptive"):
                e = mech["effects"][rg][k]
                if e.get("diff") is not None and not (isinstance(e["diff"], float) and math.isnan(e["diff"])):
                    res.put(f"mech.{k}.diff", f"{e['diff']:+.3f}")
                    res.put(f"mech.{k}.absdiff", f"{abs(e['diff']):.3f}")
                    res.put(f"mech.{k}.peq", "<0.001" if e["p"] < 0.001 else f"={e['p']:.3f}")
                    if e.get("p_bh") is not None:
                        res.put(f"mech.{k}.pbheq",
                                "<0.001" if e["p_bh"] < 0.001 else f"={e['p_bh']:.3f}")
                    if "boot_lo" in e:
                        res.put(f"mech.{k}.ci", f"[{e['boot_lo']:+.3f}, {e['boot_hi']:+.3f}]")
        res.stats["llm_mechanisms"] = mech

    # Result macros and stats dump.
    macro = ["% Generated by scripts/make_paper_artifacts.py; do not edit.",
             "\\makeatletter",
             "\\newcommand{\\res}[1]{\\ifcsname res@#1\\endcsname\\csname res@#1\\endcsname"
             "\\else\\textbf{??#1??}\\fi}"]
    for k, v in sorted(res.values.items()):
        macro.append(f"\\@namedef{{res@{k}}}{{{v}}}")
    macro.append("\\makeatother")
    write(gen, "results_macros.tex", "\n".join(macro) + "\n")
    (data / "paper_stats.json").write_text(json.dumps(res.stats, indent=2, default=str))
    print(f"wrote {len(res.values)} macros and tables to {gen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
