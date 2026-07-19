#!/usr/bin/env python3
"""Emit paper/macros.tex and paper/tables.tex from the scored data so every
number in the paper is regenerated from the analysis pipeline."""
import json
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "paper")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(7)
NBOOT = 10000
ROUNDS = [1, 2, 3, 4, 5]
ARMS = ["A", "B", "C"]

df = pd.read_csv(os.path.join(BASE, "results.csv"))
calls = pd.read_csv(os.path.join(BASE, "calls.csv"))
main = df[~df.chain_id.str.startswith(("pilot", "repA0"))]
summary = json.load(open(os.path.join(BASE, "analysis", "summary.json")))


def chain_curves(sub):
    piv = sub.pivot_table(index="chain_id", columns="round",
                          values="survived", aggfunc="mean")
    return piv[ROUNDS].values


def boot_ci(mat, stat):
    k = mat.shape[0]
    vals = [stat(mat[RNG.integers(0, k, k)]) for _ in range(NBOOT)]
    return np.percentile(vals, [2.5, 97.5])


# LaTeX control words cannot contain digits, so spell rounds out.
NUM = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}

macros = []
ARMNAME = {"A": "ArmA", "B": "ArmB", "C": "ArmC"}
mats = {}
for arm in ARMS:
    sub = main[main.arm == arm]
    mat = chain_curves(sub)
    mats[arm] = mat
    mean = mat.mean(axis=0)
    st = summary[arm]
    for i, r in enumerate(ROUNDS):
        macros.append(f"\\newcommand{{\\{ARMNAME[arm]}Sr{NUM[r]}}}"
                      f"{{{mean[i]:.3f}}}")
        macros.append(f"\\newcommand{{\\{ARMNAME[arm]}Cnt{NUM[r]}}}"
                      f"{{{int(sub[sub['round'] == r].survived.sum())}}}")
    macros.append(f"\\newcommand{{\\{ARMNAME[arm]}s}}{{{st['s']:.3f}}}")
    macros.append(f"\\newcommand{{\\{ARMNAME[arm]}sLo}}{{{st['s_ci'][0]:.3f}}}")
    macros.append(f"\\newcommand{{\\{ARMNAME[arm]}sHi}}{{{st['s_ci'][1]:.3f}}}")
    hl = st["half_life"]
    hltxt = f"{hl:.2f}" if np.isfinite(hl) else "$\\infty$"
    macros.append(f"\\newcommand{{\\{ARMNAME[arm]}HalfLife}}{{{hltxt}}}")

mc = calls[~calls.chain_id.str.startswith(("pilot", "repA0"))]
macros.append(f"\\newcommand{{\\TotalCost}}{{{calls.total_cost_usd.sum():.2f}}}")
macros.append(f"\\newcommand{{\\MeanLatency}}"
              f"{{{calls.duration_api_ms.mean()/1000:.1f}}}")
macros.append(f"\\newcommand{{\\NCalls}}{{{len(calls)}}}")
mut = main[main.mutated == 1]
macros.append(f"\\newcommand{{\\NMutated}}{{{len(mut)}}}")

# budget compliance
for arm in ARMS:
    s = mc[mc.arm == arm]
    macros.append(f"\\newcommand{{\\{ARMNAME[arm]}MeanWords}}"
                  f"{{{s.summary_words.mean():.0f}}}")
    macros.append(f"\\newcommand{{\\{ARMNAME[arm]}OverBudget}}"
                  f"{{{int((s.summary_words > s.budget_words).sum())}}}")

# pilot calibration gate
pilot = df[df.chain_id.str.startswith("pilot")]
pr1 = pilot[pilot["round"] == 1].survived.mean()
macros.append(f"\\newcommand{{\\PilotRoneRetention}}{{{pr1:.3f}}}")

# probes
probe_path = os.path.join(BASE, "probes", "probe_results.json")
if os.path.exists(probe_path):
    pr = pd.DataFrame(json.load(open(probe_path)))
    ret = pr[pr.kind == "retained"]
    lost = pr[pr.kind.isin(["lost", "mutated"])]
    macros.append(f"\\newcommand{{\\NProbed}}{{{len(pr)}}}")
    macros.append(f"\\newcommand{{\\NProbeRetained}}{{{len(ret)}}}")
    macros.append(f"\\newcommand{{\\NProbeRetainedExact}}"
                  f"{{{int(ret.exact_match.sum())}}}")
    macros.append(f"\\newcommand{{\\NProbeLost}}{{{len(lost)}}}")
    macros.append(f"\\newcommand{{\\NProbeAbstained}}"
                  f"{{{int(lost.abstained.sum())}}}")
    macros.append(f"\\newcommand{{\\NProbeConfab}}"
                  f"{{{int(lost.codelike_wrong.sum())}}}")

# cost including probes
import glob
probe_cost = sum(json.load(open(p)).get("total_cost_usd", 0)
                 for p in glob.glob(os.path.join(BASE, "probes",
                                                 "probe-A-*.json")))
macros.append(f"\\newcommand{{\\ProbeCost}}{{{probe_cost:.2f}}}")
macros.append(f"\\newcommand{{\\GrandCost}}"
              f"{{{calls.total_cost_usd.sum() + probe_cost:.2f}}}")

# pre-registered arm contrasts (same rule as analyze.py)
r5stat = lambda m: m.mean(axis=0)[4]


def fit_s(curve):
    S = np.clip(curve, 1e-3, None)
    r = np.array(ROUNDS, dtype=float)
    return float(np.exp((r * np.log(S)).sum() / (r * r).sum()))


sfit = lambda m: fit_s(m.mean(axis=0))
CONTRASTS = [("HTwoRfive", "B", "A", r5stat), ("HTwoS", "B", "A", sfit),
             ("HThreeRfive", "C", "A", r5stat), ("HThreeS", "C", "A", sfit),
             ("CBRfive", "C", "B", r5stat)]
contrast_vals = {}
for name, a1, a2, statf in CONTRASTS:
    m1, m2 = mats[a1], mats[a2]
    d = statf(m1) - statf(m2)
    vals = []
    for _ in range(NBOOT):
        i1 = RNG.integers(0, m1.shape[0], m1.shape[0])
        i2 = RNG.integers(0, m2.shape[0], m2.shape[0])
        vals.append(statf(m1[i1]) - statf(m2[i2]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    contrast_vals[name] = (d, lo, hi)
    macros.append(f"\\newcommand{{\\{name}Diff}}{{{d:+.3f}}}")
    macros.append(f"\\newcommand{{\\{name}Lo}}{{{lo:+.3f}}}")
    macros.append(f"\\newcommand{{\\{name}Hi}}{{{hi:+.3f}}}")

with open(os.path.join(OUT, "macros.tex"), "w") as f:
    f.write("\n".join(macros) + "\n")

# survival table with CIs
rows = []
for arm in ARMS:
    sub = main[main.arm == arm]
    mat = chain_curves(sub)
    mean = mat.mean(axis=0)
    cells = []
    for i in range(5):
        lo, hi = boot_ci(mat, lambda m, i=i: m.mean(axis=0)[i])
        cells.append(f"{mean[i]:.2f} [{lo:.2f},{hi:.2f}]")
    st = summary[arm]
    label = {"A": "A: 150w", "B": "B: 400w", "C": "C: 150w+verbatim"}[arm]
    hl = st["half_life"]
    hltxt = f"{hl:.2f}" if np.isfinite(hl) and hl < 100 else "$>$100"
    rows.append(f"{label} & " + " & ".join(cells) +
                f" & {st['s']:.3f} & {hltxt} \\\\")

tab = ("\\begin{table*}[t]\\centering\\footnotesize\n"
       "\\setlength{\\tabcolsep}{2pt}\n"
       "\\caption{Fact survival $S(r)$ by arm (mean over 6 chains, 95\\% "
       "bootstrap CI over chains, 10{,}000 reps), fitted per-round survival "
       "constant $s$, and compaction half-life in rounds.}\n"
       "\\label{tab:survival}\n"
       "\\begin{tabular}{lccccccc}\n\\toprule\n"
       "Arm & $S(1)$ & $S(2)$ & $S(3)$ & $S(4)$ & $S(5)$ & $s$ & "
       "half-life \\\\\n\\midrule\n" + "\n".join(rows) +
       "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# pre-registered contrast table
CLABEL = {"HTwoRfive": ("H2: B $-$ A", "round-5 survival"),
          "HTwoS": ("H2: B $-$ A", "fitted $s$"),
          "HThreeRfive": ("H3: C $-$ A", "round-5 survival"),
          "HThreeS": ("H3: C $-$ A", "fitted $s$"),
          "CBRfive": ("C $-$ B", "round-5 survival")}
crows = []
for name, (lbl, metric) in CLABEL.items():
    d, lo, hi = contrast_vals[name]
    sig = "yes" if (lo > 0 or hi < 0) else "no"
    crows.append(f"{lbl} & {metric} & ${d:+.3f}$ & $[{lo:+.3f},{hi:+.3f}]$ "
                 f"& {sig} \\\\")
ctab = ("\\begin{table}[t]\\centering\\small\n"
        "\\caption{Pre-registered arm contrasts. Difference and 95\\% "
        "bootstrap CI (chains resampled independently per arm, 10{,}000 "
        "reps); a difference is claimed only if the CI excludes 0.}\n"
        "\\label{tab:contrasts}\n"
        "\\begin{tabular}{llccc}\n\\toprule\n"
        "Contrast & Metric & Diff & 95\\% CI & Claimed \\\\\n\\midrule\n" +
        "\n".join(crows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# spot-repeat table
rep_ids = ["A-0", "repA0-rep1", "repA0-rep2"]
srows = []
for cid in rep_ids:
    sub = df[df.chain_id == cid]
    cnt = [int(sub[sub["round"] == r].survived.sum()) for r in ROUNDS]
    srows.append(f"{cid} & " + " & ".join(map(str, cnt)) + " \\\\")
stab = ("\\begin{table}[t]\\centering\\small\n"
        "\\caption{Run-to-run stability: the identical chain spec (A-0) run "
        "three times with fresh model calls. Surviving codes out of 16 by "
        "round.}\n\\label{tab:spotrepeat}\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "Run & $r{=}1$ & $r{=}2$ & $r{=}3$ & $r{=}4$ & $r{=}5$ \\\\\n"
        "\\midrule\n" + "\n".join(srows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

with open(os.path.join(OUT, "tables.tex"), "w") as f:
    f.write(tab + "\n" + ctab + "\n" + stab)

# copy plots next to the paper so the LaTeX build stays self-contained
import shutil
FIGS = ["retention_by_round.png", "survival_fit.png",
        "budget_compliance.png", "position_survival.png"]
os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
for fig in FIGS:
    src = os.path.join(BASE, "analysis", fig)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, "figs", fig))
print("paper assets written")
