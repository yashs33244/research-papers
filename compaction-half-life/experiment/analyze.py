#!/usr/bin/env python3
"""Full analysis: every number in analysis.md is computed here from
results.csv / calls.csv / probe_results.json / raw JSONs. No hand-entered
figures."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(7)
NBOOT = 10000
ROUNDS = [1, 2, 3, 4, 5]
ARMS = ["A", "B", "C"]
ARM_LABEL = {"A": "A (150w)", "B": "B (400w)", "C": "C (150w+verbatim)"}

df = pd.read_csv(os.path.join(BASE, "results.csv"))
calls = pd.read_csv(os.path.join(BASE, "calls.csv"))

main = df[~df.chain_id.str.startswith(("pilot", "repA0"))]
pilot = df[df.chain_id.str.startswith("pilot")]
reps = df[df.chain_id.str.startswith("repA0") | (df.chain_id == "A-0")]


def chain_curves(sub):
    """rows: chains, cols: rounds -> survival fraction."""
    piv = sub.pivot_table(index="chain_id", columns="round",
                          values="survived", aggfunc="mean")
    return piv[ROUNDS].values


def boot_ci(mat, stat, n=NBOOT):
    """Bootstrap over chains (rows). stat: matrix -> scalar or vector."""
    k = mat.shape[0]
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, k, k)
        vals.append(stat(mat[idx]))
    vals = np.array(vals)
    return np.percentile(vals, 2.5, axis=0), np.percentile(vals, 97.5, axis=0)


def fit_s(curve):
    """LS fit of log S(r) = r log s through origin; clamp zeros."""
    S = np.clip(curve, 1e-3, None)
    r = np.array(ROUNDS, dtype=float)
    logs = np.log(S)
    return float(np.exp((r * logs).sum() / (r * r).sum()))


def cond_delta(mat):
    m = mat.mean(axis=0)
    S = np.concatenate([[1.0], m])
    c = S[1:] / np.clip(S[:-1], 1e-9, None)
    return c[0] - c[1:].mean()


report = []
report.append("# Analysis: The Compaction Half-Life\n")
report.append("All numbers below are computed by analyze.py from results.csv, "
              "calls.csv, probes/probe_results.json and the raw per-call JSON "
              "files in results/. Nothing is hand-entered.\n")

# ---- headline retention table ----
arm_stats = {}
report.append("## 1. Survival curves S(r) by arm\n")
report.append("Retention = fraction of the 96 planted codes (16 facts x 6 "
              "chains) present verbatim (case-insensitive substring) in the "
              "round-r summary. CI = 95 percent bootstrap over chains "
              "(10,000 reps).\n")
report.append("| Arm | r1 | r2 | r3 | r4 | r5 | fitted s | half-life (rounds) |")
report.append("|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    sub = main[main.arm == arm]
    mat = chain_curves(sub)
    mean = mat.mean(axis=0)
    lo, hi = boot_ci(mat, lambda m: m.mean(axis=0))
    s = fit_s(mean)
    slo, shi = boot_ci(mat, lambda m: fit_s(m.mean(axis=0)))
    hl = np.log(2) / -np.log(s) if s < 1 else float("inf")
    arm_stats[arm] = {"mat": mat, "mean": mean, "lo": lo, "hi": hi,
                      "s": s, "slo": float(slo), "shi": float(shi), "hl": hl}
    cells = " | ".join(f"{mean[i]:.3f} [{lo[i]:.3f},{hi[i]:.3f}]"
                       for i in range(5))
    report.append(f"| {ARM_LABEL[arm]} | {cells} | {s:.3f} "
                  f"[{float(slo):.3f},{float(shi):.3f}] | {hl:.2f} |")
report.append("")

# raw counts
report.append("Raw surviving-code counts out of 96 per cell:\n")
report.append("| Arm | r1 | r2 | r3 | r4 | r5 |")
report.append("|---|---|---|---|---|---|")
for arm in ARMS:
    sub = main[main.arm == arm]
    cnt = [int(sub[sub['round'] == r].survived.sum()) for r in ROUNDS]
    report.append(f"| {ARM_LABEL[arm]} | " + " | ".join(map(str, cnt)) + " |")
report.append("")

# ---- H1 shape test ----
report.append("## 2. H1: geometric vs front-loaded decay "
              "(pre-registered rule)\n")
report.append("delta = c_1 - mean(c_2..c_5) where c_r = S(r)/S(r-1) on chain "
              "means; FRONT-LOADED iff 95 percent bootstrap CI of delta "
              "excludes 0 and delta < 0.\n")
report.append("| Arm | c1 | c2 | c3 | c4 | c5 | delta | 95% CI | verdict |")
report.append("|---|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    st = arm_stats[arm]
    S = np.concatenate([[1.0], st["mean"]])
    c = S[1:] / np.clip(S[:-1], 1e-9, None)
    d = cond_delta(st["mat"])
    dlo, dhi = boot_ci(st["mat"], cond_delta)
    verdict = ("FRONT-LOADED" if (float(dhi) < 0 and d < 0) else
               ("BACK-LOADED (c1 higher)" if float(dlo) > 0
                else "consistent with geometric"))
    report.append(f"| {ARM_LABEL[arm]} | " +
                  " | ".join(f"{x:.3f}" for x in c) +
                  f" | {d:+.3f} | [{float(dlo):+.3f},{float(dhi):+.3f}] "
                  f"| {verdict} |")
report.append("")

# ---- H2/H3 arm contrasts ----
report.append("## 3. H2/H3: arm contrasts (pre-registered rule)\n")
report.append("Bootstrap CI of the between-arm difference; chains resampled "
              "independently per arm.\n")
report.append("| Contrast | metric | diff | 95% CI | CI excludes 0? |")
report.append("|---|---|---|---|---|")


def contrast(a1, a2, statf, label, metric):
    m1, m2 = arm_stats[a1]["mat"], arm_stats[a2]["mat"]
    d = statf(m1) - statf(m2)
    vals = []
    for _ in range(NBOOT):
        i1 = RNG.integers(0, m1.shape[0], m1.shape[0])
        i2 = RNG.integers(0, m2.shape[0], m2.shape[0])
        vals.append(statf(m1[i1]) - statf(m2[i2]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    report.append(f"| {label} | {metric} | {d:+.3f} | [{lo:+.3f},{hi:+.3f}] "
                  f"| {'YES' if lo > 0 or hi < 0 else 'no'} |")


r5 = lambda m: m.mean(axis=0)[4]
sfit = lambda m: fit_s(m.mean(axis=0))
contrast("B", "A", r5, "H2: B(400w) - A(150w)", "round-5 survival")
contrast("B", "A", sfit, "H2: B(400w) - A(150w)", "fitted s")
contrast("C", "A", r5, "H3: C(verbatim) - A(plain)", "round-5 survival")
contrast("C", "A", sfit, "H3: C(verbatim) - A(plain)", "fitted s")
contrast("C", "B", r5, "C(150w+verbatim) - B(400w)", "round-5 survival")
report.append("")

# ---- corruption ----
report.append("## 4. Integrity loss (mutated codes) vs availability loss\n")
report.append("A mutated code = a token (6-10 alphanumerics) in the summary "
              "at Levenshtein distance 1-2 from a planted code that is absent "
              "verbatim.\n")
report.append("| Arm | round | absent (availability loss) | mutated (integrity loss) |")
report.append("|---|---|---|---|")
for arm in ARMS:
    sub = main[main.arm == arm]
    for r in ROUNDS:
        s = sub[sub['round'] == r]
        lost = int((1 - s.survived).sum())
        mut = int(s.mutated.sum())
        if r in (1, 5) or mut > 0:
            report.append(f"| {ARM_LABEL[arm]} | {r} | {lost}/96 | {mut} |")
mut_rows = main[main.mutated == 1]
report.append(f"\nTotal mutated-code observations across all arms/rounds: "
              f"{len(mut_rows)}.")
if len(mut_rows):
    ex = mut_rows.iloc[0]
    report.append(f"Example: planted {ex.code} -> summary token "
                  f"{ex.mutant_token} ({ex.chain_id} r{ex['round']}).")
report.append("")

# ---- position ----
report.append("## 5. Positional decay (needle position in round-0 source)\n")
report.append("Survival by original position bucket (word-offset terciles).\n")
report.append("| Arm | bucket | n facts | r1 survival | r5 survival |")
report.append("|---|---|---|---|---|")
for arm in ARMS:
    sub = main[main.arm == arm]
    for b in ["early", "middle", "late"]:
        bb = sub[sub.position_bucket == b]
        n = bb[bb['round'] == 1].shape[0]
        s1 = bb[bb['round'] == 1].survived.mean()
        s5 = bb[bb['round'] == 5].survived.mean()
        report.append(f"| {ARM_LABEL[arm]} | {b} | {n} | {s1:.3f} | {s5:.3f} |")
report.append("")

# ---- exploratory: secret-like vs neutral descriptors ----
SECRET_LIKE = {
    "staging database password", "production API key suffix",
    "VPN gateway auth token", "container registry token",
    "SSH deploy key label", "session recovery nonce",
    "webhook signing secret", "license activation code",
    "on-call routing key", "TLS certificate serial fragment"}
report.append("## 5b. Exploratory: secret-like vs neutral identifiers\n")
report.append("EXPLORATORY, not pre-registered: added after the pilot showed "
              "the model redacting credential values into a security warning "
              "during recompaction (defined before main-run data were "
              "scored). Descriptors split into secret-like (password/token/"
              "secret/key/nonce) vs neutral (ticket/incident/UID/tag/etc).\n")
report.append("| Arm | class | r1 survival | r5 survival |")
report.append("|---|---|---|---|")
main = main.assign(secretlike=main.descriptor.isin(SECRET_LIKE))
for arm in ARMS:
    sub = main[main.arm == arm]
    for cls, lbl in [(True, "secret-like"), (False, "neutral")]:
        cc = sub[sub.secretlike == cls]
        s1 = cc[cc['round'] == 1].survived.mean()
        s5 = cc[cc['round'] == 5].survived.mean()
        report.append(f"| {ARM_LABEL[arm]} | {lbl} | {s1:.3f} | {s5:.3f} |")
report.append("")

# ---- budget compliance ----
report.append("## 6. Budget compliance\n")
mc = calls[~calls.chain_id.str.startswith(("pilot", "repA0"))]
report.append("| Arm | budget | mean summary words | min | max | rounds over budget |")
report.append("|---|---|---|---|---|---|")
for arm in ARMS:
    s = mc[mc.arm == arm]
    over = int((s.summary_words > s.budget_words).sum())
    report.append(f"| {ARM_LABEL[arm]} | {int(s.budget_words.iloc[0])} | "
                  f"{s.summary_words.mean():.0f} | {int(s.summary_words.min())} "
                  f"| {int(s.summary_words.max())} | {over}/{len(s)} |")
report.append("")

# ---- chain-level variance + spot repeat ----
report.append("## 7. Chain-level variance and run-to-run stability\n")
report.append("Per-chain round-5 survival (out of 16 facts):\n")
for arm in ARMS:
    sub = main[(main.arm == arm) & (main['round'] == 5)]
    per = sub.groupby("chain_id").survived.sum().astype(int)
    vals = ", ".join(f"{cid}:{v}" for cid, v in per.items())
    report.append(f"- {ARM_LABEL[arm]}: {vals} "
                  f"(mean {per.mean():.2f}, sd {per.std():.2f})")
rep_ids = ["A-0", "repA0-rep1", "repA0-rep2"]
if set(rep_ids) <= set(df.chain_id.unique()):
    report.append("\nSpot-repeat: identical chain spec (A-0) run 3 times "
                  "(fresh model calls each time). Survival counts by round:\n")
    report.append("| run | r1 | r2 | r3 | r4 | r5 |")
    report.append("|---|---|---|---|---|---|")
    for cid in rep_ids:
        sub = df[df.chain_id == cid]
        cnt = [int(sub[sub['round'] == r].survived.sum()) for r in ROUNDS]
        report.append(f"| {cid} | " + " | ".join(map(str, cnt)) + " |")
report.append("")

# ---- pilot ----
report.append("## 8. Pilot and calibration gate\n")
psub = pilot
if len(psub):
    cnt = [int(psub[psub['round'] == r].survived.sum()) for r in ROUNDS]
    r1 = cnt[0] / 16
    report.append(f"Pilot chain (Arm A, seed 90000) surviving codes by round: "
                  f"{cnt} (of 16). Round-1 retention {r1:.3f}; "
                  f"pre-registered gate [0.15, 0.95] "
                  f"{'PASSED' if 0.15 <= r1 <= 0.95 else 'FAILED'}, "
                  f"main run proceeded without recalibration.")
report.append("")

# ---- probes ----
pr_path = os.path.join(BASE, "probes", "probe_results.json")
report.append("## 9. Validation probes (substring metric vs answerability, "
              "and silent failure)\n")
if os.path.exists(pr_path):
    pr = pd.DataFrame(json.load(open(pr_path)))
    ret = pr[pr.kind == "retained"]
    lost = pr[pr.kind.isin(["lost", "mutated"])]
    report.append(f"- Probed facts: {len(pr)} across 6 Arm-A chains "
                  f"({len(ret)} substring-retained, "
                  f"{len(lost)} lost/mutated).")
    if len(ret):
        report.append(f"- Retained codes answered exactly: "
                      f"{int(ret.exact_match.sum())}/{len(ret)}.")
    if len(lost):
        report.append(f"- Lost codes: abstained (UNKNOWN) "
                      f"{int(lost.abstained.sum())}/{len(lost)}; "
                      f"confabulated a code-like wrong answer "
                      f"{int(lost.codelike_wrong.sum())}/{len(lost)}.")
        mutp = pr[pr.kind == "mutated"]
        if len(mutp):
            bound = int((mutp.answer.str.upper() ==
                         mutp.mutant_in_summary.str.upper()).sum())
            report.append(f"- Mutated-code probes: {len(mutp)}; answered with "
                          f"the mutated (wrong) token {bound} times.")
report.append("")

# ---- cost/latency ----
report.append("## 10. Cost and latency (from raw JSON fields only)\n")
tot_cost = calls.total_cost_usd.sum()
lat = calls.duration_api_ms
report.append(f"- Chain calls logged: {len(calls)}. Total cost: "
              f"USD {tot_cost:.4f}. API latency per call: mean "
              f"{lat.mean()/1000:.1f}s, min {lat.min()/1000:.1f}s, max "
              f"{lat.max()/1000:.1f}s.")
probe_cost = 0.0
nprobe = 0
import glob as _g
for p in _g.glob(os.path.join(BASE, "probes", "probe-A-*.json")):
    d = json.load(open(p))
    probe_cost += d.get("total_cost_usd", 0)
    nprobe += 1
report.append(f"- Probe calls: {nprobe}, cost USD {probe_cost:.4f}.")
report.append(f"- Grand total LLM calls with retained raw JSON: "
              f"{len(calls) + nprobe}. Grand total cost: "
              f"USD {tot_cost + probe_cost:.4f}.")
report.append("")

# ---- plots ----
colors = {"A": "#d62728", "B": "#1f77b4", "C": "#2ca02c"}
plt.figure(figsize=(7, 4.5))
for arm in ARMS:
    st = arm_stats[arm]
    x = [0] + ROUNDS
    y = [1.0] + list(st["mean"])
    plt.plot(x, y, "-o", color=colors[arm], label=ARM_LABEL[arm])
    plt.fill_between(ROUNDS, st["lo"], st["hi"], color=colors[arm], alpha=0.15)
plt.axhline(0.5, ls=":", c="gray", lw=1)
plt.xlabel("compaction round")
plt.ylabel("fraction of planted codes surviving")
plt.title("Fact survival under iterated compaction (95% bootstrap CI)")
plt.ylim(0, 1.02)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE, "analysis", "retention_by_round.png"), dpi=160)
plt.close()

plt.figure(figsize=(7, 4.5))
for arm in ARMS:
    st = arm_stats[arm]
    plt.semilogy(ROUNDS, np.clip(st["mean"], 1e-3, None), "o",
                 color=colors[arm], label=ARM_LABEL[arm])
    r = np.linspace(0, 5, 50)
    plt.semilogy(r, st["s"] ** r, "-", color=colors[arm], alpha=0.6)
plt.xlabel("compaction round")
plt.ylabel("survival (log scale)")
plt.title("Geometric fit S(r) = s^r")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE, "analysis", "survival_fit.png"), dpi=160)
plt.close()

plt.figure(figsize=(7, 4))
for arm in ARMS:
    s = mc[mc.arm == arm]
    g = s.groupby("round").summary_words.mean()
    plt.plot(g.index, g.values, "-o", color=colors[arm], label=ARM_LABEL[arm])
plt.axhline(150, ls="--", c="#d62728", lw=1, alpha=0.5)
plt.axhline(400, ls="--", c="#1f77b4", lw=1, alpha=0.5)
plt.xlabel("round")
plt.ylabel("mean summary words")
plt.title("Budget compliance (dashed = budgets)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE, "analysis", "budget_compliance.png"), dpi=160)
plt.close()

plt.figure(figsize=(7, 4))
w = 0.25
for k, arm in enumerate(ARMS):
    sub = main[(main.arm == arm) & (main['round'] == 5)]
    ys = [sub[sub.position_bucket == b].survived.mean()
          for b in ["early", "middle", "late"]]
    plt.bar(np.arange(3) + (k - 1) * w, ys, width=w, color=colors[arm],
            label=ARM_LABEL[arm])
plt.xticks(range(3), ["early", "middle", "late"])
plt.ylabel("round-5 survival")
plt.title("Round-5 survival by original needle position")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE, "analysis", "position_survival.png"), dpi=160)
plt.close()

with open(os.path.join(BASE, "analysis.md"), "w") as f:
    f.write("\n".join(report))
print("analysis.md + 4 plots written")

# machine-readable summary for the paper
summary = {arm: {"S": list(map(float, arm_stats[arm]["mean"])),
                 "s": arm_stats[arm]["s"],
                 "s_ci": [arm_stats[arm]["slo"], arm_stats[arm]["shi"]],
                 "half_life": arm_stats[arm]["hl"]} for arm in ARMS}
with open(os.path.join(BASE, "analysis", "summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
