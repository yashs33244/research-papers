#!/usr/bin/env python3
"""Score retention/corruption from raw result JSONs. Emits results.csv and
summary tables used by analyze.py. All numbers derive from files on disk."""
import json
import glob
import os
import re
import csv

BASE = os.path.dirname(os.path.abspath(__file__))
TOKEN_RE = re.compile(r"[A-Za-z0-9]{6,10}")


def lev(a, b, cap=3):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_summary(summary, facts):
    """Return per-fact dict: survived (verbatim substring), mutated (near-miss)."""
    text = re.sub(r"\s+", " ", summary).upper()
    tokens = set(t.upper() for t in TOKEN_RE.findall(summary))
    planted = {f["code"].upper() for f in facts}
    out = []
    for f in facts:
        code = f["code"].upper()
        survived = code in text
        mutated = False
        mutant = ""
        if not survived:
            for t in tokens:
                if t in planted:
                    continue
                if lev(t, code, cap=2) <= 2:
                    mutated = True
                    mutant = t
                    break
        out.append({"survived": survived, "mutated": mutated, "mutant": mutant})
    return out


def main():
    rows = []
    call_rows = []
    for spec_path in sorted(glob.glob(os.path.join(BASE, "chains", "*.spec.json"))):
        spec = json.load(open(spec_path))
        cid = spec["chain_id"]
        for r in range(1, 6):
            rp = os.path.join(BASE, "results", f"{cid}-r{r}.json")
            if not os.path.exists(rp):
                continue
            data = json.load(open(rp))
            summary = data["result"].strip()
            per_fact = score_summary(summary, spec["facts"])
            call_rows.append({
                "chain_id": cid, "arm": spec["arm"], "round": r,
                "summary_words": len(summary.split()),
                "budget_words": spec["budget_words"],
                "total_cost_usd": data.get("total_cost_usd"),
                "duration_api_ms": data.get("duration_api_ms"),
                "n_survived": sum(x["survived"] for x in per_fact),
                "n_mutated": sum(x["mutated"] for x in per_fact),
            })
            for fi, (f, x) in enumerate(zip(spec["facts"], per_fact)):
                rows.append({
                    "chain_id": cid, "arm": spec["arm"],
                    "chain_idx": spec["chain_idx"], "round": r,
                    "fact_idx": fi, "descriptor": f["descriptor"],
                    "code": f["code"],
                    "position_frac": f["position_frac"],
                    "position_bucket": f["position_bucket"],
                    "survived": int(x["survived"]),
                    "mutated": int(x["mutated"]),
                    "mutant_token": x["mutant"],
                })
    with open(os.path.join(BASE, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(BASE, "calls.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(call_rows[0].keys()))
        w.writeheader()
        w.writerows(call_rows)
    print(f"wrote {len(rows)} fact-rows, {len(call_rows)} call-rows")


if __name__ == "__main__":
    main()
