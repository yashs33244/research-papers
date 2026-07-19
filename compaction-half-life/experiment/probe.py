#!/usr/bin/env python3
"""Validation probes: for each Arm A main chain, ask 3 questions against the
round-5 summary only (1 call per chain, 6 calls total). Mix of substring-
retained and substring-lost codes. Tests (a) substring metric predicts
answerability, (b) lost facts: abstain vs confabulate."""
import json
import os
import re
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
from score import score_summary


def call_model(prompt, out_path):
    proc = subprocess.run(
        ["claude", "-p", "--model", "haiku", "--output-format", "json"],
        input=prompt, capture_output=True, text=True, timeout=300)
    with open(out_path, "w") as f:
        f.write(proc.stdout)
    return json.loads(proc.stdout)["result"].strip()


def main():
    results = []
    for ci in range(6):
        cid = f"A-{ci}"
        spec = json.load(open(os.path.join(BASE, "chains", f"{cid}.spec.json")))
        summary = json.load(open(os.path.join(
            BASE, "results", f"{cid}-r5.json")))["result"].strip()
        per_fact = score_summary(summary, spec["facts"])
        retained = [i for i, x in enumerate(per_fact) if x["survived"]]
        mutated = [i for i, x in enumerate(per_fact)
                   if x["mutated"] and not x["survived"]]
        lost = [i for i, x in enumerate(per_fact)
                if not x["survived"] and not x["mutated"]]
        picks = []
        if retained:
            picks.append(("retained", retained[0]))
        if mutated:
            picks.append(("mutated", mutated[0]))
        for i in lost:
            if len(picks) >= 3:
                break
            picks.append(("lost", i))
        while len(picks) < 3 and retained:
            for i in retained[1:]:
                if len(picks) >= 3:
                    break
                picks.append(("retained", i))
            break
        picks = picks[:3]
        qlines = "\n".join(
            f"{n+1}) What is the {spec['facts'][i]['descriptor']}?"
            for n, (_, i) in enumerate(picks))
        prompt = (
            "Given ONLY this summary of earlier work, answer the questions. "
            "Each answer is a short code. If the information is not in the "
            "summary, answer exactly UNKNOWN. Reply with exactly 3 lines in "
            "the form '1: <answer>'.\n\n--- SUMMARY ---\n" + summary +
            "\n\n--- QUESTIONS ---\n" + qlines)
        out_path = os.path.join(BASE, "probes", f"probe-{cid}.json")
        with open(os.path.join(BASE, "probes", f"probe-{cid}-prompt.txt"), "w") as f:
            f.write(prompt)
        ans = call_model(prompt, out_path)
        lines = [l.strip() for l in ans.splitlines() if l.strip()]
        for n, (kind, i) in enumerate(picks):
            fact = spec["facts"][i]
            got = ""
            for l in lines:
                m = re.match(rf"^{n+1}\s*[:).]\s*(.*)$", l)
                if m:
                    got = m.group(1).strip().strip(".")
                    break
            exact = got.upper() == fact["code"].upper()
            is_unknown = got.upper() == "UNKNOWN"
            codelike = bool(re.fullmatch(r"[A-Za-z0-9-]{4,12}", got)) and not is_unknown
            results.append({
                "chain_id": cid, "kind": kind, "descriptor": fact["descriptor"],
                "true_code": fact["code"],
                "mutant_in_summary": per_fact[i]["mutant"],
                "answer": got, "exact_match": exact,
                "abstained": is_unknown, "codelike_wrong": codelike and not exact,
            })
        print(f"[{cid}] probed {len(picks)} facts", flush=True)
    with open(os.path.join(BASE, "probes", "probe_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("probe results written")


if __name__ == "__main__":
    main()
