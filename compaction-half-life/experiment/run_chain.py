#!/usr/bin/env python3
"""Run one compaction chain: 5 sequential summarize-and-continue rounds.

Round 1 prompt: instruction + round-0 source.
Round r>1 prompt: instruction + previous summary + next distractor segment.
Every raw CLI JSON response is saved to results/<chain_id>-r<r>.json and the
exact prompt to prompts/<chain_id>-r<r>.txt.
"""
import json
import subprocess
import sys
import os
import time

ROUNDS = 5
MODEL = "haiku"
BASE = os.path.dirname(os.path.abspath(__file__))


def call_model(prompt, out_path, retries=2):
    for attempt in range(retries + 1):
        try:
            proc = subprocess.run(
                ["claude", "-p", "--model", MODEL, "--output-format", "json"],
                input=prompt, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                if data.get("result", "").strip():
                    with open(out_path, "w") as f:
                        f.write(proc.stdout)
                    return data["result"].strip()
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"model call failed after {retries + 1} attempts: {out_path}")


def build_prompt(spec, round_num, prev_summary):
    instr = spec["instruction"]
    if round_num == 1:
        body = spec["source"]
        return (f"{instr}\n\nOutput only the compressed context, nothing else."
                f"\n\n--- WORKING CONTEXT ---\n{body}")
    distractor = spec["distractors"][round_num - 2]
    return (f"{instr}\n\nOutput only the compressed context, nothing else."
            f"\n\n--- WORKING CONTEXT ---\n"
            f"Previous working summary:\n{prev_summary}\n\n"
            f"New activity since last summary:\n{distractor}")


def run(spec_path):
    with open(spec_path) as f:
        spec = json.load(f)
    cid = spec["chain_id"]
    prev = None
    for r in range(1, ROUNDS + 1):
        out_json = os.path.join(BASE, "results", f"{cid}-r{r}.json")
        if os.path.exists(out_json):
            with open(out_json) as f:
                prev = json.load(f)["result"].strip()
            print(f"[{cid}] round {r} cached", flush=True)
            continue
        prompt = build_prompt(spec, r, prev)
        with open(os.path.join(BASE, "prompts", f"{cid}-r{r}.txt"), "w") as f:
            f.write(prompt)
        t0 = time.time()
        prev = call_model(prompt, out_json)
        print(f"[{cid}] round {r} done in {time.time()-t0:.1f}s "
              f"({len(prev.split())} words)", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
