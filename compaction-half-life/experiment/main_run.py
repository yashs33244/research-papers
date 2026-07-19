#!/usr/bin/env python3
"""Run many chains in parallel (4 workers); each chain is sequential inside."""
import sys
import glob
import concurrent.futures as cf
import run_chain

specs = sorted(glob.glob(sys.argv[1]))
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
print(f"{len(specs)} chains, {workers} workers", flush=True)
fails = []
with cf.ThreadPoolExecutor(max_workers=workers) as ex:
    futs = {ex.submit(run_chain.run, s): s for s in specs}
    for fut in cf.as_completed(futs):
        try:
            fut.result()
        except Exception as e:
            fails.append((futs[fut], str(e)))
            print(f"FAIL {futs[fut]}: {e}", flush=True)
print(f"done, {len(fails)} failures", flush=True)
sys.exit(1 if fails else 0)
