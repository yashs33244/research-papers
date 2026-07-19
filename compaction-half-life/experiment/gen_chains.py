#!/usr/bin/env python3
"""Generate chain specs for the compaction half-life experiment.

Each chain spec contains:
- 16 atomic facts "The <descriptor> is <CODE>" with random 8-char codes
- a ~2800-word round-0 source (facts interleaved into task-log filler)
- 5 fact-free distractor segments (~1200 words each)
- the arm's compaction instruction

All randomness is seeded; seeds recorded in config.json.
"""
import json
import random
import argparse
import os

# Codes avoid ambiguous chars (0/O, 1/I/L) so scoring is unambiguous.
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

DESCRIPTORS = [
    "staging database password",
    "production API key suffix",
    "deploy ticket reference",
    "VPN gateway auth token",
    "S3 backup bucket suffix",
    "feature flag rollout ID",
    "incident tracking number",
    "container registry token",
    "SSH deploy key label",
    "monitoring dashboard UID",
    "message queue topic suffix",
    "session recovery nonce",
    "webhook signing secret",
    "license activation code",
    "build artifact hash prefix",
    "customer account reference",
    "on-call routing key",
    "TLS certificate serial fragment",
    "nightly cron job identifier",
    "rollback release tag",
]

SERVICES = ["auth-service", "billing-worker", "ingest-api", "report-builder",
            "notification-daemon", "search-indexer", "payment-gateway",
            "user-profile-service", "audit-logger", "cache-warmer",
            "scheduler-core", "export-pipeline", "webhook-relay",
            "metrics-collector", "session-manager"]
ENVS = ["staging", "production", "the QA cluster", "the dev sandbox",
        "the canary environment", "the blue pool", "the green pool"]
ACTIONS = ["restarted", "redeployed", "scaled up", "scaled down", "rolled back",
           "paused", "resumed", "reconfigured", "migrated", "drained",
           "health-checked", "patched", "upgraded", "downgraded", "inspected"]
OUTCOMES = ["completed without errors", "took longer than expected",
            "emitted a handful of warnings", "passed all smoke tests",
            "required a manual retry", "recovered after a brief stall",
            "showed normal latency afterwards", "cleared the alert",
            "left the queue depth unchanged", "reduced memory pressure",
            "stabilized the error rate", "finished ahead of schedule"]
TIMES = ["around 09:15", "just before lunch", "mid-afternoon", "at 14:40",
         "late in the evening shift", "shortly after standup", "at 11:05",
         "near the end of the sprint day", "at 16:20", "before the sync call"]
NOTES = [
    "Logs were archived to the usual location for later review.",
    "No customer impact was observed during the window.",
    "The dashboard confirmed the change within a few minutes.",
    "A follow-up task was filed to tidy the configuration.",
    "The team agreed to revisit the thresholds next week.",
    "Grafana panels showed the expected dip and recovery.",
    "The retry queue drained back to zero shortly after.",
    "Documentation was updated to reflect the new procedure.",
    "The change was announced in the operations channel.",
    "A brief postmortem note was added to the runbook.",
    "CPU utilization returned to baseline within the hour.",
    "The pager stayed quiet for the rest of the shift.",
]


def make_code(rng):
    return "".join(rng.choice(CODE_ALPHABET) for _ in range(8))


def filler_sentence(rng):
    s = rng.choice(SERVICES)
    e = rng.choice(ENVS)
    a = rng.choice(ACTIONS)
    o = rng.choice(OUTCOMES)
    t = rng.choice(TIMES)
    n = rng.choice(NOTES)
    return f"{t.capitalize()}, {s} in {e} was {a} and the operation {o}. {n}"


def filler_block(rng, target_words):
    words = 0
    out = []
    while words < target_words:
        sent = filler_sentence(rng)
        out.append(sent)
        words += len(sent.split())
    return " ".join(out)


def build_source(rng, facts, target_words=2800):
    """Interleave 16 fact sentences into filler; record word positions."""
    n_blocks = len(facts) + 1
    per_block = max(30, (target_words - 16 * 9) // n_blocks)
    blocks = [filler_block(rng, per_block) for _ in range(n_blocks)]
    order = list(range(len(facts)))
    rng.shuffle(order)
    parts = []
    word_count = 0
    fact_positions = {}
    for i in range(n_blocks):
        parts.append(blocks[i])
        word_count += len(blocks[i].split())
        if i < len(facts):
            fi = order[i]
            f = facts[fi]
            sent = f"Note for the record: the {f['descriptor']} is {f['code']}."
            parts.append(sent)
            fact_positions[fi] = word_count
            word_count += len(sent.split())
    total = word_count
    for fi, pos in fact_positions.items():
        frac = pos / total
        facts[fi]["position_word"] = pos
        facts[fi]["position_frac"] = round(frac, 4)
        facts[fi]["position_bucket"] = (
            "early" if frac < 1 / 3 else ("middle" if frac < 2 / 3 else "late"))
    return "\n\n".join(parts), total


ARMS = {
    "A": {"budget": 150,
          "instruction": ("Compress the following working context into at most "
                          "150 words, preserving everything needed to continue "
                          "the task.")},
    "B": {"budget": 400,
          "instruction": ("Compress the following working context into at most "
                          "400 words, preserving everything needed to continue "
                          "the task.")},
    "C": {"budget": 150,
          "instruction": ("Compress the following working context into at most "
                          "150 words, preserving everything needed to continue "
                          "the task. Preserve all identifiers, codes, numbers, "
                          "and proper nouns verbatim.")},
}


def gen_chain(arm, chain_idx, base_seed, outdir, tag=None):
    # Same chain_idx across arms shares the fact/filler seed, so arms differ
    # only in instruction/budget.
    content_seed = base_seed + chain_idx
    rng = random.Random(content_seed)
    used = set()
    descs = rng.sample(DESCRIPTORS, 16)
    facts = []
    for d in descs:
        c = make_code(rng)
        while c in used:
            c = make_code(rng)
        used.add(c)
        facts.append({"descriptor": d, "code": c})
    source, total_words = build_source(rng, facts)
    distractors = [filler_block(rng, 1200) for _ in range(5)]
    chain_id = tag or f"{arm}-{chain_idx}"
    spec = {
        "chain_id": chain_id,
        "arm": arm,
        "chain_idx": chain_idx,
        "content_seed": content_seed,
        "budget_words": ARMS[arm]["budget"],
        "instruction": ARMS[arm]["instruction"],
        "facts": facts,
        "source": source,
        "source_words": total_words,
        "distractors": distractors,
    }
    path = os.path.join(outdir, f"{chain_id}.spec.json")
    with open(path, "w") as f:
        json.dump(spec, f, indent=1)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "main", "repeat"], required=True)
    ap.add_argument("--outdir", default="chains")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.mode == "pilot":
        # Pilot uses its own seed space so it never overlaps main chains.
        p = gen_chain("A", 0, base_seed=90000, outdir=args.outdir, tag="pilot-A-0")
        print(p)
    elif args.mode == "main":
        for arm in ["A", "B", "C"]:
            for ci in range(6):
                p = gen_chain(arm, ci, base_seed=42000, outdir=args.outdir)
                print(p)
    elif args.mode == "repeat":
        # Spot-repeat: same spec as A-0 but new chain_id (identical prompts).
        for rep in [1, 2]:
            p = gen_chain("A", 0, base_seed=42000, outdir=args.outdir,
                          tag=f"repA0-rep{rep}")
            print(p)


if __name__ == "__main__":
    main()
