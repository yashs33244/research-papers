# Analysis: The Compaction Half-Life

All numbers below are computed by analyze.py from results.csv, calls.csv, probes/probe_results.json and the raw per-call JSON files in results/. Nothing is hand-entered.

## 1. Survival curves S(r) by arm

Retention = fraction of the 96 planted codes (16 facts x 6 chains) present verbatim (case-insensitive substring) in the round-r summary. CI = 95 percent bootstrap over chains (10,000 reps).

| Arm | r1 | r2 | r3 | r4 | r5 | fitted s | half-life (rounds) |
|---|---|---|---|---|---|---|---|
| A (150w) | 0.333 [0.135,0.615] | 0.250 [0.031,0.562] | 0.198 [0.000,0.531] | 0.031 [0.000,0.094] | 0.031 [0.000,0.094] | 0.484 [0.193,0.589] | 0.95 |
| B (400w) | 0.729 [0.500,0.927] | 0.417 [0.094,0.740] | 0.125 [0.000,0.292] | 0.083 [0.000,0.250] | 0.083 [0.000,0.250] | 0.573 [0.208,0.733] | 1.24 |
| C (150w+verbatim) | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | inf |

Raw surviving-code counts out of 96 per cell:

| Arm | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| A (150w) | 32 | 24 | 19 | 3 | 3 |
| B (400w) | 70 | 40 | 12 | 8 | 8 |
| C (150w+verbatim) | 96 | 96 | 96 | 96 | 96 |

## 2. H1: geometric vs front-loaded decay (pre-registered rule)

delta = c_1 - mean(c_2..c_5) where c_r = S(r)/S(r-1) on chain means; FRONT-LOADED iff 95 percent bootstrap CI of delta excludes 0 and delta < 0.

| Arm | c1 | c2 | c3 | c4 | c5 | delta | 95% CI | verdict |
|---|---|---|---|---|---|---|---|---|
| A (150w) | 0.333 | 0.750 | 0.792 | 0.158 | 1.000 | -0.342 | [-0.704,+0.175] | consistent with geometric |
| B (400w) | 0.729 | 0.571 | 0.300 | 0.667 | 1.000 | +0.095 | [-0.131,+0.719] | consistent with geometric |
| C (150w+verbatim) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 | [+0.000,+0.000] | consistent with geometric |

## 3. H2/H3: arm contrasts (pre-registered rule)

Bootstrap CI of the between-arm difference; chains resampled independently per arm.

| Contrast | metric | diff | 95% CI | CI excludes 0? |
|---|---|---|---|---|
| H2: B(400w) - A(150w) | round-5 survival | +0.052 | [-0.062,+0.219] | no |
| H2: B(400w) - A(150w) | fitted s | +0.089 | [-0.321,+0.457] | no |
| H3: C(verbatim) - A(plain) | round-5 survival | +0.969 | [+0.906,+1.000] | YES |
| H3: C(verbatim) - A(plain) | fitted s | +0.516 | [+0.411,+0.807] | YES |
| C(150w+verbatim) - B(400w) | round-5 survival | +0.917 | [+0.750,+1.000] | YES |

## 4. Integrity loss (mutated codes) vs availability loss

A mutated code = a token (6-10 alphanumerics) in the summary at Levenshtein distance 1-2 from a planted code that is absent verbatim.

| Arm | round | absent (availability loss) | mutated (integrity loss) |
|---|---|---|---|
| A (150w) | 1 | 64/96 | 0 |
| A (150w) | 5 | 93/96 | 0 |
| B (400w) | 1 | 26/96 | 0 |
| B (400w) | 5 | 88/96 | 0 |
| C (150w+verbatim) | 1 | 0/96 | 0 |
| C (150w+verbatim) | 5 | 0/96 | 0 |

Total mutated-code observations across all arms/rounds: 0.

## 5. Positional decay (needle position in round-0 source)

Survival by original position bucket (word-offset terciles).

| Arm | bucket | n facts | r1 survival | r5 survival |
|---|---|---|---|---|
| A (150w) | early | 30 | 0.433 | 0.100 |
| A (150w) | middle | 36 | 0.306 | 0.000 |
| A (150w) | late | 30 | 0.267 | 0.000 |
| B (400w) | early | 30 | 0.767 | 0.133 |
| B (400w) | middle | 36 | 0.778 | 0.056 |
| B (400w) | late | 30 | 0.633 | 0.067 |
| C (150w+verbatim) | early | 30 | 1.000 | 1.000 |
| C (150w+verbatim) | middle | 36 | 1.000 | 1.000 |
| C (150w+verbatim) | late | 30 | 1.000 | 1.000 |

## 5b. Exploratory: secret-like vs neutral identifiers

EXPLORATORY, not pre-registered: added after the pilot showed the model redacting credential values into a security warning during recompaction (defined before main-run data were scored). Descriptors split into secret-like (password/token/secret/key/nonce) vs neutral (ticket/incident/UID/tag/etc).

| Arm | class | r1 survival | r5 survival |
|---|---|---|---|
| A (150w) | secret-like | 0.375 | 0.000 |
| A (150w) | neutral | 0.292 | 0.062 |
| B (400w) | secret-like | 0.896 | 0.042 |
| B (400w) | neutral | 0.562 | 0.125 |
| C (150w+verbatim) | secret-like | 1.000 | 1.000 |
| C (150w+verbatim) | neutral | 1.000 | 1.000 |

## 6. Budget compliance

| Arm | budget | mean summary words | min | max | rounds over budget |
|---|---|---|---|---|---|
| A (150w) | 150 | 99 | 40 | 165 | 1/30 |
| B (400w) | 400 | 214 | 85 | 311 | 0/30 |
| C (150w+verbatim) | 150 | 119 | 89 | 174 | 3/30 |

## 7. Chain-level variance and run-to-run stability

Per-chain round-5 survival (out of 16 facts):

- A (150w): A-0:0, A-1:3, A-2:0, A-3:0, A-4:0, A-5:0 (mean 0.50, sd 1.22)
- B (400w): B-0:0, B-1:8, B-2:0, B-3:0, B-4:0, B-5:0 (mean 1.33, sd 3.27)
- C (150w+verbatim): C-0:16, C-1:16, C-2:16, C-3:16, C-4:16, C-5:16 (mean 16.00, sd 0.00)

Spot-repeat: identical chain spec (A-0) run 3 times (fresh model calls each time). Survival counts by round:

| run | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| A-0 | 16 | 16 | 16 | 0 | 0 |
| repA0-rep1 | 0 | 0 | 0 | 0 | 0 |
| repA0-rep2 | 4 | 2 | 2 | 2 | 2 |

## 8. Pilot and calibration gate

Pilot chain (Arm A, seed 90000) surviving codes by round: [15, 0, 0, 0, 0] (of 16). Round-1 retention 0.938; pre-registered gate [0.15, 0.95] PASSED, main run proceeded without recalibration.

## 9. Validation probes (substring metric vs answerability, and silent failure)

- Probed facts: 18 across 6 Arm-A chains (1 substring-retained, 17 lost/mutated).
- Retained codes answered exactly: 1/1.
- Lost codes: abstained (UNKNOWN) 17/17; confabulated a code-like wrong answer 0/17.

## 10. Cost and latency (from raw JSON fields only)

- Chain calls logged: 105. Total cost: USD 3.3819. API latency per call: mean 14.4s, min 6.4s, max 30.0s.
- Probe calls: 6, cost USD 0.1412.
- Grand total LLM calls with retained raw JSON: 111. Grand total cost: USD 3.5230.
