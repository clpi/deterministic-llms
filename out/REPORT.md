# PRLM report (rebuilt from the log)

Log: 4672 records, chain head bdb9e3cb59edc393.

## T1 continuation (35 prompts, 24 tokens)

| system | ref_match /24 | ref_f1 | rep3 | distinct2 | ms/token | calls/output |
| --- | --- | --- | --- | --- | --- | --- |
| samp-t1.0 | 0.75 | 0.156 | 0.009 | 0.975 | 44.8 | 24.0 |
| samp-t0.7 | 0.95 | 0.193 | 0.075 | 0.885 | 43.9 | 24.0 |
| samp-p0.9 | 0.78 | 0.160 | 0.022 | 0.952 | 50.1 | 24.0 |
| greedy-f32 | 1.11 | 0.202 | 0.147 | 0.802 | 43.1 | 24.0 |
| prlm-0 | 1.11 | 0.202 | 0.147 | 0.802 | 43.2 | 24.0 |
| prlm-theta | 0.83 | 0.151 | 0.001 | 0.975 | 43.2 | 24.0 |
| ctl-null | 1.11 | 0.202 | 0.147 | 0.802 | 44.5 | 24.0 |
| GAMED-oracle | 11.29 | 0.525 | 0.123 | 0.841 | 45.1 | 24.0 |

Paired differences against prlm-0 (mean, 95% bootstrap interval over prompts):

- greedy-f32 − prlm-0, ref_match: +0.000 [+0.000, +0.000] → greedy-f32 tie
- greedy-f32 − prlm-0, ref_f1: +0.000 [+0.000, +0.000] → greedy-f32 tie
- greedy-f32 − prlm-0, rep3: +0.000 [+0.000, +0.000] → greedy-f32 tie
- greedy-f32 − prlm-0, distinct2: +0.000 [+0.000, +0.000] → greedy-f32 tie
- prlm-theta − prlm-0, ref_match: -0.286 [-0.714, +0.086] → prlm-theta tie
- prlm-theta − prlm-0, ref_f1: -0.051 [-0.105, -0.006] → prlm-theta lower
- prlm-theta − prlm-0, rep3: -0.145 [-0.214, -0.084] → prlm-theta lower
- prlm-theta − prlm-0, distinct2: +0.173 [+0.119, +0.236] → prlm-theta higher
- samp-t1.0 − prlm-0, ref_match: -0.366 [-0.777, -0.017] → samp-t1.0 lower
- samp-t1.0 − prlm-0, ref_f1: -0.047 [-0.084, -0.015] → samp-t1.0 lower
- samp-t1.0 − prlm-0, rep3: -0.138 [-0.203, -0.077] → samp-t1.0 lower
- samp-t1.0 − prlm-0, distinct2: +0.173 [+0.115, +0.240] → samp-t1.0 higher
- samp-t0.7 − prlm-0, ref_match: -0.166 [-0.520, +0.097] → samp-t0.7 tie
- samp-t0.7 − prlm-0, ref_f1: -0.010 [-0.037, +0.017] → samp-t0.7 tie
- samp-t0.7 − prlm-0, rep3: -0.072 [-0.122, -0.029] → samp-t0.7 lower
- samp-t0.7 − prlm-0, distinct2: +0.083 [+0.033, +0.135] → samp-t0.7 higher
- samp-p0.9 − prlm-0, ref_match: -0.337 [-0.771, -0.017] → samp-p0.9 lower
- samp-p0.9 − prlm-0, ref_f1: -0.042 [-0.087, -0.003] → samp-p0.9 lower
- samp-p0.9 − prlm-0, rep3: -0.125 [-0.194, -0.067] → samp-p0.9 lower
- samp-p0.9 − prlm-0, distinct2: +0.149 [+0.091, +0.213] → samp-p0.9 higher

prlm-0 equals greedy-f32 on 35/35 prompts; 840/840 steps certified (rest accepted ties). Do-nothing control equals prlm-0 on 35/35.

## T2 closed cloze (40 items)

| system | accuracy | per-seed range |
| --- | --- | --- |
| samp-t1.0 | 0.230 | 0.150 to 0.375 |
| samp-t0.7 | 0.345 | 0.250 to 0.450 |
| samp-p0.9 | 0.265 | 0.200 to 0.325 |
| greedy-f32 | 0.450 |  |
| prlm-0 | 0.450 |  |
| ctl-null | 0.450 |  |
| GAMED-oracle | 0.900 |  |
| alt-beam4 | 0.350 |  |
| alt-bestfirst | 0.350 |  |
| alt-ngram | 0.025 |  |
| alt-rules | 0.450 |  |

greedy-f32 − samp-p0.9 accuracy: +0.185 [+0.085, +0.300] → higher

## T3 control (20 prompts per kind, 24 tokens)

| kind | system | success | held-out hit (topic) | NLL/token under base |
| --- | --- | --- | --- | --- |
| ban | samp-p0.9 | 0.14 |  | 2.99 |
| ban | greedy-f32 | 0.10 |  | 1.07 |
| ban | prlm-0 | 0.10 |  | 1.07 |
| ban | ctl-null | 0.10 |  | 1.07 |
| ban | prlm-theta:ban (CORRECTED: sign bug, boosted banned words) | 0.10 |  | 1.72 |
| ban | prlm-theta:ban-fixed | 1.00 |  | 1.21 |
| ban | prlm-theta:ban-soft (CORRECTED: sign bug, boosted banned words) | 0.05 |  | 1.66 |
| ban | prlm-theta:ban-soft-fixed | 0.95 |  | 1.17 |
| ban | samp-p0.9 best-of-5 with checker | 0.45 |  | mean tries 4.00 (×24 calls) |
| topic | samp-p0.9 | 0.00 | 0.01 | 2.99 |
| topic | greedy-f32 | 0.00 | 0.00 | 1.07 |
| topic | prlm-0 | 0.00 | 0.00 | 1.07 |
| topic | ctl-null | 0.00 | 0.00 | 1.07 |
| topic | prlm-theta:embdir | 0.75 | 0.00 | 2.09 |
| topic | prlm-theta:embdir-rand | 0.00 | 0.00 | 1.21 |
| topic | prlm-theta:topic | 0.00 | 0.00 | 1.07 |
| topic | GAMED-oracle | 0.00 | 0.20 | 1.10 |
| topic | samp-p0.9 best-of-5 with checker | 0.00 |  | mean tries 5.00 (×24 calls) |
| end | samp-p0.9 | 0.52 |  | 2.99 |
| end | greedy-f32 | 0.85 |  | 1.07 |
| end | prlm-0 | 0.85 |  | 1.07 |
| end | ctl-null | 0.85 |  | 1.07 |
| end | prlm-theta:end | 1.00 |  | 1.13 |
| end | samp-p0.9 best-of-5 with checker | 1.00 |  | mean tries 2.35 (×24 calls) |

## On-the-fly weighting (post hoc; chosen on DEV, every candidate scored on TEST)

| candidate | DEV ref_match | DEV rep3 | TEST ref_match | TEST rep3 | TEST distinct2 |
| --- | --- | --- | --- | --- | --- |
| greedy | 0.85 | 0.123 | 1.11 | 0.147 | 0.802 |
| gated:rep1.5+noloop4 | 0.85 | 0.025 | 1.11 | 0.029 | 0.918 |
| noloop4 | 0.90 | 0.011 | 1.00 | 0.008 | 0.932 |
| rep0.5 | 0.80 | 0.066 | 1.09 | 0.091 | 0.878 |
| rep1.5+noloop4 | 0.65 | 0.009 | 0.83 | 0.001 | 0.975 |
- TEST gated:rep1.5+noloop4 − prlm-0, ref_match: +0.000 [-0.086, +0.086] → tie
- TEST gated:rep1.5+noloop4 − prlm-0, rep3: -0.118 [-0.186, -0.058] → lower
- TEST noloop4 − prlm-0, ref_match: -0.114 [-0.286, +0.029] → tie
- TEST noloop4 − prlm-0, rep3: -0.139 [-0.209, -0.078] → lower
- TEST rep0.5 − prlm-0, ref_match: -0.029 [-0.457, +0.343] → tie
- TEST rep0.5 − prlm-0, rep3: -0.056 [-0.109, +0.000] → tie
- TEST rep1.5+noloop4 − prlm-0, ref_match: -0.286 [-0.743, +0.086] → tie
- TEST rep1.5+noloop4 − prlm-0, rep3: -0.145 [-0.217, -0.083] → lower

## Reproducibility (T1)

- prlm-0: mean distinct outputs per prompt 1.00, max 1 (runs 1 to 3 from an empty cache, run 4 at 1 thread)
- greedy-f32: mean distinct outputs per prompt 1.00, max 1 (runs 1 to 3 from an empty cache)
- samp-p0.9: mean distinct outputs per prompt 5.00, max 5 (seeds 0 to 4 (unseeded serving))
- samp-p0.9-seed0-rerun: mean distinct outputs per prompt 1.00, max 1 (seed 0 rerun three times)
- logits bitwise equal at 1 and 4 threads: 35/35 prompts
- seeded sampling seed 0 equals its rerun on 35/35 prompts

## Alternative constructions

| system | items | certified | mean calls | mean NLL of output (beam/best-first) | equals greedy |
| --- | --- | --- | --- | --- | --- |
| alt-beam4 | 75 | 12 | 20.5 | 6.36 | 33/75 |
| alt-bestfirst | 75 | 45 | 107.7 | 4.36 | 51/75 |
| alt-greedy-short | 75 | n/a | 5.9 | | |

Best-first certified a sequence different from greedy on 24 items: each is, by its certificate, the most likely sequence of its length, so greedy's is less likely there (t1/ast, t1/difflib, t1/operator, t1/posixpath, t1/smtplib, t1/socketserver, t2/2, t2/3 ...). Beam-4's NLL column is its own summed score; beam certified only 12 items because its bound (the best pruned prefix) is loose.

Trigram automaton on T1: ref_match 0.49/24, ref_f1 0.133, rep3 0.110, ms/token 0.004.
Rule table (495 rules): fired on 4/840 T1 steps (calls saved), output equals greedy on 34/35 prompts, 830/840 tokens equal at position.

## Weight sweeps on T3 topic prompts (dose response, reweighting cost)

### topic

| weight (logits) | hit TRAIN words | hit HELD-OUT words | NLL/token |
| --- | --- | --- | --- |
| 0 | 0.00 | 0.00 | 1.07 |
| 1 | 0.00 | 0.00 | 1.07 |
| 2 | 0.00 | 0.00 | 1.07 |
| 3 | 0.00 | 0.00 | 1.07 |
| 4 | 0.00 | 0.00 | 1.07 |
| 6 | 0.05 | 0.00 | 1.09 |
| 8 | 0.55 | 0.00 | 1.58 |

Incremental (content-addressed reuse) vs scratch: 694 vs 3360 model calls; identical outputs 140/140.

### embdir

| weight (logits) | hit TRAIN words | hit HELD-OUT words | NLL/token |
| --- | --- | --- | --- |
| 0 | 0.00 | 0.00 | 1.07 |
| 0.005 | 0.00 | 0.00 | 1.18 |
| 0.01 | 0.05 | 0.05 | 1.39 |
| 0.02 | 0.75 | 0.00 | 2.09 |
| 0.04 | 1.00 | 0.00 | 2.61 |

Incremental (content-addressed reuse) vs scratch: 1905 vs 2400 model calls; identical outputs 100/100.

## Lean replay of recorded steps

- t1-prlm0.bin: exit 0, steps 840, harness pick = proved pick 840, rescored pick = harness recompute 840, rescored = recomputed 840, certified 840, cone box held 17 (correct 17), pick changed by the weight change 406 (15.61 s)
- t3-theta.bin: exit 0, steps 1440, harness pick = proved pick 1440, rescored pick = harness recompute 1440, rescored = recomputed 1440, certified 1440, cone box held 45 (correct 45), pick changed by the weight change 188 (25.83 s)

## Pareto front over T1 systems (minimize; Lean `prlm front`)

Objectives: (24 − ref_match)×100, rep3×1000, distinct outputs per prompt ×100, ms/token ×10, calls per token ×100.

    samp-t1.0 2325 9 500 449 100
    samp-t0.7 2305 75 494 445 100
    samp-p0.9 2322 22 500 485 100
    greedy-f32 2289 147 100 431 100
    prlm-0 2289 147 100 432 100
    prlm-theta 2317 1 100 432 100
    alt-ngram 2351 110 100 0 0
    alt-rules 2289 147 100 424 100

With latency as 10% buckets (the contract's tie rule):

    front samp-t0.7 [2305, 75, 494, 39, 100] chebmin [samp-t0.7]
    front greedy-f32 [2289, 147, 100, 39, 100] chebmin [greedy-f32, prlm-0, alt-rules]
    front prlm-0 [2289, 147, 100, 39, 100] chebmin [greedy-f32, prlm-0, alt-rules]
    front prlm-theta [2317, 1, 100, 39, 100] chebmin [prlm-theta]
    front alt-ngram [2351, 110, 100, 0, 0] chebmin [alt-ngram]
    front alt-rules [2289, 147, 100, 39, 100] chebmin [greedy-f32, prlm-0, alt-rules]

Front members and the rows that minimize Chebyshev weights solved from each:

    front samp-t0.7 [2305, 75, 494, 445, 100] chebmin [samp-t0.7]
    front prlm-theta [2317, 1, 100, 432, 100] chebmin [prlm-theta]
    front alt-ngram [2351, 110, 100, 0, 0] chebmin [alt-ngram]
    front alt-rules [2289, 147, 100, 424, 100] chebmin [alt-rules]
