# PRLM report (rebuilt from the log)

Log: 662 records, chain head 6b5e42e890dacf2b.

## T1 continuation (35 prompts, 24 tokens)

| system | ref_match /24 | ref_f1 | rep3 | distinct2 | ms/token | calls/output |
| --- | --- | --- | --- | --- | --- | --- |

Paired differences against prlm-0 (mean, 95% bootstrap interval over prompts):

- greedy-f32 − prlm-0, ref_match: +0.000 [+0.000, +0.000] → greedy-f32 tie
- greedy-f32 − prlm-0, ref_f1: +0.000 [+0.000, +0.000] → greedy-f32 tie
- greedy-f32 − prlm-0, rep3: +0.000 [+0.000, +0.000] → greedy-f32 tie
- greedy-f32 − prlm-0, distinct2: +0.000 [+0.000, +0.000] → greedy-f32 tie
- prlm-theta − prlm-0, ref_match: +0.000 [+0.000, +0.000] → prlm-theta tie
- prlm-theta − prlm-0, ref_f1: +0.000 [+0.000, +0.000] → prlm-theta tie
- prlm-theta − prlm-0, rep3: +0.000 [+0.000, +0.000] → prlm-theta tie
- prlm-theta − prlm-0, distinct2: +0.000 [+0.000, +0.000] → prlm-theta tie
- samp-t1.0 − prlm-0, ref_match: +0.000 [+0.000, +0.000] → samp-t1.0 tie
- samp-t1.0 − prlm-0, ref_f1: +0.000 [+0.000, +0.000] → samp-t1.0 tie
- samp-t1.0 − prlm-0, rep3: +0.000 [+0.000, +0.000] → samp-t1.0 tie
- samp-t1.0 − prlm-0, distinct2: +0.000 [+0.000, +0.000] → samp-t1.0 tie
- samp-t0.7 − prlm-0, ref_match: +0.000 [+0.000, +0.000] → samp-t0.7 tie
- samp-t0.7 − prlm-0, ref_f1: +0.000 [+0.000, +0.000] → samp-t0.7 tie
- samp-t0.7 − prlm-0, rep3: +0.000 [+0.000, +0.000] → samp-t0.7 tie
- samp-t0.7 − prlm-0, distinct2: +0.000 [+0.000, +0.000] → samp-t0.7 tie
- samp-p0.9 − prlm-0, ref_match: +0.000 [+0.000, +0.000] → samp-p0.9 tie
- samp-p0.9 − prlm-0, ref_f1: +0.000 [+0.000, +0.000] → samp-p0.9 tie
- samp-p0.9 − prlm-0, rep3: +0.000 [+0.000, +0.000] → samp-p0.9 tie
- samp-p0.9 − prlm-0, distinct2: +0.000 [+0.000, +0.000] → samp-p0.9 tie

prlm-0 equals greedy-f32 on 0/0 prompts; 0/0 steps certified (rest accepted ties). Do-nothing control equals prlm-0 on 0/0.

## T2 closed cloze (40 items)

| system | accuracy | per-seed range |
| --- | --- | --- |

greedy-f32 − samp-p0.9 accuracy: +0.000 [+0.000, +0.000] → tie

## T3 control (20 prompts per kind, 24 tokens)

| kind | system | success | held-out hit (topic) | NLL/token under base |
| --- | --- | --- | --- | --- |
| ban | samp-p0.9 | 0.13 |  | 5.87 |
| ban | greedy-f32 | 0.10 |  | 2.34 |
| ban | prlm-0 | 0.10 |  | 2.34 |
| ban | ctl-null | 0.10 |  | 2.34 |
| ban | prlm-theta:ban (CORRECTED: sign bug, boosted banned words) | 0.10 |  | 2.13 |
| ban | prlm-theta:ban-fixed | 1.00 |  | 2.45 |
| ban | prlm-theta:ban-soft (CORRECTED: sign bug, boosted banned words) | 0.10 |  | 2.13 |
| ban | prlm-theta:ban-soft-fixed | 1.00 |  | 2.45 |
| ban | samp-p0.9 best-of-5 with checker | 0.45 |  | mean tries 3.90 (×24 calls) |
| topic | samp-p0.9 | 0.00 | 0.00 | 5.87 |
| topic | greedy-f32 | 0.00 | 0.00 | 2.34 |
| topic | prlm-0 | 0.00 | 0.00 | 2.34 |
| topic | ctl-null | 0.00 | 0.00 | 2.34 |
| topic | prlm-theta:embdir | 1.00 | 0.00 | 4.71 |
| topic | prlm-theta:embdir-rand | 0.00 | 0.00 | 2.47 |
| topic | prlm-theta:topic | 0.25 | 0.00 | 2.60 |
| topic | GAMED-oracle | 0.00 | 0.40 | 2.89 |
| topic | samp-p0.9 best-of-5 with checker | 0.00 |  | mean tries 5.00 (×24 calls) |
| end | samp-p0.9 | 0.56 |  | 5.87 |
| end | greedy-f32 | 0.50 |  | 2.34 |
| end | prlm-0 | 0.50 |  | 2.34 |
| end | ctl-null | 0.50 |  | 2.34 |
| end | prlm-theta:end | 1.00 |  | 2.84 |
| end | samp-p0.9 best-of-5 with checker | 1.00 |  | mean tries 2.15 (×24 calls) |

## Reproducibility (T1)

- prlm-0: mean distinct outputs per prompt nan, max 0 (runs 1 to 3 from an empty cache, run 4 at 1 thread)
- greedy-f32: mean distinct outputs per prompt nan, max 0 (runs 1 to 3 from an empty cache)
- samp-p0.9: mean distinct outputs per prompt nan, max 0 (seeds 0 to 4 (unseeded serving))
- samp-p0.9-seed0-rerun: mean distinct outputs per prompt nan, max 0 (seed 0 rerun three times)
- seeded sampling seed 0 equals its rerun on 0/0 prompts

## Alternative constructions

| system | items | certified | mean calls | mean NLL of output (beam/best-first) | equals greedy |
| --- | --- | --- | --- | --- | --- |
| alt-greedy-short | 0 | n/a | 0.0 | | |

Best-first certified a sequence different from greedy on 0 items: each is, by its certificate, the most likely sequence of its length, so greedy's is less likely there (). Beam-4's NLL column is its own summed score; beam certified only 0 items because its bound (the best pruned prefix) is loose.
## Weight sweeps on T3 topic prompts (dose response, reweighting cost)

### topic

| weight (logits) | hit TRAIN words | hit HELD-OUT words | NLL/token |
| --- | --- | --- | --- |
### embdir

| weight (logits) | hit TRAIN words | hit HELD-OUT words | NLL/token |
| --- | --- | --- | --- |
## Lean replay of recorded steps


## Pareto front over T1 systems (minimize; Lean `prlm front`)

Objectives: (24 − ref_match)×100, rep3×1000, distinct outputs per prompt ×100, ms/token ×10, calls per token ×100.


With latency as 10% buckets (the contract's tie rule):


Front members and the rows that minimize Chebyshev weights solved from each:

