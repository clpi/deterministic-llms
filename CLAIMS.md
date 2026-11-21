# PRLM claims ledger

2026-11-20. One 4-vCPU container, GPT-2 small (124M) LM-head ONNX on
onnxruntime 1.29.0 CPU, Lean 4.19.0. Every number below is rebuilt from
`out/log.jsonl` by `harness/report.py` (output: `out/REPORT.md`). Verdicts:
VERIFIED (run or proved here), CORRECTED, REFUTED, UNMEASURED (with the
reason). Small samples: 35 continuation prompts, 40 cloze items, 20 prompts
per control kind. Treat every quality number as a small-sample estimate;
intervals are paired bootstrap over prompts.

## Proofs (lean/check.sh: cold build, warnings as errors, no sorry, no custom axiom, no escape hatch)

| # | Claim | Verdict | Receipt |
| --- | --- | --- | --- |
| P1 | Determinism under any candidate order (`best_perm`, `gen_perm`, `max_unique`) | VERIFIED | Prlm/Core.lean |
| P2 | Certification soundness: a certified pick beats every other token by at least the margin (`certify_sound`); completeness (`certify_complete`) | VERIFIED | Prlm/Core.lean |
| P3 | Reweighting equals recompute, per step and per generation (`rescore_eq`, `stepRe_eq`, `gen_reweight`) | VERIFIED | Prlm/Core.lean |
| P4 | Cone: inside the margin box the pick is unchanged without recompute (`stable_of_bound`) | VERIFIED | Prlm/Core.lean |
| P5 | Parameter locality: an axis zero on the candidates changes nothing, per step and per generation (`scores_local`, `dot_set_zero`, `gen_local`) | VERIFIED | Prlm/Core.lean |
| P6 | A correct cache is invisible (`gen_cache`); outputs extend content (`gen_prefix`) | VERIFIED | Prlm/Core.lean |
| P7 | Reused from srce PR 7 unchanged: archive = Pareto front, order independent; front-solved Chebyshev weights pick their target | VERIFIED (rebuilt) | Prlm/AlgoSearch.lean |
| P8 | The proofs are about the functions that decided the runs | VERIFIED | `prlm decide` replayed 2,280 recorded GPT-2 steps (all 50,257 integer scores each): 2,280/2,280 picks equal, 2,280/2,280 rescored = recomputed |
| P9 | The float forward pass is deterministic | not provable here; measured as E3 | |

Limits of the proofs: P4 is a sufficient test and is weak in practice
(E8). Nothing proved says an axis means what its name says, or that
outputs are good.

## Empirical (contract tasks/CONTRACT.json, frozen before any axis code)

| # | Claim | Verdict | Numbers |
| --- | --- | --- | --- |
| E1 | PRLM with weights 0 is greedy decoding, certified | VERIFIED | equal outputs 35/35 T1 prompts; 840/840 steps certified (margin > 0, no ties needed) |
| E2 | Deterministic decoding beats sampling on closed-answer tasks | VERIFIED (small sample) | T2 accuracy greedy/prlm-0 0.450 vs samp-p0.9 0.265, samp-t0.7 0.345, samp-t1.0 0.230; greedy − samp-p0.9 +0.185 [+0.085, +0.300] |
| E3 | Reproducible across runs and thread counts | VERIFIED | prlm-0 1 distinct output per prompt over 5 runs (3 from empty cache, 1 at 1 thread); logits bitwise equal at 1 and 4 threads on 35/35 prompts |
| E3b | Reproducibility is an advantage over sampling | CORRECTED | only over UNSEEDED sampling (5 distinct outputs of 5). Seeded sampling reproduced 35/35 too. The deterministic gain is that the output is a function of content and named weights, not of a seed. |
| E4 | Deterministic decoding loses on repetition | VERIFIED (loss) | T1 rep3 greedy 0.147 vs samplers 0.009 to 0.075, distinct2 0.802 vs 0.885 to 0.975; every sampler's rep3 is lower than prlm-0's (intervals exclude 0) |
| E5 | Deterministic decoding wins on reference fidelity | VERIFIED vs samp-t1.0 and samp-p0.9 (ref_match −0.37 and −0.34 for them, ref_f1 −0.047, −0.042), tie vs samp-t0.7 | small effects: 1.11 vs 0.75 to 0.95 tokens of 24 |
| E6 | Named axes control the output deterministically | VERIFIED for end (1.00 vs 0.85 greedy, 0.52 sampled, NLL 1.13 vs 1.07) and ban (fixed sign: hard 1.00, soft 0.95 vs 0.10 greedy; best-of-5 sampling with a checker 0.45 after 4.0 tries) | |
| E6b | Declared ban weights | CORRECTED (my bug) | feature −1 times declared −4000/−8 boosted banned words; records kept, labelled; correction record `correction:ban-sign`, rerun under `ban-fixed` and `ban-soft-fixed` |
| E6c | Topic axis at its declared weight (4 logits) controls topic | REFUTED at that weight | 0/20 hits, output equal to greedy; it takes 8 logits for 0.55, at NLL 1.58 vs 1.07 (sweep) |
| E7 | Reweighting reuses work | VERIFIED | sweeps: topic 694 vs 3,360 model calls (79% saved), embdir 1,905 vs 2,400 (21% saved); identical outputs 240/240 (empirical gen_reweight and gen_cache) |
| E8 | The cone test avoids recompute often | REFUTED as stated | the proved box decided 17/840 (T1) and 45/1,440 (T3) steps without recompute, all correct; the weight change moved the pick on 406/840 and 188/1,440 steps |
| E9 | Do-nothing control changes nothing | VERIFIED | ctl-null equals prlm-0 on 35/35 T1, 40/40 T2, 60/60 T3 (empirical gen_local) |
| E10 | Gamed controls | labelled GAMED, never counted | oracle reaches 11.29/24 ref_match, 0.90 cloze, 0.20 held-out topic words: what reading the answer looks like |
| E11 | Latency | tie | 43.1 to 43.2 ms/token for greedy and PRLM vs 43.9 to 50.1 for samplers (nucleus sorting costs ~15%); same model calls (24 per output) |
| E12 | Fluency cost of control (NLL under the unmodified model) | VERIFIED but biased | NLL favors greedy by construction (it picks the likeliest token), so it shows cost, not quality |

## Alternative constructions

| # | Construction | Verdict | Numbers |
| --- | --- | --- | --- |
| A1 | Best-first search with certificate (exact most likely sequence) | VERIFIED where certified | 45/75 certified within 200 calls; mean 107.7 calls vs 5.9 for greedy; on 24 items it certified a sequence other than greedy's, so greedy's was not the most likely there. Cloze accuracy fell to 0.35 from 0.45: most likely is not most correct. |
| A2 | Beam-4 with certificate | weak | certified 12/75 (the pruned-prefix bound is loose), 20.5 calls, cloze 0.35 |
| A3 | Trigram automaton (no network) | loses on quality, wins on cost | ref_match 0.49/24, cloze 0.025, 0.004 ms/token, 0 model calls; trained on 15,685 tokens only |
| A4 | Rules distilled from greedy | no gain | 495 rules fired on 4/840 steps; 830/840 tokens equal greedy; its latency edge is inside the 10% tie rule |
| A5 | Hard-filter axis as constrained decoding | VERIFIED | E6 ban-fixed |
| A6 | On-the-fly weighting (weights as a function of content) | VERIFIED on TEST, post hoc | written after 7/35 T1 prompts were seen; compared on DEV (TRAIN documents); no selection rule was declared in advance, so all four are reported on TEST. DEV front: noloop4 and rep1.5+noloop4. noloop4: TEST ref_match −0.11 [−0.29, +0.03] (tie), rep3 0.008 vs 0.147. Gated rep+noloop (on only after a repeated 3-gram): ref_match tie (+0.000), rep3 0.029. Fixed rep1.5+noloop4 loses ref_f1 (−0.051 [−0.105, −0.006]). First run crashed (my recursion bug); its records stay, superseded. |

Pareto front over T1 systems with latency in 10% buckets (Lean `prlm front`):
samp-t0.7; {greedy, prlm-0, rules} as one tied point; prlm-theta; trigram.
Each is the unique Chebyshev minimizer under weights solved from its own
point (the tied three share theirs). No single system dominates: PRLM
with weights 0 owns reference fidelity and reproducibility, PRLM with
anti-repetition weights owns repetition, sampling at 0.7 trades both,
and the automaton owns cost.

## Hypotheses

| # | Hypothesis | Verdict | Test |
| --- | --- | --- | --- |
| H-D | Decompiling weights into named axes turns a model into a PRLM | REFUTED for embedding directions at the declared weight; broader claim UNMEASURED | embdir axis from the TRAIN half of each lexicon: TRAIN-word hits 0.75 at 0.02 (1.00 at 0.04) but HELD-OUT hits 0.00, equal to the random-direction control (0.00); one held-out hit in 20 at 0.01. The direction encodes the listed words, not the topic. Attention and MLP weights, whole-model rewriting, and "model as executable compressed bytes" were not attempted. |
| H1 (project) | Intelligence handled incrementally | supported narrowly | E7: reweighting reuses cached content nodes and gives identical outputs; nothing here shows compounding |

## UNMEASURED

- Any model other than GPT-2 small (no larger model could be fetched; huggingface.co is blocked by the proxy).
- Quality judged by people; long outputs (only 24 tokens); tasks beyond these three.
- Determinism across machines, CPU types, or onnxruntime versions (one container only).
- Native or Idol implementations (Python harness, Lean replay only).
- Exact cone region per step (only the sufficient box was measured).
- Weight decompilation beyond the token embedding (H-D).
