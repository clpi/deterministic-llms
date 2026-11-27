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

## Forward skipping by continuation equivalences (2026-11-21)

Ask: Chris 2026-11-21T00:07Z. Contract tasks/SKIP_CONTRACT.json, logged as
record 3391 before harness/skip.py existed. Model: GPT-2 small (the same
pinned file). Chris's message cites a Qwen run that certified shared
reply-bound accounting across histories; no such run is in this
project's ledger or files (searched for "qwen" across everything), so
nothing here builds on it and no Qwen result is claimed. 55 histories (20
DEV, then 35 TEST), replies end at the first newline or period token or
at 24 tokens; 479 reply tokens in the clean replay.

| # | Claim | Verdict | Numbers |
| --- | --- | --- | --- |
| S1 | Verified skipping is exact | VERIFIED | k = 2, 4, 8: 0/55 reply mismatches, 0/55 stop-certificate mismatches (step, token, integer margin, certified, reason), 0 pick or margin mismatches at any step |
| S2 | Its premise holds here: one forward over content plus draft gives the same logits as per-step forwards | VERIFIED on this run | 479/479 positions bitwise equal in every verified configuration (onnxruntime 1.29.0 CPU, 4 threads) |
| S3 | Skipping saves model calls on real histories | VERIFIED but small | calls vs clean 479: k2 418 (13% saved), k4 438 (9%), k8 459 (4%). Gamed ceiling (store seeded with the clean replay's own continuation): 81 to 83 (83%) |
| S4 | The work is shared across histories | REFUTED on this data | accepted drafted tokens from the same history vs an earlier one: k2 64 vs 3, k4 45 vs 0, k8 23 vs 0. Almost all reuse is a history repeating itself (greedy loops), not one history reusing another's continuation |
| S5 | Unverified skipping (trust the class, no model call) is exact | REFUTED | reply mismatches k2 20/55, k4 3/55, k8 1/55. It also cannot produce a stop certificate at all: the margin needs the logits it skipped (stop certificates differ 55/55 even when seeded with the right answer) |
| S6 | Wall time | measured once, small | verified k2 15.2 s vs 17.2 s for the empty-store control (same machine, one run each); k8 19.7 s, slower than control despite fewer calls (longer verify sequences, rejected drafts). This ONNX model has no KV cache, so every call recomputes the prefix; the numbers do not carry over to a KV-cached server |
| S7 | Draft acceptance is exactly the rule's generation | VERIFIED (proved) | Prlm/Skip.lean `accept_gen`; `accept_batch` makes S2 an explicit PREMISE rather than an assumption hidden in the proof |
| S8 | Do-nothing control | VERIFIED | empty store: 479 calls, identical to clean replay |

Gate correction: lean/check.sh built only the executable's imports and
did not fail when the axiom check itself errored. Fixed (builds the
library and the executable, exits 1 on any axiom-check error); a
tamper control with an inserted sorry now fails the gate (exit 1). The
first-pass gate result stands because every module was an import of the
executable then.

UNMEASURED: Qwen or any other model; equivalence classes other than the
last k tokens (for example classes keyed on hidden state); probability
mass pooled across histories (each witness's mass is recorded, nothing
pools it); a KV-cached runtime.

## End-to-end improvements (2026-11-21, after PR 1 merged)

Ask: Chris 2026-11-21T01:17Z, "Improve end to end". Targets were the
three weakest measured points: the box test decided 73 of 2,280 steps
without recompute (E8), verified skipping saved 4% to 13% of calls with
almost no cross-history reuse (S3, S4), and the repository had no CI.
Contract tasks/IMPROVE_CONTRACT.json, logged as record 4179 before
harness/reweight.py and harness/skip2.py existed. lean/Prlm/Sparse.lean
was written before that record; the record says so.

### Exact sparse reweighting

New rule (`Prlm/Sparse.lean`, `best_split`, proved): after a weight
change, the new pick is the best of the candidates whose score moved plus
the single best unmoved candidate, which is the first unmoved one in the
old order. No margin and no box: it is exact whenever the moved set is
known. The Lean executable checks it against a full recompute at every
recorded step (log records 4180 to 4183).

| # | Claim | Verdict | Numbers |
| --- | --- | --- | --- |
| R1 | The sparse rule equals a full recompute on real steps | VERIFIED | 2,280/2,280 steps (T1 840, T3 topic 480, end 480, ban 480), Lean exe exit 0 on every file |
| R2 | It decides every step without rescoring the vocabulary | VERIFIED | mean work per step (moved candidates + rank of first unmoved + 1) vs 50,257 for a rescore: T1 33.8 (max 53), topic 13.0 (max 13), end 179.1 (max 182), ban 47.3 (max 51). The old box test decided 73/2,280 on the same steps (17, 28, 28, 0) |
| R3 | The steps tested include ones where the pick changes | VERIFIED | picks changed by the weight change: T1 406/840, end 87/480, ban 88/480, topic 0/480. Topic 4 moved no pick on these prompts, so the topic row only shows the rule agrees when nothing changes |
| R4 | Cost scales with the moved set, not the vocabulary | VERIFIED here, with a limit | the `end` axis colors about 178 tokens per step, so it costs about 179; an axis that colors most of the vocabulary would cost about a rescore. The rule needs the old full order (one sort per step, cached in the replay), which is not counted in the work above |

### Drafting from prompt and history n-grams

Same 55 histories, reply rule and clean replay as S1 to S8; verified mode
only, so every drafted token is checked by the model and the output is
the rule's own generation (`accept_gen`). Index: every suffix of length
1 to 8 points to its most recent occurrence; the draft is the next 8
tokens after the longest match. Records 4184 to 4408.

| # | Claim | Verdict | Numbers |
| --- | --- | --- | --- |
| D1 | Exact | VERIFIED | every source: 0/55 reply mismatches, 0/55 stop-certificate mismatches |
| D2 | Fewer model calls than the first pass | VERIFIED | vs clean 479 (first-pass best was 418): gen 400 (16% saved), ctx 329 (31%), all 324 (32%), all+corpus 310 (35%). Gamed ceiling from S3 is unchanged at 81 to 83 |
| D3 | Work shared across histories | CORRECTED (S4 was REFUTED) | accepted drafted tokens from outside the current history: all 16 of 170, all+corpus 39 of 184 (29 from TRAIN-corpus documents, 10 from earlier histories). It is real but a minority; most of the gain is the history copying its own prompt (81 to 92 tokens) |
| D4 | Wall time | measured once, small | gen 16.3 s, ctx 16.7 s, all 13.9 s, all+corpus 13.2 s vs 17.2 s for the empty-store control; same machine, one run each, no KV cache, so these do not carry over to a KV-cached server |
| D5 | Adding sources is not free | REPORTED | with the corpus added, same-prompt acceptances fell from 91 to 81 and earlier-history ones from 16 to 10, because a longer match from another source now wins the lookup. Total calls still fell |

### CI

`.github/workflows/check.yml`: installs Lean 4.19.0 from the GitHub
release, runs `lean/check.sh` cold, replays `tests/fixture.bin` (the
first 3 recorded GPT-2 steps of the T1 replay, 50,257 scores each)
through the proved functions (must exit 0), requires
`tests/fixture-tampered.bin` (one harness pick changed by 1) to exit 1,
and verifies the log hash chain. Gate record 4409 (`lean-gate:3`) logs
the local cold run with `best_split`. Whether the workflow passes on
GitHub is reported on the PR, not claimed here.

UNMEASURED: a KV-cached runtime; any model other than GPT-2 small;
drafting from equivalence classes other than literal suffixes; the cost
of keeping the old order when no replay is cached.

## More models and architecture as a parameter (2026-11-21)

Asks: Chris 02:01Z ("test a variety of models and architectures"), 02:12Z
(cover other model types generally and parametrically), 02:24Z and 02:25Z
(density must be solvable on the Pareto front, and is the computed
consequence of what is being reweighted). Contracts, each written before
its code: tasks/MULTI_CONTRACT.json (record 4410),
tasks/ARCH_CONTRACT.json (4418), tasks/DENSITY_CONTRACT.json (4419); the
last two were logged after multi.py stopped writing, and their sha256s at
write time are in the project ledger. Full tables: out/REPORT-multi.md
(harness/multireport.py rebuilds it from the log). This section measures
two items the previous section listed as UNMEASURED: a KV-cached runtime
and models other than GPT-2 small.

What could be reached: huggingface.co, GitHub release pages, Ollama,
Kaggle and ModelScope are not reachable from this session, and no mirror
was used. The onnx/models zoo files and Google's t5-data bucket were.

### Six models (records 4411 to 4417)

gpt2-fp32-ort (baseline), gpt2-int8-ort (the same file through
onnxruntime's dynamic int8 quantization), gpt2-numpy-kv (the same weights
in a numpy forward with a KV cache), t5-base-ort (encoder-decoder, own
vocabulary of 32,128), llama-style-random and mamba-style-random (random
weights, mechanics only).

| # | Claim | Verdict | Numbers |
| --- | --- | --- | --- |
| M1 | The proved rule decides exactly on every model's recorded scores | VERIFIED | Lean replay agree = sparseEq = steps on all 6 (840 each; T5 206, it stops at its end token); a tampered first record exits 1 on all 6 |
| M2 | Same content gives the same logits on repeat | VERIFIED, one exception | bitwise on all 6 in process and in fresh processes, except llama-style-random in a fresh process at 1 BLAS thread: 0/192 rows bitwise, 192/192 picks equal |
| M3 | One call over a longer sequence gives the same logits as per-step calls (the premise of verified drafting) | holds only for 2 of 6 bitwise | bitwise: gpt2-fp32-ort 840/840, mamba-style-random 840/840, T5 171/206, gpt2-numpy-kv and llama 70/840. Picks: equal everywhere except int8, 633/840 |
| M4 | A KV cache gives the same logits as recomputing | REFUTED bitwise, VERIFIED for picks | 35/840 rows bitwise (the first row of each history) on all three KV models; 840/840 picks equal |
| M5 | Verified drafting is exact on every model | REFUTED for int8 | replies equal on 5 of 6 models. gpt2-int8-ort: 28/35 replies differ. Its dynamic quantization takes the scale from the whole call, so a position's logits depend on later tokens in the same call. gpt2-numpy-kv: replies equal, certificate margins differ in 2/35 histories |
| M6 | Calls saved by drafting | measured | fp32 and numpy 840 to 577 calls; T5 and the random models save nothing |
| M7 | Quantization keeps the model | REFUTED for this quantization | int8 vs fp32 teacher-forced: 377/840 picks equal, 0/35 sequences equal; greedy text collapses (' the the the'). The axes still steer it (T3 on int8, out/REPORT-int8-t3.md: ban fixed 1.00, end 1.00, topic 0.25; its own end baseline 0.50 vs fp32 0.85) |
| M8 | Another runtime for the same weights keeps the picks | VERIFIED | gpt2-numpy-kv vs fp32: 840/840 picks, 35/35 sequences, but 0/840 integer-score rows equal |

### Architecture as a parameter (records 4420 to 4450)

harness/arch.py: one implementation over an array module, JAX for
training and numpy float32 for every measurement. The config picks the
layer kinds, positions, KV heads, feed-forward and norm. Ten configs: A1
learned-position MHA GELU LN (GPT-2-like), A2 RoPE GQA SwiGLU RMS
(Llama-like), A3 RoPE MQA, A4 ALiBi (BLOOM-like), A5 RoPE GQA MoE 4
experts top-2 (Mixtral-like), S1 selective SSM (Mamba-like), R1 RWKV-4-like,
R2 GRU, R3 LSTM, H1 attention/SSM hybrid (Jamba-like). Each config ran with
random weights and with weights trained for 600 Adam steps on the
15,685-token TRAIN corpus: 20 models, about 1.1 to 2.7M parameters each.

| # | Claim | Verdict | Numbers |
| --- | --- | --- | --- |
| X1 | Repeatable in process and across BLAS thread counts | VERIFIED | 192/192 rows bitwise for all 20 models, in process and in fresh processes at 1 and 4 threads |
| X2 | The proved rule decides exactly | VERIFIED | Lean agree = sparseEq = 840/840 for all 20; tamper exits 1 for all 20 |
| X3 | Full call vs per step | depends on family | bitwise 840/840 for every recurrent family (S1, R1, R2, R3), random and trained; about 70/840 for attention families and the hybrid. Picks 840/840 everywhere |
| X4 | Incremental path (KV cache or recurrent state) vs per step | REFUTED bitwise, VERIFIED for picks | 0/840 rows bitwise for every one of the 20 models; 840/840 picks |
| X5 | Verified drafting is exact on every family | VERIFIED for replies | 0 reply mismatches on all 20; one certificate-margin mismatch (A1 trained). Calls: random 456 to 574 of 840 (they loop), trained 788 to 827 |
| X6 | Tiny trained models beat the baselines on T1 | REFUTED | ref_f1 0.098 to 0.127 vs the trigram automaton 0.133 trained on the same corpus; best ref_match 0.49 (H1), equal to the trigram; GPT-2 small 1.11 and 0.202. Quality here is about models trained on 15,685 tokens, not the families at scale |

So the decision rule, sparse reweighting and Lean replay are exact on
every architecture tried, because they act on the recorded integer
scores. What depends on the architecture and the runtime is whether those
scores are the same along different computation paths. In float32 they
are not bitwise the same, but every pick matched. Under dynamic int8
quantization they differ enough to change picks, and verified drafting
breaks. A PRLM therefore has to fix one computation path as the definition
of base(c, v). This repo uses a full per-step recompute. Any other path
has to be checked against it, as done here.

### Density as the computed consequence of the weight change (records 4451 to 4454)

Density is not a setting. For a weight change, the moved set at a step
is every candidate whose score the change alters. Density is its share of
the vocabulary, and the exact work of the step is moved + rank of the
first unmoved candidate + 1 (best_split). Steps: the 60 T3 prompts × 24
along GPT-2's prlm-0 trajectory, with all five vocabulary axes recorded.
Search space: rep, noloop, ban, topic and end each at 0, 2 or 8 logits
(243 weight changes). Six objectives, all minimized: work, ban hits,
held-out topic misses, end misses, loops and drift from the base pick.

| # | Claim | Verdict | Numbers |
| --- | --- | --- | --- |
| P1 | Every (change, step) decision is solvable exactly from the moved set | VERIFIED | 349,920 decisions, sparse rule vs dense recompute: 0 disagreements |
| P2 | Density is small and computed, not chosen | measured | mean density 0 to 0.21% of the vocabulary; work 1 to 107 per step vs 50,257 for a dense rescore |
| P3 | The front is solved incrementally and deterministically | VERIFIED | Lean `prlm front` recomputed after each of 243 appends, exit 0 every time; every new front lay inside the previous front plus the new point; final front 62 of 243 points, each the Chebyshev minimizer for weights aimed at it (3 pairs tie) |
| P4 | Front points replay exactly in Lean | VERIFIED | all 62 front points: agree = agreeRe = reEqRec = sparseEq = 1,440/1,440, exit 0; Lean's work count equals the harness's for every point (record 4454) |
| P5 | Topic weights reach held-out topic words | REFUTED again | 480/480 topic steps miss at every topic weight up to 8; the front's topic points come from side effects on loops and drift |

These objectives are teacher-forced: each one is the rule's decision at
a step of the fixed prlm-0 trajectory, not a regenerated output.
Regenerating with the chosen points is a follow-up.

UNMEASURED: any pretrained Llama, Qwen, Mistral, Mixtral, Mamba, RWKV or
other model above 220M parameters; GPU kernels; a KV-cached ONNX export;
batched serving with requests sharing one call; quantization other than
onnxruntime dynamic int8.

## Self-improving, scope-solving Pareto search (2026-11-21)

Ask: Chris 05:00Z, "improve and prove and make self improving self scope
solving and algorithm parametrically optimizing at Pareto optimum".
Contract tasks/LOOP_CONTRACT.json (record 4456), logged before
harness/loop.py existed. Proofs: lean/Prlm/Loop.lean (cold gate passes,
record 4467). Run records 4457 to 4466.

The loop searches weight levels for rep, noloop, ban, topic and end, in
quarter logits from 0 to 16, against the six density objectives. It
selects on DEV prompts only (ten per T3 kind, 720 steps) and is scored
on TEST prompts (the other ten per kind). Each round runs four parts.
Scope removes axes that cannot matter and freezes axes that did not
matter. Proposals are refined around the front (midpoints toward
neighbouring levels, or doubling at the edge). Admission uses the
archive rule. Lean recomputes the front, which must equal the archive.

| # | Claim | Verdict | Numbers |
| --- | --- | --- | --- |
| L1 | One append changes the front only by the new point | PROVED | `arch_cons_sub` |
| L2 | Appending any batch of candidates never loses ground: every old front point stays on the front or is dominated by a new one | PROVED | `arch_covers` (this is what lets the loop run unattended: a round can only keep or improve the front it inherits) |
| L3 | Admission and rejection need only the archive, not every past candidate | PROVED | `admit_sound`, `reject_sound` |
| L4 | An axis that is zero on every candidate of every step changes no decision and so no objective, at any weight | PROVED | `scope_sound`; all five axes were structurally in scope on DEV, so it removed none here |
| L5 | Measured scope (freeze an axis whose weight never changed a DEV decision) | measured, not proved | no axis was frozen in any round |
| L6 | The loop's archive equals Lean's front every round | VERIFIED | 4/4 rounds (loop) and 4/4 (GAMED control); Lean exit 0 |
| L7 | Every decision exact | VERIFIED | sparse rule vs dense recompute: 0 disagreements over 600 points x 720 DEV steps |
| L8 | The loop improves on its own starting front on held-out TEST prompts | VERIFIED | final front (260 points, after 3 rounds and 600 evaluations) strictly dominates 50% of the initial front's points on TEST and weakly covers 95%; the initial front strictly dominates 1 of 260 final points (0.4%), so 259/260 show no regression |
| L9 | It beats random search at the same budget | VERIFIED | 600 random DEV evaluations give a 36-point front; on TEST the loop's front strictly dominates 67% of it, random dominates 1.9% of the loop's |
| L10 | Selecting on DEV leaves little on the table | measured | GAMED control (the same loop selecting on TEST) strictly dominates 5.4% of the loop's TEST front; the loop dominates 1.2% of it |
| L11 | It reaches a fixed point | NO | it stopped at the 600-evaluation budget with the front still growing (20, 54, 111, 260); with six integer objectives the front is large |
| L12 | The final front replays exactly in Lean | VERIFIED | all 260 final front points: agree = agreeRe = reEqRec = sparseEq = 1,440/1,440, exit 0 (record 4466) |

The improvement is to the trade-offs between objectives, measured
teacher-forced at each step of fixed trajectories. It is not a gain in
text quality, and the topic axis still misses every held-out topic word.

UNMEASURED: a loop that also proposes new axes or new objectives (scope
here covers removing and freezing, not inventing); regenerated outputs
under the chosen points; the same loop on the 20 architecture models.

## Stopping at a certified fixed point (2026-11-21)

Contract tasks/FIXPOINT_CONTRACT.json (record 4468) and amendments
tasks/FIXPOINT_AMEND_1.json (record 4491), tasks/FIXPOINT_AMEND_2.json
(record 4705) and tasks/FIXPOINT_AMEND_3.json (record 4976), each logged
before its code;
proofs in lean/Prlm/Fix.lean (cold gate record 4671). The same DEV/TEST
steps and objectives as the loop above.

| # | Claim | Verdict | Numbers |
| --- | --- | --- | --- |
| F1 | A round that proposes nothing is exactly the closed condition, and then no unit neighbour of a front point dominates it | PROVED | `closed_iff_no_proposals`, `closed_local_opt`, `point_local_opt`, `fix_stable` |
| F2 | A fixed point at exact resolution is within reach | CORRECTED | with no budget the front doubled every round: 20, 54, 111, 277, 671, 1,615, 3,465, 6,768, 12,618 after rounds 0-8 (39,106 evaluations, records 4471-4489). Stopped by hand before the 200,000 cap (record 4490); Lean's front equalled the archive every round | Searching at exact resolution from the start: REFUTED as above. Climbing the resolution ladder (F4) reached it: level 10 reads every objective exactly and closed at 38,458 cells after 142,391 evaluations. The ladder differs in coarse-to-fine order and in keeping one point per cell; which of the two made the difference is UNMEASURED |
| F3 | At resolution r, anything that exactly dominates a front point does so by less than r_i in every objective | PROVED | `coarse_le`, `coarse_front_within` |
| F4 | The loop reaches certified fixed points one resolution at a time | VERIFIED to level 10 (exact) | levels 0-10 each ended with nothing left to evaluate. Cells on the front (evaluations so far): level 0: 7 (164), 1: 17 (288), 2: 101 (2,795), 3: 487 (11,797), 4: 2,299 (48,104), 5: 6,858 (82,519), 6: 13,744 (103,875), 7: 24,077 (122,845), 8: 33,339 (136,068), 9: 37,491 (141,009), 10: 38,458 (142,391), under the 200,000 cap. Lean's front equalled the archive's cells in every round; 0 sparse-vs-dense disagreements (rerun records 4706-4972) |
| F5 | A rerun from scratch repeats the first run | VERIFIED | a machine restart killed the first run in level 7. Under tasks/FIXPOINT_AMEND_2.json (record 4705) the ladder checkpoints every round and was rerun from scratch: all 194 rounds that both runs reached match exactly. The rerun itself survived further restarts by resuming from its checkpoints (records 4956, 4975, 4977) |
| F6 | On TEST the finest front improves on round 0 and on PR 4's loop | VERIFIED | share of the other front strictly dominated, weakly covered: fix covers round 0 (0.5, 0.9), round 0 covers fix (0.0, 0.001); fix covers PR 4's final front (0.1115, 0.9538), PR 4 covers fix (0.0034, 0.0144). All 38,458 fix points are undominated by round 0 and 38,329 by PR 4 (record 4972). The gap between strict and weak shares is wide: against PR 4, most of the cover is by points that tie in some objectives |
| F7 | Random search and do-nothing controls | VERIFIED | random search with the same 142,391 evaluations reached a DEV front of 1,292; fix covers random (0.8142, 0.8336), random covers fix (0.0144, 0.0183) (record 4973). Doing nothing leaves the initial front unchanged (record 4974) |
| F8 | GAMED control (the same ladder tuned on TEST) | TRUNCATED, lower bound | its level 9 ran 46 rounds without settling, admitting 14-15 points every round, with the cap about 1,200 rounds away. Under tasks/FIXPOINT_AMEND_3.json (record 4976, logged before the code change) it was stopped between rounds and scored at its finest certified level, 8: TEST front 25,904; gamed covers fix (0.1195, 0.8186), fix covers gamed (0.0004, 0.71) (record 4978). Tuning on TEST strictly beats 12% of fix; no claim that fix matches GAMED is drawn. Gamed level 9's fixed point is UNMEASURED; its state is checkpointed, so it can be resumed |
| F9 | Lean replays the finest certified front of at most 600 points exactly | VERIFIED | level 3, all 487 front points: agree = agreeRe = reEqRec = sparseEq = 1,440/1,440, exit 0 (record 4979). Finer fronts (2,299 points and up) exceed the 600-point replay limit and are not replayed point by point |
| F10 | Lean replays a random sample of the exact-resolution (level 10) front | VERIFIED | 600 of the 38,458 front points, drawn with default_rng(4972) before any replay (FIXPOINT_AMEND_4, records 4981-4982): agree = agreeRe = reEqRec = sparseEq = 1,440/1,440 and exit 0 on all 600 (864,000 steps, record 4986). A sample: with no failure in 600, the failing share of the whole front is below 0.5% at 95% confidence, not proved zero |

One bug was found and fixed during the run: the first ladder run kept every
point of an occupied cell, so phase A could not stop. It was stopped after
three rounds and restarted with one representative per cell (record 4497).

## Output structure as a parametric decision (2026-11-21)

Contract tasks/STRUCT_CONTRACT.json (record 4469), logged before
harness/structure.py existed; proofs in lean/Prlm/Struct.lean. GPT-2
small picks a reply's structure (essay, song, steps, demo, table) by the
certified rule over five candidate tokens, plus a cue-word axis and
per-structure biases. Requests, topics and users are synthetic and
labelled by construction; DEV and TEST share no topic and no template.

| # | Claim | Verdict | Numbers |
| --- | --- | --- | --- |
| S1 | A bias shift makes structure p the pick exactly when p beats every rival outright | PROVED | `picked_iff`, `wins_iff` |
| S2 | The revealed-preference update is minimal: the shift `dstar` wins every observed event and no smaller non-negative shift does | PROVED, and VERIFIED on DEV | `minshift_wins`, `minshift_minimal`, `minshiftAll_*`; at Delta every user wins 80/80 DEV events, at Delta-1 79/80 (all three users) |
| S3 | Weight updates compose in any order; replacing one segment's inputs changes only that segment's decision | PROVED | `rescore_comm`, `replan_local` |
| S4 | Rendering is reversible and re-rendering one segment rewrites only its slice | PROVED, and VERIFIED | `parse_render`, `render_set`; 40/40 GPT-2 replies parse back to the same structure and segments at the token level in both modes |
| S5 | The model's own scores pick the named structure | WEAK | model alone: 42.3% on TEST explicit requests (40.3% DEV); with the cue-word axis fitted on DEV (8.25 logits) 81.3% on TEST, 100% on DEV. The misses are the templates with no cue word ("a long-form piece on X" 0/20, "a playground for X" 0/20, "X side by side" 6/20) |
| S6 | Revealed preference generalizes to new ambiguous requests | VERIFIED | TEST ambiguous requests answered in the user's structure: essay user 100%, demo user 97.5%, song user 95% (do-nothing: 0%, 0%, 1.25%; random 19-25%) |
| S7 | The update is neither over- nor under-tuned | REFUTED for the contracted global bias | TEST explicit accuracy fell from 81.3% to 73.3% / 68.7% / 68.0%; a 16-logit over-shift drops it to 20% |
| S8 | Gating the bias to requests with no cue word removes the over-tuning (amendment 1, record 4483, declared before its code) | PARTLY | DEV explicit decisions unchanged 300/300; TEST explicit requests that contain a cue word unchanged (199/200 before and after); the 100 TEST explicit requests with no cue word are still overridden, as stated in advance (overall TEST explicit 73.0% for every user). Under gating the shift's size stops mattering on TEST: Delta-1, Delta and Delta+16 logits give the same TEST numbers |
| S9 | Few preference events suffice | VERIFIED | one DEV event already gives 96% (essay), 95% (demo) and 81% (song) on TEST ambiguous requests; 8 events give 99%, 95%, 95% |
| S10 | The margin certificate lets most decisions skip recompute after an update | REFUTED here | 0, 1 and 0 of 380 TEST decisions had margin above twice the shift (shifts are 6-10 logits, margins smaller); the one skipped decision was unchanged on recompute |
| S11 | Decomposed generation makes edits cheaper | VERIFIED | after replacing segment 0, decomposed mode rebuilt with 776 model calls against 2,214 for chained mode (65% fewer); every untouched segment bitwise identical (40/40) and the result equal to a full decomposed rebuild (40/40); chained mode changed 91 later segments |
| S12 | Decomposition costs coherence | small here | repeated segments: 72 of 159 decomposed, 65 of 159 chained; distinct bigrams 0.352 vs 0.358 (amendment 2, record 4487). Both are high because GPT-2 small repeats itself |
| S13 | Text-level reversibility | 39/40 | one reply's text re-tokenizes differently ("a guide to quasars"); reversibility holds at the token level, not always through text |

Not claimed: the quality of GPT-2 small's songs, essays or demos (they are
repetitive and often off-topic), real users, other models.

## Use cases: what determinism unlocks, and what pinning the seed already does (2026-11-21)

Contract tasks/USECASE_CONTRACT.json (record 4660) and 40 hand-written
prompts sealed before any code (tasks/usecase_prompts.json). GPT-2 small,
24 tokens per prompt. Five methods: prlm-0, greedy, sampling at
temperature 0.7 and 1.0 with the seed pinned per prompt, and sampling with
a fresh seed (as usually served). Backend B is the same weights in numpy
with a KV cache, which does not reproduce backend A's logits bitwise.

| # | Use case | Verdict | Numbers |
| --- | --- | --- | --- |
| U1 | Check an output without regenerating it | unlocked by pinning randomness, not by the rule | one batched forward checks all 24 steps (1 call instead of 24) for prlm-0, greedy and both seeded samplers alike: 40/40 honest outputs accepted, 40/40 tampered copies rejected. Fresh-seed sampling cannot be checked. GAMED verifier (accepts all): 40/40 accepted, 0/40 tampering caught |
| U2 | Replay on a different backend | UNLOCKED by determinism | identical outputs on backend B: prlm-0 40/40, greedy 40/40, seeded t0.7 37/40, seeded t1.0 34/40. Scores differ between backends by at most 4 units (1/64 logit), and prlm-0's certified margin exceeds twice that on 935 of 960 steps, so those steps are proved stable without comparing tokens. Greedy makes the same choices but carries no certificate |
| U3 | Cache whole answers | unlocked by pinning randomness | on a 2,000-request Zipf workload over the 40 prompts, any deterministic or seeded method needs 960 model calls instead of 48,000 (98% fewer) and returns exactly what it would recompute; caching fresh-seed sampling would change its output distribution |
| U4 | Tests that do not flake | unlocked by pinning randomness | two fresh processes give identical outputs on 40/40 prompts for prlm-0, greedy and both seeded samplers, 0/40 for fresh-seed sampling |
| U5 | Cost of determinism | LOSS | repeated trigrams: prlm-0 0.081 against 0.011 (t0.7) and 0.003 (t1.0); distinct bigrams 0.88 against 0.97 and 0.99. On T1 (report.py) sampling at t1.0 also had lower reference overlap than prlm-0 (ref_f1 0.156 vs 0.202) |
| U6 | Cheap edits, verified skipping, sparse behaviour diffs | cited, not re-run | CLAIMS S11 (decomposed edits, 65% fewer calls), PR 2 (verified drafting, 479 to 310 calls, 0 mismatches), P1-P5 and L7 (sparse rule equals dense recompute on every decision) |
| U7 | Pareto front over the five methods (eight objectives) | every method is on it | Lean's front equals Python's: all five methods, because each wins on something. Deterministic methods win on backend replay; sampling wins on diversity |

The plain reading: most of what is usually credited to determinism
(checking without regenerating, caching, flake-free tests) comes from
pinning the randomness, and seeded sampling gets it too. What the
deterministic rule adds on these numbers is replay that survives a change of
backend, with a certificate saying which steps are safe. The integer scores
are also what make exact sparse reweighting and Lean replay possible (earlier
sections). It pays for this with more repetition. prlm-0 and greedy made the
same choices on every step here.

## Cross-hardware audit (2026-11-26 to 27)

Contract tasks/XHW_CONTRACT.json (record 4980), amendments XHW_AMEND_1
(4984, five current models) and XHW_AMEND_2 (4985, Inkling 2 Mini), all
logged before their code. The "written" times inside XHW_CONTRACT,
FIXPOINT_AMEND_4 and GAP_CONTRACT were typed in, not read from a clock; the
real write times were about 20:10-20:17Z. The log order (each record before
the commit that adds its code) is the evidence of before-code, not those
fields. Same 40 sealed prompts, 24 tokens each (960 teacher-forced steps
per run), declared bound B = 4 score units (1/64 logit).

Machines: this container (R, Intel Xeon), and GitHub's ubuntu-24.04 (x86:
Intel Xeon 6975P-C or AMD EPYC 9V74, depending on the job), ubuntu-24.04-arm,
macos-latest (Apple M1, virtual), windows-latest (x86: AMD Zen 3/Zen 5 or
Intel, depending on the job) and windows-11-arm. Results were read from the
compare jobs' logs (xhw-compare runs 45924158972 and 45924385434), because
artifact downloads are blocked from this container; copies in
out/xhw/ci/.

| Model | Machines | Max score diff (X2) | Certified flips (X3) | Bound B=4 | Certified steps (margin > 2B) | prlm-0 / greedy identical (X4) | Lean (X5) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 small, ONNX Runtime, 4 configs | 6 of 6 | 3 | 0 | VERIFIED | 935/960 | 40/40 and 40/40 on every machine and config | exact (R, both Linux runners) |
| Qwen3.5-0.8B-Base (DeltaNet hybrid) | 4 of 5 | 1 | 0 | VERIFIED | 928/960 | 40/40 and 40/40 | UNMEASURED (Lean exits -6, likely memory, 248k vocab) |
| Qwen3-0.6B-Base (dense) | 5 of 5 | 1 | 0 | VERIFIED | 935/960 | 40/40 and 40/40 | exact |
| LFM2-350M (conv + attention) | 5 of 5 | 1 | 0 | VERIFIED | 930/960 | 40/40 and 40/40 | exact |
| Granite 4.0 H 350M (Mamba2 + attention) | 5 of 5 | 1 | 0 | VERIFIED | 927/960 | 40/40 and 40/40, except windows-latest: 39/40 and 40/40 | exact |
| Granite 3.1 1B-A400M (MoE) | 5 of 5 | 1 | 0 | VERIFIED | 932/960 | 40/40 and 40/40 | exact |
| Inkling 2 Mini, IQ1_S GGUF, llama.cpp | 2 of 2 (x86, Arm) | 1524 | 18 | REFUTED | 3/960 under the observed bound | 19/40 and 20/40 on Arm | exact on both |

| # | Claim | Verdict | Numbers |
| --- | --- | --- | --- |
| X1 | Float logits bitwise equal across machines | no verdict (by contract) | GPT-2: windows-latest equals R on all 960 steps with graph optimization on; the ubuntu-24.04 x86 runner on 246; every Arm and Mac machine on 0. Current models: 0 steps on every machine other than the reference runner. Changing threads alone changes some floats on the same machine (912 or 792 of 960 for three models) |
| X2 | Integer scores stay within the declared bound | VERIFIED for GPT-2 and the five current models; REFUTED for Inkling | largest difference 3 (GPT-2), 1 (current models), 1524 (Inkling) |
| X3 | Every step with margin > 2B keeps its certified pick on every machine | VERIFIED for GPT-2 and the five current models; REFUTED for Inkling | 0 disagreements on certified steps over every measured machine and config; Inkling: 31 disagreements on 960 steps, 18 of them on steps certified under B = 4 |
| X4 | Free generation identical to the reference | prlm-0 never beats greedy; it loses twice | prlm-0 and greedy 40/40 everywhere except Granite 4.0 H on windows-latest (prlm-0 39/40, greedy 40/40: one pick changed at step 739, where the reference margin was 1 unit: on windows-latest the two top scores came out equal (margin 0) and the tie went to the lower token id; the certificate had not covered that step, and greedy's float argmax kept its pick) and Inkling on Arm (19/40 against 20/40). Seeded sampling 34-40 (t0.7 and t1.0) on the current models, 36-40 on GPT-2, 4/40 and 1/40 on Inkling |
| X5 | Lean replays each Linux machine's records exactly | VERIFIED where it ran | 960/960 agree on both Linux runners for GPT-2, four current models and Inkling; Qwen3.5 UNMEASURED (process abort, no verdict line) |

Controls. Do-nothing: every machine's fresh-process repeat of its reference
config is bitwise identical to its first run (every float hash and every
free generation) wherever the repeat ran. Gamed: with the bound chosen after
the data (B' = 1 for the current models, 3 for GPT-2) the certificate would
cover 946-955 of 960 steps instead of 927-935; the declared B is the claim.

UNMEASURED: Qwen3.5 on macos-latest (the job hit the 5-hour limit before its
first config finished) and its fresh-process repeat on windows-11-arm (same
limit); Qwen3.5's Lean replay; Inkling on Mac and Windows (left out by the
amendment: memory, no Windows build).

Verdict rule (contract): (a) X3 holds with the declared B for GPT-2 and the
five current models, and fails for Inkling. (b) prlm-0's identical share is
never above greedy's, and is below it twice, so the outputs themselves are
not an unlock: greedy argmax replays just as well. (c) What prlm-0 adds is
the certificate: it names in advance 927-935 of 960 steps (97%) that will
not change on any of these machines, and none of them did. Inkling at 1.6
bits per weight shows the limit: llama.cpp's x86 and Arm kernels quantize
and accumulate differently, scores move by up to 1524 units (6 logits), and
a fixed bound cannot certify anything. The certificate is only as good as
the declared bound, and the bound has to be measured per engine.

## Gamed control at like resolution (2026-11-27)

Contract tasks/GAP_CONTRACT.json (record 4983), logged before harness/gap.py.
Record 4978 had the gamed ladder (tuned on TEST, stopped at level 8) strictly
dominating 11.95% of the main level 10 front on TEST. This scores both
fronts on both splits at every level 0-8 (records 4987, 4988).

| # | Measure (level 8; strict, weak) | Verdict | Numbers |
| --- | --- | --- | --- |
| G1 | Gamed advantage on its own split (TEST) | measured | gamed dominates 11.47% of main (weak 83.19%); main dominates 0.04% of gamed (weak 62.95%) |
| G2 | Main advantage on its own split (DEV) | measured | main dominates 16.65% of gamed (weak 88.61%); gamed dominates 0.46% of main (weak 49.57%) |
| G3 | Gap at like resolution, G1 minus G2 | main ahead by 0.0518 | about 0 at levels 0-2, then -0.016, -0.026, -0.044, -0.062, -0.064, -0.052 at levels 3-8 |

Reading rule: a gap within 0.05 would have meant a symmetric
generalization gap; a gamed lead above 0.05 an open gap. The result is
neither: main leads by 0.0518, a case the rule did not name, and just
outside the symmetric band. Tuning on DEV wins on DEV by more than tuning on
TEST wins on TEST, so the 11.95% is not evidence that the ladder is a weaker
optimizer than tuning on the scored data. Resolution accounts for little of
it: 11.95% at gamed level 8 against main level 10, 11.47% at level 8 against
level 8.

Controls. The contracted do-nothing (each front against itself on its own
split) gives strict share 0 at every level (record 4988, harness/gap_control.py).
gap.py had logged a different self-comparison under the same name, each front
against itself on the other split; it was added to record 4988 after the
data, so it is reported as exploratory: on the split it was not selected on, a
front's own points dominate each other at 11.09% (main on TEST) and 15.63%
(gamed on DEV), close to G1 and G2. Most of either side's advantage is the
spread any front shows on the split it was not selected on.

## Value case: does certified decoding beat greedy and sampling on quality (2026-11-27, in progress)

Contracts: tasks/VALUE_CONTRACT.json and amendments 1-5 (records 4989-4994), each logged before its code.

| id | claim | status |
|---|---|---|
| D1 | Generation stopped at the end-of-document token | **CORRECTED** (amendment 5). It did not: answers were read from a made-up continuation. Rescoring run 45984809235: the do-nothing control holds on every row; 14B greedy 21% -> 69%, margin-branch 16% -> 76%, sc-8 72% -> 75%. Earlier 14B numbers are withdrawn. |
| D2 | Branch points lie before the end of document | **REFUTED** for every row (14B margin 119 points in 61 of 100 rows; MBPP random-branch 61 and 23). Rows regenerated with the stop (workflow value5); rescored figures are provisional until then. |
| V1-V3 | margin-branch-8 beats sc-8 / greedy / random-branch-8 (math) | provisional, rescored: 1.7B (280 of 400) vs greedy +5.4 (WIN), vs sc-8 -1.4 (TIE); 4B TIE; 14B TIE |
| C1-C3 | the same on MBPP with a visible-test filter | margin rows not contaminated: 0.6B vs bo8 -0.4 (TIE), vs greedy +11.3 (WIN); 1.7B TIE / +9.3 (WIN) |
| P2 | portfolio-27 clears the 3-point bar | provisional: MBPP 0.6B +5.4 over bo8-filter (p=0.007), +17.1 over greedy; MBPP 1.7B +3.5 (p=0.064, TIE); math rows TIE. It costs the sum of the baselines' compute. |
| P1, Q3 | checker-monotone selection (Lean: portfolio_dominates_all, ratchet_monotone) | proved; measured violations 0 on both MBPP rows |
| L1, L2 | lazy ratchet (amendment 6) selects exactly what the eager ratchet selects (Lean: lazyFirst_sel, decided_mode) | proved; measured equal on 1,224 of 1,224 problems across six rows (run 45987064898, out/value/LAZY_RESCORED.json) |
| L3 | the lazy ratchet generates fewer tokens than the eager one at the same accuracy | holds: MBPP 0.6B 920 -> 270 tokens a problem (6.9 of 25 candidates), 1.7B 948 -> 215 (4.7 of 25); math 1.2x to 1.7x fewer |
| L4 | lazy-R4 on MBPP clears the 3-point bar over every baseline | provisional, all six WIN: 0.6B 69.3% vs bo8-filter 63.4% (+5.8, p=0.006) at 0.77x its tokens, vs margin-branch-8-filter +6.2 (p<0.001), vs greedy +17.5; 1.7B 79.4% vs bo8-filter 75.1% (+4.3, p=0.019) at 0.54x its tokens, vs margin-branch-8-filter +4.7 (p<0.001) at 0.92x its tokens, vs greedy +14.0. More accurate and cheaper than both sampling baselines at 1.7B, than bo8-filter at 0.6B. Provisional until the regenerated random-branch rows (value5) are scored. |
| L5 | the lazy ratchet beats sc-8 on math | **not supported**: TIE on all four math rows. R2 ties sc-8 at 0.89x to 1.04x its tokens; R4 costs 1.6x to 1.8x. It beats greedy on 1.7B (+5.75, p<0.001, 400 rows) and 14B R2 (+7, p=0.039) only. 0.6B covers 112 of 400 rows until the main run finishes. |
