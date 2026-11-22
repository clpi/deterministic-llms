# PRLM: parametric deterministic language models

Spec, version 1, 2026-11-20. Status of every claim is in CLAIMS.md. Proofs
are in `lean/Prlm/` (`Core`, `Skip`, `Sparse`); the executable that runs them on real model
scores is `lean/Main.lean`; the harness is `harness/`.

## 1. Objects

**Content.** A token prefix `c` (a list of token ids). In the one canonical
hypergraph, a content is a node identified by the sha256 of its token bytes
(`Model.key`). Two runs that reach the same prefix reach the same node.

**Candidate edge.** For content `c` and token `v`, the edge `c → c·v`. The
edges leaving `c` are the candidates of one decoding step.

**Base score.** `base(c, v)`: an integer, a function of `c` and `v` only. In
this repo it is `round(256 * logit)` from GPT-2 small (124M, ONNX, CPU),
so the rule above it is exact integer arithmetic. Whether the float logits
themselves are a function of `c` alone (across runs, thread counts and
machines) is an empirical property of the runtime, measured, not proved.

**Named axes.** A finite list of axes `a_1 .. a_k`, each with a meaning
(`rep`: earlier occurrences of the token; `noloop`: the token would repeat
a 3-gram; `ban`: the token is a listed word; `topic`: the token starts a
listed word; `end`: the token ends a sentence; `embdir`: cosine of the
token's embedding with a direction built from words; `null`: always 0;
`oracle`: reads the hidden answer, gamed control). Each axis gives an
integer feature `φ_a(c, v)`. On the hypergraph `φ_a(c, v)` is a color of
the edge `c → c·v`; an axis with value 0 leaves the edge uncolored.

**Parameters.** A weight vector `θ ∈ ℤ^k`, one weight per named axis, in the
same units as the base score (1/256 logit). These are the meaningful
parameters: every weight answers "how much does this property of the next
token matter", and nothing else moves the output.

**Determinant.** `S_θ(c, v) = base(c, v) + Σ_a θ_a φ_a(c, v)`
(`Prlm.scores`, with `dot` the weighted sum).

**Certified best-first rule.** `pick_θ(c)` is the candidate with the
highest `S_θ`, the lowest token id among equal scores (`Prlm.best`, the
strict order `Better`). Its certificate is `(v*, S*, m)` with
`m = S* − max_{v ≠ v*} S_θ(c, v)` (`Prlm.certify`). The pick is
**certified** when `m > 0`: no other token beats or ties it. When `m = 0` the
pick is **accepted** (a tie broken by token id), never called certified.

**Generation.** `gen_θ(n, c)`: apply the rule `n` times, appending each pick
(`Prlm.gen`). The fully deterministic limit of the program is this
feed-forward map from (content, θ) to output: no sampling, no seed.

**Cone.** For a certified step with margin `m`, the set of weight changes
`Δ` for which the pick is unchanged contains the box
`{Δ : |Σ_a Δ_a φ_a(c, v)| < m/2 for every candidate v}` (`stable_of_bound`).
The exact region is the polyhedron
`{θ' : S_θ'(c, v*) > S_θ'(c, v) ∀ v ≠ v*}`, convex, and the box is a
sufficient test that needs only the margin and the moves.

## 2. Hypergraph form

Everything is a record of one append-only, hash-chained log
(`out/log.jsonl`, records `{id, level, colors, premises, head, receipt,
prev}`, the shape of `Canon.Rec`):

- inputs (model, tokenizer, tasks, contract) are root records with a sha256
  receipt;
- every output of every system is a record whose premises are the
  frozen inputs and the declared weights;
- axes are colors on candidate edges; a filter is a meet with a color (a
  hard filter is a weight of −4000 logits, which no base score overcomes on
  this model), and a view is a join in the semilattice of color sets: the
  view "topic ∧ ¬ban" is the set of edges carrying both colors;
- base scores are cached by content node (`Model.cache`), so a second run
  with different weights reads the same nodes and adds only the nodes its
  trajectory newly reaches. This is append-only by construction: nothing
  computed under one weight vector is overwritten by another.

## 3. What is proved (Lean 4.19.0, no Mathlib, no sorry)

| Claim | Theorem | Statement |
| --- | --- | --- |
| Determinism (order) | `best_perm`, `gen_perm` | The pick, and every generation, is the same for any order in which candidates are enumerated or merged. |
| Determinism (function) | `gen` is a total function | Same content and θ give the same output; no hidden state. |
| Certification soundness | `certify_sound` | A certified pick strictly beats every other token, by at least the margin. |
| Certification completeness | `certify_complete` | A strictly best token is picked and certified. |
| Uniqueness | `max_unique` | The unbeaten candidate is unique. |
| Reweighting equals recompute | `rescore_eq`, `stepRe_eq`, `gen_reweight` | Adding `Δ`'s contribution to the scored list equals rescoring from base with `θ + Δ`, per step and for whole generations. |
| The cone | `stable_of_bound` | Inside the margin box the pick is unchanged, decided without recomputing base scores. |
| Parameter locality | `scores_local`, `dot_set_zero`, `gen_local` | An axis that is zero on the candidates leaves the scores, the pick and every generation unchanged, whatever its weight. |
| Caching is invisible | `gen_cache` | A cache of base scores that is correct on its keys gives the same generation. |
| Prefix | `gen_prefix` | Outputs extend their content. |
| Draft acceptance (forward skipping) | `accept_gen`, `accept_batch` | Tokens accepted from a drafted continuation are exactly the rule's next tokens; equal scores give equal acceptance (the batched-forward equality is a PREMISE, measured). |
| Exact sparse reweighting | `Sparse.best_split` | After a weight change, the pick equals the best of the moved candidates plus the best unmoved one (the first unmoved in the old order); exact, no margin needed. |
| Pareto tools (reused, srce PR 7) | `AlgoSearch.mem_arch`, `cheb_target` | The archive is exactly the Pareto front, order independent; Chebyshev weights solved from a front point make it a minimizer. |

The link to behavior: `lean/Main.lean` is compiled from these definitions.
`prlm decide` reads every recorded decoding step (all 50,257 integer base
scores, the weights, the sparse axis features) and recomputes the pick,
the certificate, the rescored pick and the recomputed pick with the proved
functions, then compares them with the harness's picks. A theorem here is
about the function that decided the runs, not about typed-in tables.

What the proofs do not cover: that the float forward pass is
deterministic (measured), that an axis means what its name says
(measured with held-out words and controls), that the outputs are good
(measured), and anything about sampling.

## 4. Empirical contract

`tasks/CONTRACT.json` (frozen before any axis or decoder code; log record
4). Same model, tokenizer, prompts, lengths, machine and metric code for
every system. Systems: three samplers (5 seeds each), greedy on floats,
PRLM with weights 0, PRLM with declared weights, alternatives (beam with a
certificate, best-first search with a certificate, a trigram automaton, a
rule table distilled from greedy). Every axis runs with a do-nothing
control and, where the metric holds hidden information, a gamed control
labelled GAMED. The weights were declared in `out/DECLARED.json`, logged
before the run. Tasks were frozen before the axes existed.

## 5. Alternative constructions

1. Certified best-first over integer scores (this spec).
2. Beam search with a certificate: optimal for its length when the best
   finished beam is no worse than the best pruned prefix.
3. Best-first (uniform-cost) search: exact most likely sequence of a given
   length, certified when it finishes inside its call budget.
4. Lookup automaton: trigram counts with backoff, argmax, no network.
5. Rules distilled from the network: (last 3 tokens) → greedy next token
   where every training occurrence agreed; the network is called only
   when no rule fires.
6. Constrained decoding as a filter: a hard-weight axis (`ban`).

Pareto selection among systems uses the front and front-derived Chebyshev
weights (`prlm front`), not a blind weighted sum.

## 6. Decompiling weights (hypothesis H-D)

H-D: an existing model becomes a PRLM by decompiling directions in its
weights into named axes. Refutation test used here, on the one direction
family that was measured: an axis built from the embedding direction of a
word list's TRAIN half must raise the rate of the HELD-OUT half more than
the same-weight random-direction control does, at no larger fluency cost.
If it does not, H-D is refuted for embedding directions. Everything beyond
that (attention or MLP weights, a whole model rewritten as a PRLM,
executable compressed bytes) is UNMEASURED.

## 7. Models and architectures (2026-11-21)

**Which path defines base(c, v).** The rule, the certificate, sparse
reweighting and the Lean replay act on recorded integer scores, so they
are exact for any model. Whether those scores are the same along
different computation paths depends on the model and the runtime. The
paths are a full recompute per step, one call over a longer sequence, and
a KV cache or recurrent state. A PRLM therefore fixes one path as the
definition of base(c, v). Here that is the full per-step recompute. Any
other path, such as drafting verification, a cache or quantized kernels,
is admitted only after a measured check against it. On the models here,
float32 paths kept every pick, but dynamic int8 quantization did not
(CLAIMS.md M3 to M5).

**Architecture as a parameter.** `harness/arch.py` builds a model from a
config: layer kinds (attention, selective SSM, RWKV, GRU, LSTM, mixed),
positions (learned, RoPE, ALiBi), KV heads (MHA, GQA, MQA), feed-forward
(GELU, SwiGLU, MoE top-k, RWKV channel mix) and norm (LayerNorm, RMSNorm).
Every property is measured through two entry points only, `full` (all
positions in one call) and `step` (one token through the cache or state).
That way a new config gets the same checks without new measurement code.

**Density.** Density is the computed consequence of a weight change, not
a parameter. At content c, a change Δ moves the candidates
{v : Σ_a Δ_a φ_a(c, v) ≠ 0}. Density is their share of the vocabulary,
and the exact work of deciding the step is |moved| + rank of the first
unmoved candidate + 1 (`Sparse.best_split`). A search over weight changes
treats that work as one objective among the behavioural ones. The front
comes from `prlm front` (AlgoSearch.arch), recomputed after every
appended point.

## 8. The self-improving loop (2026-11-21)

`harness/loop.py` runs the search that the previous sections define,
with nobody supervising it. Each round does four things. It solves
scope: an axis whose features are zero on every candidate is removed,
and `Loop.scope_sound` proves that loses nothing. An axis whose weight
never changed a decision is frozen and checked again every round. It
proposes points around the current front. It admits each one by the
archive rule. And it checks the archive against Lean's `prlm front`.
`Loop.arch_covers` is the safety property: appending any batch of
candidates leaves every inherited front point on the front or
dominated by a new one, so no round can lose ground. `admit_sound` and
`reject_sound` show the archive alone decides admission. Selection uses
DEV prompts, and improvement is claimed only on TEST.

## 9. Stopping at a certified fixed point (2026-11-21)

PR 4's loop stopped on a budget. `harness/fixpoint.py` removed the budget
and found that at exact resolution there is no fixed point in reach: the
front doubled every round (12,618 points after 39,106 evaluations). With
six objectives, two of them nearly continuous, almost every refinement is
incomparable with the rest.

So the stop is certified one resolution at a time (`harness/fixladder.py`).
Objective i is read in steps of r_i, starting coarse and halving. At each
level the archive keeps one representative per cell, and the level ends
when no neighbour of any representative is left unevaluated.
`Fix.point_local_opt` and `Fix.closed_local_opt` say that stop is a
certificate: no unit neighbour of a front point dominates it. By
`Fix.coarse_front_within`, anything that exactly dominates a front point
does so by less than one resolution step in every objective. Every
evaluation carries over to the next, finer level. The result is the finest
level certified within the evaluation cap.

## 10. Output structure (2026-11-21)

A reply's structure is one more certified pick: candidates are structures,
base scores come from the model, and named axes (cue words, per-structure
bias) add to them, so Core's theorems apply unchanged. `Struct.lean` adds
what this layer needs. `minshift_wins` and `minshift_minimal` give the
smallest bias shift that makes a revealed choice win: enough, and no more,
on the events observed. `rescore_comm` lets updates compose in any order.
`replan_local` makes replacing one segment's inputs change only that
segment's decision. `parse_render` and `render_set` make a structured reply
reversible, and editable one segment at a time.

Minimal on the observed events is not the same as harmless elsewhere. The
GPT-2 run shows where the parameter, not the shift, goes wrong: a global
bias also moves requests that name another structure (CLAIMS S7, S8).

## 11. What determinism is for (2026-11-21)

Measured on sealed prompts against greedy and seeded and unseeded sampling
(CLAIMS U1-U7). Checking an output in one batched forward, caching whole
answers and flake-free tests need only that the output be a function of
its inputs. Pinning the sampler's seed gives that too, so these are not
unlocks of the certified rule. What the rule adds is integer scores with a
margin. With them the same output replays on a backend whose logits differ
in the last bits, and the margin says in advance which steps are safe.
Integer scores also make exact sparse reweighting, incremental edits and a
Lean replay possible. The price is diversity: the rule repeats itself more
than sampling.
