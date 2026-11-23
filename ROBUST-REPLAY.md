# Tie-aware score certificates and authenticated fixture replay

An exact maximum drift certificate answers a practical question: if an
alternate runtime, compiler or quantization changes every integer score at
one recorded prefix by no more than R, can the chosen token change?

For chosen token w and rival j, let
d = score(w) - score(j) - indicator(j < w).
The pick survives **every** independent integer disturbance in [-R,R]
if and only if **2R ≤ d for every rival**. Lower token IDs win ties.
The minimum of floor(d/2) is therefore the largest allowed uniform
integer perturbation. A one-token vocabulary has no finite bound, printed
as none. The executable only requests this calculation for an exact winner.

Proof: the adversarial corner lowers w by R and raises a rival by R, changing
the pairwise difference by exactly 2R. The pair and whole-vocabulary arguments
and the executable minimum equivalence are checked in Prlm/Robust.lean.
No sign assumption for the original scores is needed. The result concerns
recorded integer scores, not floating-point kernel repeatability and not
every prefix a model might visit. Cross-runtime transfer still requires
an independent bound on actual score drift for each applicable prefix.

## Pinned GPT-2 fixture

The 604,296-byte fixture is the three full-vocabulary GPT-2 scores already
identified by the repository's fixture gate. Its SHA-256 is
97f7d942c46bb93fa34965e2112eb7a9076ea5cfc571810a8371c13e4e4ea0c2.
There are 50,257 candidates per record. Both Lean and the independent Python
checker get the same answer:

| Record | Original radius | Retuned radius |
|---|---:|---:|
| 1 | 28 | 163 |
| 2 | 21 | 21 |
| 3 | 225 | 33 |

Every one of the six next-integer perturbation controls changes the choice.
The uniform bound covering the recorded choices is 21 score units, including
the retuned choices on those supplied prefixes. No new model execution was run.

## Fail-closed replay

The complete fixture check binds digest and exactly three expected records;
a record can be correct without proving that the input file is complete.
The executable rejects empty input, partial headers or bodies, and indexes
that cannot refer to the supplied token and axis dimensions. A generic
standalone replay without the count option remains only a check of the
records it actually received. CI exercises 19 malformed or incompletely bound
variants without requiring a model download.

Run from the repo root after building the recorded Lean target:

    lean/.lake/build/bin/prlm decide tests/fixture.bin --expect 3
    lean/.lake/build/bin/prlm decide tests/fixture.bin --robust
    python3 tests/robust_replay_controls.py lean/.lake/build/bin/prlm tests/fixture.bin

### Asymmetric drift and whole replies

An exact per-token box certificate compares each rival's upper drift minus
the winner's lower drift against its tie-adjusted allowance. A single common
interval uses its width, not twice the largest absolute drift; adding the
same integer to all scores leaves the choice exactly unchanged. The exact
pointwise test uses the actual rival-minus-winner drift. This last test audits
scores already computed; it does not save any model calls. A reusable bound
must be established independently before omitting a forward.

For deterministic fixed-length generation, agreement at each exact reference
prefix implies agreement of the entire generated sequence. The converse for
same-input observations holds directly; a mismatch is not a prediction of
later generated tokens. This theorem does not certify stopping or other
output constraints.

`python3 harness/relative_controls.py tests/fixture.bin` cross-checks 15,625
three-token score/drift pairs against an independent total ordering. The
symmetric test accepts 925, the interval test 3,445, and the exact pointwise
test 10,210; 5,415 changed winners are all refused. On the three sealed GPT-2
prefixes at two recorded weight settings, artificial common shifts pass the
interval test, and the first adverse relative drift flips the winner.
These are simulated perturbations on recorded scores, not cross-runtime runs.

`PRLM_MODELS=/path/to/assets python3 harness/relative_replay.py NEW_REPORT.json
--asset-manifest exact-asset-sha256.json` exercises the original sealed 40×24
GPT-2 prompts with the FP32 NumPy KV and INT8 ORT comparison arms. Supply the
four SHA-256 entries for `gpt2-lm-head-10.onnx`,
`gpt2-lm-head-10-int8.onnx`, `encoder.json`, and `vocab.bpe`. Output records
per-reply coverage alongside independent free-generation equality; no
divergent reply is removed from teacher-forced measurements. Row and corpus
drift bounds in this report are post hoc observations only. The published
935/960 token aggregate alone allows 15–38 fully covered replies under the
old strict margin rule. Fresh live GPT-2 measurements remain unmeasured.
