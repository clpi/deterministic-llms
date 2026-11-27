# deterministic-llms (PRLM)

Parametric deterministic language models: generation as a deterministic
function of the content and a few named, meaningful weights, decided by a
certified best-first rule and rewritable incrementally.

- `SPEC.md`: the definitions, the hypergraph form, what is proved, the
  contract, the alternative constructions, the decompiling hypothesis.
- `CLAIMS.md`: every claim with its verdict (VERIFIED, CORRECTED, REFUTED,
  UNMEASURED) and its receipt.
- `lean/`: Lean 4.19.0 package. `Prlm/Core.lean` holds the definitions and
  proofs; `Main.lean` is the executable that replays recorded decoding
  steps through those same definitions; `Prlm/Skip.lean` proves draft acceptance for forward skipping;
  `Prlm/Sparse.lean` proves exact sparse reweighting (`best_split`);
  `Prlm/Loop.lean` proves the search loop's archive and scope rules;
  `Prlm/Fix.lean` proves what a certified stop means; `Prlm/Struct.lean`
  proves minimal preference updates, composition and reversible rendering;
  `Prlm/AlgoSearch.lean` is copied
  unchanged from software-redundant-compute-expands PR 7 (Pareto archive,
  Chebyshev weights). `check.sh` is the cold gate.
- `tasks/`: the frozen tasks (`make_tasks.py` builds `tasks.json`) and the
  contract, both logged before any axis or decoder code existed.
- `harness/`: model wrapper, axes, decoders, the contract run (`run.py`),
  the post-hoc adaptive-weight study (`adapt.py`), the forward-skipping
  test (`skip.py`, contract `tasks/SKIP_CONTRACT.json`), sparse reweighting
  (`reweight.py`) and n-gram drafting (`skip2.py`), both under
  `tasks/IMPROVE_CONTRACT.json`, and `report.py`, which
  rebuilds every number from the log.
- `harness/models.py`, `harness/multi.py`: six models through one adapter surface
  (contract `tasks/MULTI_CONTRACT.json`). `harness/arch.py`, `harness/archrun.py`:
  architecture as a parameter, ten families, random and trained weights
  (`tasks/ARCH_CONTRACT.json`). `harness/density.py`: density computed from
  each weight change and the Pareto front over 243 changes
  (`tasks/DENSITY_CONTRACT.json`). `harness/multireport.py` writes
  `out/REPORT-multi.md` from the log; `out/multi-int8-t3.jsonl` is the int8
  task run's own hash-chained log (`out/REPORT-int8-t3.md`).
- `harness/loop.py`: the self-improving, scope-solving Pareto search
  (`tasks/LOOP_CONTRACT.json`, proofs in `lean/Prlm/Loop.lean`).
- `harness/fixpoint.py` and `harness/fixladder.py`: the same loop run to a
  certified fixed point, at exact resolution (refuted) and then one
  resolution at a time (`tasks/FIXPOINT_CONTRACT.json`,
  `tasks/FIXPOINT_AMEND_1.json` to `tasks/FIXPOINT_AMEND_3.json`, proofs in
  `lean/Prlm/Fix.lean`).
- `harness/structure.py`: output structure (essay, song, steps, demo,
  table) as a certified pick on GPT-2, with revealed-preference updates,
  reversible rendering and decomposed editing (`tasks/STRUCT_CONTRACT.json`
  and its two amendments, proofs in `lean/Prlm/Struct.lean`).
- `tests/`: a 3-step fixture of recorded GPT-2 steps and a tampered copy;
  CI (`.github/workflows/check.yml`) runs the cold gate, replays the
  fixture (must pass), the tampered copy (must fail) and checks the log chain.
- `out/log.jsonl`: the hash-chained, append-only log of every input,
  output and measurement. `out/REPORT.md` is generated from it.

## Rerun

```
# model files (GPT-2 small LM head, ONNX; tokenizer pinned by sha256)
#   gpt2-lm-head-10.onnx from onnx/models, encoder.json and vocab.bpe
#   into $PRLM_MODELS (default /tmp/claude-0/models)
pip install onnxruntime numpy tiktoken onnx
bash lean/check.sh                          # cold build + hygiene gate
cp lean/.lake/build/bin/prlm /tmp/claude-0/prlm
cd harness
PRLM_LOG=../out/rerun.jsonl python3 run.py declare t1 t2 t3 repro alt sweep replay
python3 adapt.py
python3 skip.py && python3 skip.py origin
python3 reweight.py && python3 skip2.py
# more models and architectures (T5 files from the onnx/models zoo and
# Google's t5-data bucket; int8 made with onnxruntime.quantization; jax for training)
python3 multi.py run
python3 archrun.py run <arch contract record id>
python3 density.py <density contract record id>
python3 multireport.py
python3 loop.py <loop contract record id>
python3 -c "import pickle,density as D;from common import Model;pickle.dump(D.record_steps(Model(4)),open('/tmp/claude-0/steps_t3.pkl','wb'))"
python3 fixpoint.py <fixpoint contract record id>      # exact resolution
python3 fixladder.py <amendment 2 record id> rerun    # resolution ladder; resumes from
                                    # checkpoints after a restart (LADDER_CKPT_DIR, default
                                    # prlm-ckpt next to the repo); LADDER_GAMED_STOP=1
                                    # stops the gamed control as amendment 3 describes
python3 structure.py <struct contract record id>
python3 structure.py gated <amendment 1 record id>
python3 structure.py dup <amendment 2 record id>
python3 report.py
```

The record sha256s in the log name the exact inputs; a rerun on another
machine is a new log, compared record by record.
