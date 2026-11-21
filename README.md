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
  `Prlm/AlgoSearch.lean` is copied
  unchanged from software-redundant-compute-expands PR 7 (Pareto archive,
  Chebyshev weights). `check.sh` is the cold gate.
- `tasks/`: the frozen tasks (`make_tasks.py` builds `tasks.json`) and the
  contract, both logged before any axis or decoder code existed.
- `harness/`: model wrapper, axes, decoders, the contract run (`run.py`),
  the post-hoc adaptive-weight study (`adapt.py`), the forward-skipping
  test (`skip.py`, contract `tasks/SKIP_CONTRACT.json`), and `report.py`, which
  rebuilds every number from the log.
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
python3 report.py
```

The record sha256s in the log name the exact inputs; a rerun on another
machine is a new log, compared record by record.
