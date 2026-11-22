# GPT-2 last-position projection against stronger serial baselines

November 22, 2026. Compare five modes on the same exact GPT-2 FP32 weights and
40 sealed prompts. Each emits 24 tokens. One 5-arm balanced run, shuffled
prompt order, every arm in every order position eight times. One warm-up per
arm excluded from timings. Linux x86-64 CPU, ONNX Runtime CPU provider with
two intra-op threads, two OpenBLAS and OMP threads. See the exact upstream
assets and runtime versions in the relative-audit records and the raw replay.

| Mode | Aggregate measured s | ORT + KV calls | Transformer input positions | Head rows | Replies match | Whole decision receipts match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-head serial ORT | 49.87 | 960 + 0 | 21,528 | 21,528 | 40/40 | 40/40 |
| Last-head serial ORT | 42.67 | 960 + 0 | 21,528 | 960 | 40/40 | 40/40 |
| Direct last-head NumPy KV | 35.49 | 0 + 960 | 1,357 | 960 | 40/40 | 25/40 |
| Full-head NumPy draft, ORT verification | 40.51 | 40 + 960 | 2,714 | 2,714 | 40/40 | 40/40 |
| Last-head NumPy draft, ORT verification | 39.36 | 40 + 960 | 2,714 | 2,317 | 40/40 | 40/40 |

Strongest observed serial baseline: 42.67 s at full receipt equivalence.
The optimized verified lane took 7.8% less elapsed time, not the 17.2% from
the earlier different baseline run. Projecting the draft's final hidden row
first removed 397 logical head rows and this run reduced whole-draft-lane
time by 1.15 s relative to its paired unoptimized draft arm. A head row is a
source-level count, not physical instructions or measured FLOPs.

The direct KV route was fastest and generated identical tokens, but 22 of
960 step receipts changed, affecting 15 prompts. A receipt includes the
winner, its quantized score and its tie-adjusted allowance. Over measured
wall time, transformer positions, head rows, and total model calls, direct KV
is the measured frontier under the reply-only observer. Last-head serial
ORT and verified last-head drafting are both undominated when exact receipts
are required. The former uses fewer calls and head rows; the latter uses
fewer transformer input positions and less observed time. These are fronts
within the five measured candidates, not general hardware optima.

The altered draft was refused and the exact serial fallback passed. The
same-weight ONNX transformation verified 2,533 upstream nodes and 345
inherited initializers unchanged. It moves the last-row selection before
the terminal head multiplication, preserving other outputs. Lean checks the
row-local mathematical commutation law, not floating-kernel equivalence.
This run validates only the declared observed decisions and fixed horizon;
no adaptive stop, model quality, memory or cross-runtime claim.

## Reproduce

Set `PRLM_MODELS` to the directory holding the pinned FP32 model and tokenizer
from [relative-gpt2](../relative-gpt2/README.md). Use the adjacent pinned packages.

```sh
python harness/gpt2_last_projection.py "$PRLM_MODELS/gpt2-lm-head-10.onnx" "$PRLM_MODELS/gpt2-last-head.onnx"
python harness/verify_head_model.py "$PRLM_MODELS/gpt2-lm-head-10.onnx" "$PRLM_MODELS/gpt2-last-head.onnx"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python harness/benchmark_head.py new.json --manifest out/relative-gpt2/historical-asset-sha256.json \
    --last gpt2-last-head.onnx --last-sha "$(cat out/head-projection/last-head-sha.txt)"
(cd out/head-projection; sha256sum -c SHA256SUMS)
gzip -dc out/head-projection/gpt2-head-qualification.json.gz > observed.json
python harness/verify_head.py observed.json
```
