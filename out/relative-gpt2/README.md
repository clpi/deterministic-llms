# Live GPT-2 exact-input comparison, November 22, 2026

40 sealed prompts, 24 new tokens per prompt, 50,257 scores per prefix.
Reference: GPT-2 small FP32 ONNX Runtime, serial per-token decode, 2 CPU
intra-op threads. All comparisons use its identical prefixes. Free
generation is measured independently, never used to filter rows.
The three FP32/tokenizer assets and dynamically regenerated INT8 asset
match all four original SHA-256 hashes in the historical log (0, 1,
2, 4410). This is a new execution on a pinned runtime, not reused historical results.

| Variant relative to reference | Same choices on common prefixes | Entire reference reply matched | Same free replies |
| --- | ---: | ---: | ---: |
| FP32 ORT batch vs FP32 ORT step | 960/960 | 40/40 | Not run (batch replay only) |
| FP32 NumPy KV vs FP32 ORT step | 960/960 | 40/40 | 40/40 |
| INT8 ORT step vs FP32 ORT step | 378/960 | 0/40 | 0/40 |
| INT8 ORT batch vs INT8 ORT step | 704/960 | 0/40 | Not run (batch replay only) |

The last row is the same INT8 artifact under two evaluation schedules.
It is not a FP32-vs-INT8 comparison. Fresh FP32 batch rows match after
integer rounding; the audit does not compare raw floats bit for bit.

FP32 NumPy KV drift across the corpus was [−2,+1] integer score units.
The observed interval-width test covers 949/960 choices and 29/40 entire
replies, versus 945/960 and 25/40 from a symmetric ±2 bound. Per-row
intervals cover 955/960 and 35/40, but are known only after the forwards.
All 40 emitted replies actually agreed. On prompts 0–19 the measured
interval was [−2,+1]. On prompts 20–39 all 480 rows stayed in that interval,
and its assumed width covers 473/480 choices and 13/20 whole replies.
That check is empirical, not a prospective error theorem.

No model forwards were skipped. This experiment identifies a stronger,
independently checkable condition for convergence across numerical routes;
it does not establish an output-quality win, a speedup, or a stop certificate.
The result archive stores per-prefix derived margins, drift extrema, and
picks, not full logit tensors. The checker recounts the archive; rerun
inference to reproduce the underlying measurements.

## Reproduce

Use Python 3.12 with the packages in `requirements-relative.txt`. Fetch the
[pinned official FP32 ONNX artifact](https://huggingface.co/onnxmodelzoo/gpt2-lm-head-10/tree/dd75887d8572cb74f1c670280ad1a001e671b7ae).
Fetch [encoder.json](https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json)
and [vocab.bpe](https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe).
In an isolated directory with those files, run
`python harness/rebuild_relative_int8.py "$PRLM_MODELS"` if the pinned INT8
artifact is absent. Verify the historical manifest rather than treating a
different rebuilt file as historical. The no-INT8 run uses only the three
FP32/tokenizer pins and omits `--int8-kind`.

```sh
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python harness/relative_replay.py new-run.json \
  --asset-manifest out/relative-gpt2/historical-asset-sha256.json --int8-kind historical
gzip -dc out/relative-gpt2/gpt2-relative-real.json.gz > observed.json
python harness/verify_relative_report.py observed.json out/relative-gpt2/historical-asset-sha256.json
python harness/relative_holdout.py observed.json
```
