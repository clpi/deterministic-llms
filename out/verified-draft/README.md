# Live verified drafting on the sealed GPT-2 workload

On November 22, 2026, FP32 NumPy KV drafted and the same pinned FP32 ONNX
Runtime model verified once per prompt using a teacher-forced batch. The
independent serial reference, the alternative, and the altered-draft control
all ran on the same Linux x86-64 CPU environment. All 40 sealed prompts
used exactly 24 output tokens. Paired runs alternated order, after warm-up.

| | Serial FP32 target | Live draft plus FP32 verification |
|---|---:|---:|
| Exact replies and decision receipts matching serial | Reference | 40/40 both |
| Target calls | 960 | 40 |
| Target input positions actually evaluated | 21,528 | 1,357 |
| Draft calls | 0 | 960 |
| Draft input positions | 0 | 1,357 |
| Observed sum of wall time | 58.24 s | 48.22 s |
| Per-prompt median wall time | 1,382.54 ms | 1,155.81 ms |

The verified lane took 17.2% less wall time on this single paired run,
including 44.48 seconds of real drafting; faster on 35 of 40 prompts.
Total model invocations increased from 960 to 1,000. The two backends
need not cost the same per evaluated position. Startup, warm-up, and the
separate altered-draft test are not in the paired sums. There was no
fallback on the sealed prompts. The altered token was refused and the
serial fallback recovered the same reply and decision receipt.

Each receipt contains the chosen token ID, its quantized score, and
its exact tie-adjusted winning allowance. The stopping rule is fixed
length, not a probability-mass or adaptive stop certificate. No regression
was observed against the cold serial route on these prompts; it is not
a proof of universal FP32 batch/serial equivalence. In the adjacent audit,
INT8 with the same artifact failed that equality. The companion Lean
theorem proves exact trajectory preservation with fallback **when checked
hits equal cold decisions at their exact prefixes**. This experiment
uses all-or-nothing batch verification and never uses the retrospective
score interval to skip work.

Use the FP32 and tokenizer assets and requirements in [relative-gpt2](../relative-gpt2/README.md).
```sh
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python harness/live_verified_draft.py --manifest out/relative-gpt2/historical-asset-sha256.json new.json
(cd out/verified-draft; sha256sum -c SHA256SUMS)
gzip -dc out/verified-draft/live-verified-draft.json.gz > observed.json
python harness/verify_live_draft.py observed.json
```
