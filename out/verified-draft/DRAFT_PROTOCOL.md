Run paired CPU comparisons on the same 40 sealed GPT-2 prompts, 24 new tokens.
Tie rule: first ID at maximum rounded-to-1/256 score. Stop is fixed at 24.
The cold serial target is FP32 ONNX Runtime CPU with two threads. A live FP32
NumPy KV draft is checked against the FP32 target with one teacher-forced batch.
If any draft token disagrees with its row's target choice, run the cold serial
fallback from the beginning. Serial target replay is compared independently
in the experiment; no future equivalence between numerical routes is assumed.
Record target calls, target positions, draft calls/time, total time and exact
reply plus observed per-step decision certificates. A changed draft is a
negative control. Prior measured drift is not used to skip any evaluation.
