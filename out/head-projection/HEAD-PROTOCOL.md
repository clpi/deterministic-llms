Same exact GPT-2 FP32 weights, sealed 40 prompts, 24 output tokens, two ORT
CPU threads. Five arms: full-output serial ORT; last-position serial ORT;
last-position NumPy KV without target verification; full-projection live
NumPy draft plus full-target batch; last-projection NumPy draft plus
full-target batch. Run one warm-up for every arm; shuffle prompts with the
declared seed and balance arm positions by rotation. Keep the old full
serial observer unchanged. Compare whole emitted token histories, chosen
score, tie-adjusted winning allowance and fixed stop separately from
latency, transformer positions, vocabulary projection rows and calls.
No change to past/key-value work in the head projection. Include corrupt
draft control with exact serial fallback. Compare measured Pareto fronts
over these five realized candidates only, with a complete reply and with
per-step decision-receipt equality as two different constraints.
