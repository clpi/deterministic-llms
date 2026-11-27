"""Diagnose the 14B row of VALUE_AMEND_2 (greedy 21% against sampling 55%): print the
greedy and sampled completions and the first step's top-5 tokens for the first N problems.
Diagnostic only; nothing here changes a result.

  python value_llama_debug.py N
"""
import json, sys
import numpy as np
import value as V
import value_llama as L

lm = L.LlamaLM(4)
for pr in V.load_test(None)[:int(sys.argv[1])]:
    ids = lm.encode(V.PREFIX + f"Q: {pr['question']}\nA:")
    start = lm.extend(None, ids)
    p, _ = V.run_path(lm, start, [], [], "greedy")
    s, _ = V.run_path(lm, start, [], [], "sample", np.random.default_rng(pr["index"]))
    row = start[0]
    top = np.argsort(-row)[:5]
    print("DEBUG", json.dumps({"index": pr["index"], "gold": pr["gold"], "prompt_tokens": len(ids), "first_ids": ids[:5],
                      "top5_first": [[int(t), lm.decode([int(t)]), round(float(row[t]), 3)] for t in top],
                      "greedy_tokens": len(p.tokens), "greedy": p.text[:700], "greedy_answer": p.answer,
                      "sample_tokens": len(s.tokens), "sample": s.text[:700], "sample_answer": s.answer}), flush=True)
