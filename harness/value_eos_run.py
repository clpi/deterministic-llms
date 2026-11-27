"""Value amendment 5 (tasks/VALUE_AMEND_5.json, record 4994), regeneration of rows whose
branch points fell past the end of document: the same decoders as value.py, with the
end-of-document token (<|endoftext|> or <|im_end|>) added as a stop. The stop token is
kept in the path and dropped from the text before extraction, so a branch can still
replace it (the alternative to ending) but no branch can sit after it.

  python value_eos_run.py gsm14b OUTDIR --shard K --nshards N            llama.cpp, first 100 TEST problems
  python value_eos_run.py gsm MODEL OUTDIR --shard K --nshards N         HF float32, the 400 TEST problems
  python value_eos_run.py gsm4b MODEL OUTDIR --shard K --nshards N       HF bfloat16, first 100
  python value_eos_run.py mbpp MODEL OUTDIR --shard K --nshards N        HF float32, MBPP sanitized
Rows carry "amendment": "VALUE_AMEND_5" and live in their own folder and artifacts,
so they never mix with the uncorrected runs.
"""
import json, os, sys
import numpy as np
import value as V

EOS_IDS = (151643, 151645)     # Qwen3: <|endoftext|>, <|im_end|> (checked against the tokenizer at start)
EOS_TEXT = ("<|endoftext|>", "<|im_end|>")


def patch(lm):
    raw = lm.decode
    got = [raw([i]) for i in EOS_IDS]
    assert got == list(EOS_TEXT), got

    def dec(ids):
        ids = list(ids)
        i = next((k for k, x in enumerate(ids) if x in EOS_IDS), None)
        return raw(ids if i is None else ids[:i])
    lm.decode = dec
    return lm


def run_path(lm, start, prefix, prefix_recs, mode, rng=None, origin=None):
    """value.run_path with the end-of-document token as a stop."""
    tokens, recs = list(prefix), list(prefix_recs)
    row0, state = start
    if prefix:
        row, past = lm.extend(state, prefix)
    else:
        row, past = row0, lm.fork(state)
    gen = 0
    while len(tokens) < V.MAX_NEW:
        if mode == "greedy":
            v, w, m = int(np.argmax(row)), None, None
        elif mode == "prlm":
            v, w, m = V.certify(row)
        else:
            v, w, m = V.sample(row, rng), None, None
        tokens.append(v)
        recs.append((w, m))
        gen += 1
        if v in EOS_IDS:
            break
        text = lm.decode(tokens)
        if any(s in text for s in V.STOPS) or len(tokens) >= V.MAX_NEW:
            break
        row, past = lm.feed(past, [v])
    p = V.Path(tokens, recs, origin)
    V.finish(lm, p)
    return p, gen


V.run_path = run_path


def run(kind, model, outdir, shard, nshards, threads=4):
    os.makedirs(outdir, exist_ok=True)
    if kind == "gsm14b":
        import value_llama as L
        probs = [p for k, p in enumerate(V.load_test(None)[:L.N_14B]) if k % nshards == shard]
        lm = L.LlamaLM(threads)
    elif kind == "gsm4b":
        import value_code as C
        probs = [p for k, p in enumerate(V.load_test(None)[:C.N_4B]) if k % nshards == shard]
        lm = C.HF16(model, threads)
    elif kind == "gsm":
        probs = [p for k, p in enumerate(V.load_test(None)) if k % nshards == shard]
        lm = V.HF(model, threads)
    else:
        import value_code as C
        probs = [p for k, p in enumerate(C.mbpp_load(None)) if k % nshards == shard]
        lm = V.HF(model, threads)
    patch(lm)
    task = "gsm8k" if kind == "gsm" else kind      # same task names, so every summary applies; kept apart by folder
    path = os.path.join(outdir, f"eos-{kind}-{model.replace('/', '_')}-shard{shard}.jsonl")
    done = {json.loads(l)["index"] for l in open(path) if l.strip()} if os.path.exists(path) else set()
    info = {"task": task, "machine": V.machine(), "inputs": lm.inputs, "amendment": "VALUE_AMEND_5"}
    for pr in probs:
        if pr["index"] in done:
            continue
        if kind == "mbpp":
            import value_code as C
            r = C.solve_mbpp(lm, pr)
            key, ms = "pass", C.CODE_METHODS
        else:
            r = V.solve(lm, pr["question"], pr["gold"], pr["index"])
            key, ms = "correct", V.METHODS
        r.update(info)
        r["data_sha256"] = pr.get("data_sha256")
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(pr["index"], {m: int(r["methods"][m][key]) for m in ms}, flush=True)


def main():
    a = sys.argv[1:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    kind = a[0]
    if kind == "gsm14b":
        model, out = "Qwen3-14B-Base-Q4_K_M", a[1]
    else:
        model, out = a[1], a[2]
    run(kind, model, out, int(opt("--shard", 0)), int(opt("--nshards", 1)), int(opt("--threads", 4)))


if __name__ == "__main__":
    main()
