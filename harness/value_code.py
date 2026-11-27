"""Value amendment 1 (tasks/VALUE_AMEND_1.json, record 4990).

  python value_code.py mbpp MODEL OUTDIR --shard K --nshards N [--data JSONL]
  python value_code.py gsm4b OUTDIR --shard K --nshards N     Qwen3-4B-Base, bf16, first 100 TEST problems
  python value_code.py summary OUTFILE DIR...

The decoders are value.py's (run_path, branch8, certify); this file adds the
MBPP task with an execution checker, the checker-based selection, and the
bfloat16 loader for the 4B model. A run writes one JSON line per problem and
resumes after a cancelled job.
"""
import copy, glob, json, os, subprocess, sys, tempfile, time
import numpy as np
import value as V

CODE_MAX_NEW = int(os.environ.get("VALUE_MAX_NEW_CODE", 512))   # the amendment fixes 512; smaller only for smoke tests
CODE_METHODS = ["greedy", "prlm-0", "sample", "bo8-filter", "margin-branch-8-filter", "random-branch-8-filter"]
N_4B = 100


class HF16(V.HF):
    """value.HF with the weights in bfloat16 (Qwen3-4B's float32 copy does not fit a 16 GB runner)."""

    def __init__(self, repo, threads=4):
        import torch
        from huggingface_hub import snapshot_download, HfApi
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.set_num_threads(threads)
        self.torch = torch
        rev = HfApi().model_info(repo).sha
        d = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
        self.inputs = {"repo": repo, "commit": rev, "dtype": "bfloat16",
                       "weights_sha256": {os.path.relpath(p, d): V.sha256_file(p) for p in sorted(glob.glob(os.path.join(d, "*.safetensors")))}}
        self.tok = AutoTokenizer.from_pretrained(d)
        self.m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, attn_implementation="eager").eval()
        self.forwards = 0


# ---------------------------------------------------------------- MBPP
def mbpp_load(data):
    if data:
        return [json.loads(l) for l in open(data)]
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    repo = "google-research-datasets/mbpp"
    pt = hf_hub_download(repo, "sanitized/test-00000-of-00001.parquet", repo_type="dataset")
    pp = hf_hub_download(repo, "sanitized/prompt-00000-of-00001.parquet", repo_type="dataset")
    test, shots = pq.read_table(pt).to_pylist(), pq.read_table(pp).to_pylist()
    assert len(test) == 257, len(test)
    sha = {"test": V.sha256_file(pt), "prompt": V.sha256_file(pp)}
    return [{"index": k, "task_id": r["task_id"], "prompt": r["prompt"], "test_list": list(r["test_list"]),
             "test_imports": list(r["test_imports"] or []), "shots": shots[:3], "data_sha256": sha} for k, r in enumerate(test)]


def block(prompt, test, code=None):
    s = f"You are an expert Python programmer, and here is your task: {prompt} Your code should pass this test:\n\n{test}\n[BEGIN]\n"
    return s + (f"{code}\n[DONE]\n\n" if code is not None else "")


def program(text):
    i = text.find("[DONE]")
    return (text[:i] if i >= 0 else text).strip()


def runs(code, imports, tests, timeout=10):
    src = "\n".join(imports) + "\n" + code + "\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.py")
        open(f, "w").write(src)
        try:
            return subprocess.run([sys.executable, "-I", f], cwd=d, capture_output=True, timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def code_finish(lm, p):
    p.text = lm.decode(p.tokens)
    p.answer = program(p.text)
    p.ans_tokens = len(p.tokens)   # every position before the stop is eligible


def solve_mbpp(lm, pr):
    V.STOPS = ("[DONE]",)
    V.MAX_NEW = CODE_MAX_NEW
    V.finish = code_finish
    prompt = "".join(block(s["prompt"], s["test_list"][0], s["code"]) for s in pr["shots"]) + block(pr["prompt"], pr["test_list"][0])
    ids = lm.encode(prompt)
    state = lm.extend(None, ids)
    vis, allt, imp, i = pr["test_list"][:1], pr["test_list"], pr["test_imports"], pr["index"]
    out = {}
    for name in CODE_METHODS:
        f0, t0 = lm.forwards, time.perf_counter()
        points = None
        if name == "greedy":
            p, g = V.run_path(lm, state, [], [], "greedy"); paths = [p]
        elif name == "prlm-0":
            p, g = V.run_path(lm, state, [], [], "prlm"); paths = [p]
        elif name == "sample":
            p, g = V.run_path(lm, state, [], [], "sample", np.random.default_rng(i)); paths = [p]
        elif name == "bo8-filter":
            paths, g = [], 0
            for j in range(V.PATHS):
                p, gj = V.run_path(lm, state, [], [], "sample", np.random.default_rng(1000 * i + j))
                paths.append(p); g += gj
        else:
            paths, g, points = V.branch8(lm, state, i, "margin" if name.startswith("margin") else "random")
        progs = [p.answer for p in paths]
        visible = [runs(c, imp, vis) for c in progs] if len(paths) > 1 else None
        sel = next((k for k, v in enumerate(visible) if v), 0) if visible else 0
        full = [runs(c, imp, allt) for c in progs]
        out[name] = {"selected": sel, "pass": full[sel], "visible_pass_selected": runs(progs[sel], imp, vis),
                     "oracle": any(full), "visible": visible, "full": full, "programs": progs,
                     "paths": [p.tokens for p in paths], "gen_tokens": g, "forwards": lm.forwards - f0,
                     "seconds": round(time.perf_counter() - t0, 2)}
        if points is not None:
            out[name]["branch_points"] = points
    return {"index": i, "task_id": pr["task_id"], "max_new": CODE_MAX_NEW, "prompt_tokens": len(ids), "methods": out}


# ---------------------------------------------------------------- commands
def run(kind, model, outdir, shard, nshards, data, threads):
    os.makedirs(outdir, exist_ok=True)
    if kind == "mbpp":
        probs = [p for k, p in enumerate(mbpp_load(data)) if k % nshards == shard]
        lm = V.GPT2Local(threads) if model == "gpt2-local" else V.HF(model, threads)
    else:
        probs = [p for k, p in enumerate(V.load_test(data)[:N_4B]) if k % nshards == shard]
        lm = V.GPT2Local(threads) if model == "gpt2-local" else HF16(model, threads)
    path = os.path.join(outdir, f"{kind}-{model.replace('/', '_')}-shard{shard}.jsonl")
    done = {json.loads(l)["index"] for l in open(path) if l.strip()} if os.path.exists(path) else set()
    info = {"task": kind, "machine": V.machine(), "inputs": lm.inputs}
    for pr in probs:
        if pr["index"] in done:
            continue
        r = solve_mbpp(lm, pr) if kind == "mbpp" else V.solve(lm, pr["question"], pr["gold"], pr["index"])
        r.update(info)
        r["data_sha256"] = pr.get("data_sha256")
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        ms = CODE_METHODS if kind == "mbpp" else V.METHODS
        key = "pass" if kind == "mbpp" else "correct"
        print(pr["index"], {m: int(r["methods"][m][key]) for m in ms}, flush=True)


def summary(outfile, dirs):
    rows = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    rows.setdefault((r.get("task", "gsm8k"), r["inputs"]["repo"]), {})[r["index"]] = r
    out = {}
    for (task, repo), R in sorted(rows.items()):
        idx = sorted(R)
        if task == "mbpp":
            ms, key, b, s, rnd = CODE_METHODS, "pass", "margin-branch-8-filter", "bo8-filter", "random-branch-8-filter"
        else:
            ms, key, b, s, rnd = V.METHODS, "correct", "margin-branch-8", "sc-8", "random-branch-8"
        C = {m: [R[i]["methods"][m][key] for i in idx] for m in ms}
        res = {"task": task, "problems": len(idx), "inputs": sorted({json.dumps(R[i]["inputs"], sort_keys=True) for i in idx}),
               "accuracy": {m: round(sum(C[m]) / len(idx), 4) for m in ms},
               "oracle_best_of_8": {m: round(sum(R[i]["methods"][m]["oracle"] for i in idx) / len(idx), 4) for m in (s, b, rnd)},
               "mean_gen_tokens": {m: round(sum(R[i]["methods"][m]["gen_tokens"] for i in idx) / len(idx), 1) for m in ms},
               "mean_forwards": {m: round(sum(R[i]["methods"][m]["forwards"] for i in idx) / len(idx), 1) for m in ms},
               "branch_vs_sampling": V.verdict(C[b], C[s]), "branch_vs_greedy": V.verdict(C[b], C["greedy"]),
               "branch_vs_random": V.verdict(C[b], C[rnd])}
        if task == "mbpp":
            res["visible_pass_hidden_fail"] = {m: sum(R[i]["methods"][m]["visible_pass_selected"] and not R[i]["methods"][m]["pass"] for i in idx)
                                               for m in ms}
        out[f"{task}:{repo}"] = res
    json.dump(out, open(outfile, "w"), indent=1)
    print(json.dumps(out))


def main():
    a = sys.argv[1:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    if a[0] == "mbpp":
        run("mbpp", a[1], a[2], int(opt("--shard", 0)), int(opt("--nshards", 1)), opt("--data"), int(opt("--threads", 4)))
    elif a[0] == "gsm4b":
        model = opt("--model", "Qwen/Qwen3-4B-Base")
        run("gsm4b", model, a[1], int(opt("--shard", 0)), int(opt("--nshards", 1)), opt("--data"), int(opt("--threads", 4)))
    elif a[0] == "summary":
        summary(a[1], a[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
