"""Value test (tasks/VALUE_CONTRACT.json, record 4989): does a deterministic
decoder that spends extra compute where the certified margin is smallest beat
greedy and seeded sampling on GSM8K accuracy at an equal number of paths?

  python value.py run MODEL OUTDIR --shard K --nshards N [--data JSONL] [--again]
  python value.py summary OUTFILE DIR...

MODEL is a Hugging Face repo id (CI) or "gpt2-local" (this repo's ONNX GPT-2,
for a smoke test of the mechanics only). A run writes one JSON line per
problem to OUTDIR/<model>-shard<K>.jsonl and skips problems already written,
so a cancelled job can be resumed.
"""
import copy, glob, hashlib, json, math, os, platform, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
QSCALE = 256
MAX_NEW = int(os.environ.get("VALUE_MAX_NEW", 256))   # the contract fixes 256; smaller only for smoke tests
PATHS = 8
TEMP = 0.7
SEED = 20261127
N_TEST = 400
METHODS = ["greedy", "prlm-0", "sample", "sc-8", "margin-branch-8", "random-branch-8"]

# The 8-shot chain-of-thought exemplars of Wei et al. (2022), as in
# lm-evaluation-harness gsm8k-cot.yaml (fewshot first_n, "Q: ...\nA: ...", "\n\n" between).
EXEMPLARS = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
]
PREFIX = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in EXEMPLARS)
STOPS = ("\nQ:", "\n\n")
ANS_RE = re.compile(r"The answer is (-?[0-9.,]+)")
NUM_RE = re.compile(r"-?[0-9][0-9,]*\.?[0-9]*")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- answers
def to_num(s):
    s = s.replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def truncate(text):
    cut = len(text)
    for s in STOPS:
        i = text.find(s)
        if i >= 0:
            cut = min(cut, i)
    return text[:cut]


def extract(text):
    """(answer as float or None, end offset of the answer in text or None)."""
    t = truncate(text)
    m = list(ANS_RE.finditer(t))
    if m:
        return to_num(m[-1].group(1)), m[-1].end()
    m = [x for x in NUM_RE.finditer(t) if to_num(x.group(0)) is not None]
    if m:
        return to_num(m[-1].group(0)), m[-1].end()
    return None, None


def same(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-6


def vote(answers):
    """Majority over extracted answers; ties to the earliest; None never wins over a number."""
    counts, first = {}, {}
    for i, a in enumerate(answers):
        if a is None:
            continue
        k = round(a, 6)
        counts[k] = counts.get(k, 0) + 1
        first.setdefault(k, i)
    if not counts:
        return None
    return max(counts, key=lambda k: (counts[k], -first[k]))


# ---------------------------------------------------------------- models
class HF:
    def __init__(self, repo, threads=4):
        import torch
        from huggingface_hub import snapshot_download, HfApi
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.set_num_threads(threads)
        self.torch = torch
        rev = HfApi().model_info(repo).sha
        d = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
        self.inputs = {"repo": repo, "commit": rev,
                       "weights_sha256": {os.path.relpath(p, d): sha256_file(p) for p in sorted(glob.glob(os.path.join(d, "*.safetensors")))}}
        self.tok = AutoTokenizer.from_pretrained(d)
        self.m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32, attn_implementation="eager").eval()
        self.forwards = 0

    def encode(self, text):
        return list(self.tok(text, add_special_tokens=False).input_ids)

    def decode(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=False)

    def fork(self, state):
        return copy.deepcopy(state)

    def extend(self, state, ids):
        """Logits after feeding ids on top of state (None = empty); state is not modified."""
        return self.feed(self.fork(state) if state is not None else None, ids)

    def feed(self, past, ids):
        """Like extend, but consumes past in place (for decoding a path we own)."""
        with self.torch.inference_mode():
            o = self.m(input_ids=self.torch.tensor([list(ids)]), past_key_values=past, use_cache=True)
        self.forwards += 1
        return o.logits[0, -1].float().numpy().copy(), o.past_key_values


class GPT2Local:
    """Smoke test only: the repo's ONNX GPT-2 recomputes the whole sequence each call."""

    def __init__(self, threads=4):
        sys.path.insert(0, HERE)
        import common
        self.c = common
        self.M = common.Model(threads)
        self.inputs = {"repo": "gpt2-local", "model_sha256": sha256_file(os.path.join(common.MODELS, common.MODEL_FILE))}
        self.forwards = 0

    def encode(self, text):
        return self.c.encode(text)

    def decode(self, ids):
        return self.c.decode(ids)

    def fork(self, state):
        return list(state)

    def extend(self, state, ids):
        seq = list(state or []) + list(ids)
        self.forwards += 1
        return self.M.logits(seq, use_cache=False), seq

    feed = extend


# ---------------------------------------------------------------- decoding
def certify(row):
    q = np.rint(row.astype(np.float64) * QSCALE).astype(np.int64)
    v = int(np.argmax(q))
    best = int(q[v])
    q[v] = np.iinfo(np.int64).min
    w = int(np.argmax(q))
    return v, w, best - int(q[w])


def sample(row, rng):
    lp = row.astype(np.float64) / TEMP
    lp -= lp.max()
    p = np.exp(lp)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


class Path:
    def __init__(self, tokens, recs, origin):
        self.tokens = tokens        # generated tokens
        self.recs = recs            # per position: (second-ranked token, certified margin)
        self.origin = origin        # (parent path, branch position) or None
        self.text = ""
        self.answer = None
        self.ans_tokens = None      # positions < ans_tokens come before the answer


def run_path(lm, start, prefix, prefix_recs, mode, rng=None, origin=None):
    """Decode from the prompt (start = its last logits and state) plus prefix,
    until a stop or MAX_NEW tokens in the path."""
    tokens, recs = list(prefix), list(prefix_recs)
    row0, state = start
    if prefix:
        row, past = lm.extend(state, prefix)
    else:
        row, past = row0, lm.fork(state)
    gen = 0
    while len(tokens) < MAX_NEW:
        if mode == "greedy":
            v = int(np.argmax(row))
            w, m = None, None
        elif mode == "prlm":
            v, w, m = certify(row)
        else:
            v = sample(row, rng)
            w, m = None, None
        tokens.append(v)
        recs.append((w, m))
        gen += 1
        text = lm.decode(tokens)
        if any(s in text for s in STOPS):
            break
        if len(tokens) >= MAX_NEW:
            break
        row, past = lm.feed(past, [v])
    p = Path(tokens, recs, origin)
    finish(lm, p)
    return p, gen


def finish(lm, p):
    p.text = lm.decode(p.tokens)
    p.answer, end = extract(p.text)
    p.ans_tokens = len(p.tokens)
    if end is not None:
        for t in range(1, len(p.tokens) + 1):
            if len(lm.decode(p.tokens[:t])) >= end:
                p.ans_tokens = t
                break


def eligible(paths, used):
    out = []
    for i, p in enumerate(paths):
        lo = p.origin[1] + 1 if p.origin else 0
        for t in range(lo, min(p.ans_tokens, len(p.tokens))):
            if (i, t) not in used and p.recs[t][0] is not None:
                out.append((i, t))
    return out


def branch8(lm, start, index, how):
    paths, gens, used = [], 0, set()
    p, g = run_path(lm, start, [], [], "prlm")
    paths.append(p); gens += g
    rng = np.random.default_rng(index)
    points = []
    while len(paths) < PATHS:
        cand = eligible(paths, used)
        if not cand:
            break
        if how == "margin":
            i, t = min(cand, key=lambda c: (paths[c[0]].recs[c[1]][1], c[0], c[1]))
        else:
            i, t = cand[int(rng.integers(len(cand)))]
        used.add((i, t))
        par = paths[i]
        second = par.recs[t][0]
        prefix = par.tokens[:t] + [second]
        prefix_recs = par.recs[:t] + [(None, None)]
        q, g = run_path(lm, start, prefix, prefix_recs, "prlm", origin=(i, t))
        paths.append(q); gens += g
        points.append([i, t, par.recs[t][1]])
    return paths, gens, points


def solve(lm, question, gold, index):
    prompt = PREFIX + f"Q: {question}\nA:"
    ids = lm.encode(prompt)
    f0 = lm.forwards
    state = lm.extend(None, ids)
    prefill = lm.forwards - f0
    out = {}
    for name in METHODS:
        f0, t0 = lm.forwards, time.perf_counter()
        points = None
        if name == "greedy":
            p, g = run_path(lm, state, [], [], "greedy"); paths = [p]
        elif name == "prlm-0":
            p, g = run_path(lm, state, [], [], "prlm"); paths = [p]
        elif name == "sample":
            p, g = run_path(lm, state, [], [], "sample", np.random.default_rng(index)); paths = [p]
        elif name == "sc-8":
            paths, g = [], 0
            for j in range(PATHS):
                p, gj = run_path(lm, state, [], [], "sample", np.random.default_rng(1000 * index + j))
                paths.append(p); g += gj
        else:
            paths, g, points = branch8(lm, state, index, "margin" if name.startswith("margin") else "random")
        answers = [p.answer for p in paths]
        pred = vote(answers) if len(paths) > 1 else answers[0]
        out[name] = {"pred": pred, "correct": same(pred, gold), "oracle": any(same(a, gold) for a in answers),
                     "answers": answers, "paths": [p.tokens for p in paths], "gen_tokens": g,
                     "forwards": lm.forwards - f0, "seconds": round(time.perf_counter() - t0, 2)}
        if points is not None:
            out[name]["branch_points"] = points
    return {"index": index, "gold": gold, "max_new": MAX_NEW, "prompt_tokens": len(ids), "prefill_forwards": prefill, "methods": out}


# ---------------------------------------------------------------- data and commands
def load_test(data):
    if data:
        return [json.loads(l) for l in open(data)]
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(p).to_pylist()
    assert len(t) == 1319, len(t)
    idx = sorted(int(i) for i in np.random.default_rng(SEED).permutation(1319)[:N_TEST])
    return [{"index": i, "question": t[i]["question"], "gold": to_num(t[i]["answer"].split("####")[-1].strip()),
             "data_sha256": sha256_file(p)} for i in idx]


def machine():
    cpu = platform.processor()
    try:
        if sys.platform.startswith("linux"):
            cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith(("model name", "Model")))
    except Exception:
        pass
    return {"platform": platform.platform(), "cpu": cpu, "python": platform.python_version()}


def cmd_run(model, outdir, shard, nshards, data, threads):
    os.makedirs(outdir, exist_ok=True)
    probs = [p for k, p in enumerate(load_test(data)) if k % nshards == shard]
    lm = GPT2Local(threads) if model == "gpt2-local" else HF(model, threads)
    path = os.path.join(outdir, f"{model.replace('/', '_')}-shard{shard}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["index"] for l in open(path) if l.strip()}
    info = {"machine": machine(), "inputs": lm.inputs}
    for pr in probs:
        if pr["index"] in done:
            continue
        r = solve(lm, pr["question"], pr["gold"], pr["index"])
        r.update(info)
        r["data_sha256"] = pr.get("data_sha256")
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(pr["index"], {m: int(r["methods"][m]["correct"]) for m in METHODS}, flush=True)


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans: (b better count, a better count, p)."""
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n10, n01, min(1.0, 2 * p)


def verdict(a, b):
    wa, wb, p = mcnemar(a, b)
    d = (sum(a) - sum(b)) / len(a)
    v = "WIN" if d > 0 and p < 0.05 else ("LOSS" if d < 0 and p < 0.05 else "TIE")
    return {"diff": round(d, 4), "a_only": wa, "b_only": wb, "p": round(p, 5), "verdict": v}


def cmd_summary(outfile, dirs):
    rows = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
            again = "again" in os.path.abspath(p)
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    rows.setdefault((r["inputs"]["repo"], again), {})[r["index"]] = r
    out = {}
    for (repo, again), R in sorted(rows.items()):
        if again:
            continue
        idx = sorted(R)
        res = {"problems": len(idx), "inputs": sorted({json.dumps(R[i]["inputs"], sort_keys=True) for i in idx}),
               "data_sha256": sorted({R[i].get("data_sha256") for i in idx} - {None})}
        C = {m: [R[i]["methods"][m]["correct"] for i in idx] for m in METHODS}
        res["accuracy"] = {m: round(sum(C[m]) / len(idx), 4) for m in METHODS}
        res["oracle_best_of_8"] = {m: round(sum(R[i]["methods"][m]["oracle"] for i in idx) / len(idx), 4)
                                   for m in ("sc-8", "margin-branch-8", "random-branch-8")}
        res["mean_gen_tokens"] = {m: round(sum(R[i]["methods"][m]["gen_tokens"] for i in idx) / len(idx), 1) for m in METHODS}
        res["mean_forwards"] = {m: round(sum(R[i]["methods"][m]["forwards"] for i in idx) / len(idx), 1) for m in METHODS}
        res["V1_branch_vs_sc8"] = verdict(C["margin-branch-8"], C["sc-8"])
        res["V2_branch_vs_greedy"] = verdict(C["margin-branch-8"], C["greedy"])
        res["V3_branch_vs_random"] = verdict(C["margin-branch-8"], C["random-branch-8"])
        res["prlm0_vs_greedy_same_answer"] = sum(R[i]["methods"]["prlm-0"]["pred"] == R[i]["methods"]["greedy"]["pred"] for i in idx)
        A = rows.get((repo, True), {})
        if A:
            res["V5_rerun"] = {"problems": len(A),
                               "branch_paths_identical": sum(A[i]["methods"]["margin-branch-8"]["paths"] == R[i]["methods"]["margin-branch-8"]["paths"] for i in A if i in R),
                               "sc8_paths_identical": sum(A[i]["methods"]["sc-8"]["paths"] == R[i]["methods"]["sc-8"]["paths"] for i in A if i in R)}
        out[repo] = res
    json.dump(out, open(outfile, "w"), indent=1)
    print(json.dumps(out))


def cmd_table(dirs):
    """One line per problem: index, gold, then 1/0 per method (order of METHODS)."""
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
            if "again" in os.path.abspath(p):
                continue
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    print("ROW", r["inputs"]["repo"], r["index"], r["gold"],
                          "".join(str(int(r["methods"][m]["correct"])) for m in METHODS),
                          json.dumps({m: r["methods"][m]["answers"] for m in ("sc-8", "margin-branch-8", "random-branch-8")}))


def main():
    a = sys.argv[1:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    if a[0] == "run":
        cmd_run(a[1], a[2], int(opt("--shard", 0)), int(opt("--nshards", 1)), opt("--data"), int(opt("--threads", 4)))
    elif a[0] == "summary":
        cmd_summary(a[1], a[2:])
    elif a[0] == "table":
        cmd_table(a[1:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
