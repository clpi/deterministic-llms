"""Value amendment 10 (tasks/VALUE_AMEND_10.json, record 4999): trace confidences.

  python value_conf.py MODEL OUTDIR COTDIR PALDIR --shard K --nshards N

One teacher-forced float32 pass over each stored path's generated tokens (up to its first
end-of-document token), given the prompt it was generated from. Per token: the DeepConf
token confidence C_i = -mean of the top-20 log-probabilities, the self-certainty
-(1/V) sum_j log(V p_j), and the log-probability of the chosen token. Per path the summaries
the contract names: mean, bottom-10% group and lowest group confidence (overlapping windows
of 32 tokens), tail confidence (last 32 tokens), mean self-certainty, mean log-probability,
and the share of tokens that are the teacher-forced argmax (1.0 on a greedy path when the
pass is aligned with the stored generation; checked by `check`).

  python value_conf.py check CONFDIR       greedy paths must have argmax_rate >= 0.99
"""
import copy, glob, json, math, os, sys
import value as V
import value_pal as PAL

K_TOP = 20
WINDOW = 32
TAIL = 32
EOS = {151643, 151645}
QSCALE, B = 256, 4                                         # common.qscores and the xhw.py bound (amendment 11)


def windows(x, w=WINDOW):
    n = len(x)
    if n <= w:
        return [sum(x) / n]
    s = sum(x[:w])
    out = [s / w]
    for i in range(w, n):
        s += x[i] - x[i - w]
        out.append(s / w)
    return out


def summarize(C, SC, LP, AM=None, CERT=None):
    n = len(C)
    if n == 0:
        return None
    if n <= WINDOW:
        groups = [sum(C) / n]
    else:
        s = sum(C[:WINDOW])
        groups = [s / WINDOW]
        for i in range(WINDOW, n):
            s += C[i] - C[i - WINDOW]
            groups.append(s / WINDOW)
    g = sorted(groups)
    b = max(1, math.ceil(0.1 * len(g)))
    tail = C[-TAIL:]
    return {"tokens": n, "mean": sum(C) / n, "bottom10": sum(g[:b]) / b, "lowest": g[0],
            "tail": sum(tail) / len(tail), "selfcert": sum(SC) / n, "mean_lp": sum(LP) / n,
            "argmax_rate": None if AM is None else sum(AM) / n,
            "cert_share": None if CERT is None else sum(CERT) / n,
            "cert_low": None if CERT is None else min(windows([float(c) for c in CERT]))}


def cut(p):
    i = next((k for k, x in enumerate(p) if x in EOS), None)
    return p if i is None else p[:i]


def score_paths(lm, prompt_ids, paths):
    torch = lm.torch
    with torch.inference_mode():
        o = lm.m(input_ids=torch.tensor([prompt_ids]), use_cache=True)
        last, past = o.logits[0, -1:].float(), o.past_key_values
        out = []
        for p in paths:
            p = cut(p)
            if not p:
                out.append(None)
                continue
            try:
                q = lm.m(input_ids=torch.tensor([p]), past_key_values=copy.deepcopy(past), use_cache=True)
                logits = torch.cat([last, q.logits[0, :-1].float()], 0)
            except Exception:                              # a cache that cannot be copied: one full forward
                q = lm.m(input_ids=torch.tensor([list(prompt_ids) + p]), use_cache=False)
                logits = q.logits[0, len(prompt_ids) - 1:-1].float()
            lp = torch.log_softmax(logits.double(), -1)
            Vn = lp.shape[-1]
            C = (-lp.topk(K_TOP, -1).values.mean(-1)).tolist()
            SC = (-math.log(Vn) - lp.mean(-1)).tolist()
            LP = lp[torch.arange(len(p)), torch.tensor(p)].tolist()
            AM = (lp.argmax(-1) == torch.tensor(p)).tolist()      # alignment check: 1.0 on greedy paths
            q = torch.round(logits.double() * QSCALE)              # = np.rint, as common.qscores
            t2 = q.topk(2, -1).values
            CERT = ((t2[:, 0] - t2[:, 1]) > 2 * B).tolist()        # amendment 11: certified step
            out.append(summarize(C, SC, LP, AM, CERT))
    return out


def load(d, task):
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
        if "again" in os.path.abspath(p):
            continue
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                if r.get("task", "gsm8k") == task:
                    rows[(r["inputs"]["repo"], r["index"])] = r
    return rows


def run(model, outdir, cotdir, paldir, shard, nshards, threads=4):
    os.makedirs(outdir, exist_ok=True)
    probs = [p for k, p in enumerate(V.load_test(None)) if k % nshards == shard]
    cot, pal = load(cotdir, "gsm8k"), load(paldir, "pal")
    lm = V.HF(model, threads)
    path = os.path.join(outdir, f"conf-{model.replace('/', '_')}-shard{shard}.jsonl")
    done = {json.loads(l)["index"] for l in open(path) if l.strip()} if os.path.exists(path) else set()
    for pr in probs:
        i = pr["index"]
        if i in done:
            continue
        rec = {"index": i, "repo": model, "amendment": "VALUE_AMEND_10", "cot": {}, "pal": {}}
        c = cot.get((model, i))
        if c:
            ids = lm.encode(V.PREFIX + f"Q: {pr['question']}\nA:")
            for m in ("sc-8", "prlm-0", "greedy", "sample", "margin-branch-8"):
                if m in c["methods"]:
                    rec["cot"][m] = score_paths(lm, ids, c["methods"][m]["paths"])
        q = pal.get((model, i))
        if q:
            ids = lm.encode(PAL.PAL_PREFIX + f"Q: {pr['question']}\n\n{PAL.HEAD}")
            for m in ("pal-8", "pal-prlm-0", "pal-greedy"):
                rec["pal"][m] = score_paths(lm, ids, q["methods"][m]["paths"])
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(i, len(rec["cot"]), len(rec["pal"]), flush=True)


def check(confdir):
    rates = []
    for p in sorted(glob.glob(os.path.join(confdir, "**", "conf-*.jsonl"), recursive=True)):
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                for side, m in (("cot", "greedy"), ("pal", "pal-greedy")):
                    s = (r[side].get(m) or [None])[0]
                    if s:
                        rates.append(s["argmax_rate"])
    low = sum(x < 0.99 for x in rates)
    print(json.dumps({"greedy_paths": len(rates), "below_0.99": low, "min": min(rates) if rates else None}))
    if not rates or low > 0.01 * len(rates):
        raise SystemExit("teacher-forced pass is not aligned with the stored greedy paths")


def main():
    a = sys.argv[1:]
    if a[0] == "check":
        return check(a[1])
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    run(a[0], a[1], a[2], a[3], int(opt("--shard", 0)), int(opt("--nshards", 1)), int(opt("--threads", 4)))


if __name__ == "__main__":
    main()
