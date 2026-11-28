"""Value amendment 9 (tasks/VALUE_AMEND_9.json, record 4998): program-aided certified decoding.

  python value_pal.py run MODEL OUTDIR --shard K --nshards N        float32 HF, the 400 TEST problems
  python value_pal.py summary OUTFILE PALDIR COTDIR                 COTDIR: regenerated rows of VALUE_AMEND_5

Each problem is written as 'Q: <question>' and a Python function solution(); the model
continues the function. Programs run once, sandboxed (fresh temporary directory, python -I,
empty environment, 5 s CPU and wall limit, 1 GB address space); the answer is
float(solution()). The decoders are value.py's, with the end-of-document stop of
VALUE_AMEND_5 and 'Q:' as the stop.
"""
import glob, json, os, resource, subprocess, sys, tempfile
import value as V

PAL_EXEMPLARS = [
    "    trees_initial = 15\n    trees_after = 21\n    trees_added = trees_after - trees_initial\n    return trees_added",
    "    cars_initial = 3\n    cars_arrived = 2\n    return cars_initial + cars_arrived",
    "    leah = 32\n    sister = 42\n    eaten = 35\n    return leah + sister - eaten",
    "    jason_initial = 20\n    jason_now = 12\n    return jason_initial - jason_now",
    "    toys = 5\n    from_mom = 2\n    from_dad = 2\n    return toys + from_mom + from_dad",
    "    computers_initial = 9\n    per_day = 5\n    days = 4\n    return computers_initial + per_day * days",
    "    balls_initial = 58\n    lost_tuesday = 23\n    lost_wednesday = 2\n    return balls_initial - lost_tuesday - lost_wednesday",
    "    money = 23\n    bagels = 5\n    price = 3\n    return money - bagels * price",
]
HEAD = "def solution():\n"
PAL_PREFIX = "".join(f"Q: {q}\n\n{HEAD}{c}\n\n\n" for (q, _), c in zip(V.EXEMPLARS, PAL_EXEMPLARS))
PAL_PATHS = 8
REPLAY = 5


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))


def execute(body):
    """float(solution()) of HEAD + body, or None."""
    src = HEAD + body + "\n\n_r = solution()\nprint(repr(float(_r)))\n"
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "p.py")
        open(f, "w").write(src)
        try:
            r = subprocess.run([sys.executable, "-I", f], cwd=d, env={}, capture_output=True, timeout=5,
                               preexec_fn=_limits, text=True)
        except subprocess.TimeoutExpired:
            return None
    if r.returncode != 0:
        return None
    try:
        x = float(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def body_of(text):
    i = text.find("\nQ:")
    return (text[:i] if i >= 0 else text).rstrip()


def pal_finish(lm, p):
    p.text = lm.decode(p.tokens)
    p.program = body_of(p.text)
    p.answer = execute(p.program)
    p.ans_tokens = len(p.tokens)


def run(model, outdir, shard, nshards, threads=4):
    import numpy as np
    import value_eos_run as E                 # end-of-document stop (patches V.run_path)
    V.STOPS = ("\nQ:",)
    V.finish = pal_finish
    os.makedirs(outdir, exist_ok=True)
    probs = [p for k, p in enumerate(V.load_test(None)) if k % nshards == shard]
    lm = E.patch(V.HF(model, threads))
    path = os.path.join(outdir, f"pal-{model.replace('/', '_')}-shard{shard}.jsonl")
    done = {json.loads(l)["index"] for l in open(path) if l.strip()} if os.path.exists(path) else set()
    info = {"task": "pal", "machine": V.machine(), "inputs": lm.inputs, "amendment": "VALUE_AMEND_9"}
    for n, pr in enumerate(probs):
        if pr["index"] in done:
            continue
        ids = lm.encode(PAL_PREFIX + f"Q: {pr['question']}\n\n{HEAD}")
        state = lm.extend(None, ids)
        out = {}
        for name in ("pal-greedy", "pal-prlm-0", "pal-8"):
            if name == "pal-8":
                paths, g = [], 0
                for j in range(PAL_PATHS):
                    q, gj = V.run_path(lm, state, [], [], "sample", np.random.default_rng(2000 * pr["index"] + j))
                    paths.append(q); g += gj
            else:
                q, g = V.run_path(lm, state, [], [], "greedy" if name == "pal-greedy" else "prlm")
                paths = [q]
            ans = [q.answer for q in paths]
            pred = V.vote(ans) if len(paths) > 1 else ans[0]
            out[name] = {"paths": [q.tokens for q in paths], "programs": [q.program for q in paths], "answers": ans,
                         "pred": pred, "correct": V.same(pred, pr["gold"]), "gen_tokens": g}
        r = {"index": pr["index"], "gold": pr["gold"], "methods": out, **info}
        if n < REPLAY:
            rep = {}
            for name, mode in (("pal-greedy", "greedy"), ("pal-prlm-0", "prlm")):
                q, _ = V.run_path(lm, lm.extend(None, ids), [], [], mode)
                rep[name] = q.tokens == out[name]["paths"][0]
            r["replay_identical"] = rep
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(pr["index"], {m: int(out[m]["correct"]) for m in out}, flush=True)


def load(d, task_ok):
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
        if "again" in os.path.abspath(p):
            continue
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                if task_ok(r.get("task", "gsm8k")):
                    rows[(r["inputs"]["repo"], r["index"])] = r
    return rows


def summary(outfile, paldir, cotdir):
    from value_lazy import cost
    pal = load(paldir, lambda t: t == "pal")
    cot = load(cotdir, lambda t: t == "gsm8k")
    out = {}
    for repo in sorted({k[0] for k in pal}):
        idx = sorted(i for (r_, i) in pal if r_ == repo and (repo, i) in cot)
        res = {k: [] for k in ("sc-8", "pal-8", "mixed-8", "union-16", "pal-prlm-0", "pal-greedy", "cot-greedy", "cot-prlm-0")}
        tok = {k: 0 for k in res}
        replay = [0, 0]
        for i in idx:
            P, C = pal[(repo, i)]["methods"], cot[(repo, i)]
            g = C["gold"]
            sc = C["methods"]["sc-8"]["answers"]
            pa = P["pal-8"]["answers"]
            sc_tok = sum(cost(C, "sc-8", k) for k in range(8))
            pal_tok = [len(t) for t in P["pal-8"]["paths"]]
            sc_toks = [cost(C, "sc-8", k) for k in range(8)]
            picks = {"sc-8": (V.vote(sc), sc_tok), "pal-8": (V.vote(pa), sum(pal_tok)),
                     "mixed-8": (V.vote(sc[:4] + pa[:4]), sum(sc_toks[:4]) + sum(pal_tok[:4])),
                     "union-16": (V.vote(sc + pa), sc_tok + sum(pal_tok)),
                     "pal-prlm-0": (P["pal-prlm-0"]["answers"][0], P["pal-prlm-0"]["gen_tokens"]),
                     "pal-greedy": (P["pal-greedy"]["answers"][0], P["pal-greedy"]["gen_tokens"]),
                     "cot-greedy": (C["methods"]["greedy"]["answers"][0], cost(C, "greedy", 0)),
                     "cot-prlm-0": (C["methods"]["prlm-0"]["answers"][0], cost(C, "prlm-0", 0))}
            for k, (a, t) in picks.items():
                res[k].append(V.same(a, g)); tok[k] += t
            rp = pal[(repo, i)].get("replay_identical")
            if rp is not None:
                replay[0] += 1
                replay[1] += int(all(rp.values()))
        n = len(idx)
        if not n:
            continue

        def vs(a, b):
            v = V.verdict(res[a], res[b])
            v["bar_3pt"] = v["verdict"] == "WIN" and v["diff"] >= 0.03
            v["token_ratio"] = round(tok[a] / tok[b], 3) if tok[b] else None
            v["better_on_both"] = v["verdict"] == "WIN" and tok[a] < tok[b]
            return v
        row = {"problems": n,
               "accuracy": {k: round(sum(v) / n, 4) for k, v in res.items()},
               "mean_gen_tokens": {k: round(t / n, 1) for k, t in tok.items()},
               "P1_pal8_vs_sc8": vs("pal-8", "sc-8"), "P2_mixed8_vs_sc8": vs("mixed-8", "sc-8"),
               "P3_union16_vs_sc8": vs("union-16", "sc-8"),
               "P4_palprlm_vs_cotgreedy": vs("pal-prlm-0", "cot-greedy"), "P4_palprlm_vs_cotprlm": vs("pal-prlm-0", "cot-prlm-0"),
               "P5_replay": {"problems": replay[0], "identical": replay[1], "holds": replay[0] > 0 and replay[0] == replay[1]},
               "executed_none_rate_pal8": round(sum(a is None for i in idx for a in pal[(repo, i)]["methods"]["pal-8"]["answers"]) / (8 * n), 4)}
        out[repo] = row
    p2 = [r["P2_mixed8_vs_sc8"] for r in out.values()]
    out["P2_primary_holds"] = any(v["bar_3pt"] for v in p2) and not any(v["verdict"] == "LOSS" for v in p2)
    json.dump(out, open(outfile, "w"), indent=1)
    print(json.dumps(out))


def main():
    a = sys.argv[1:]
    if a[0] == "summary":
        return summary(a[1], a[2], a[3])
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    run(a[1], a[2], int(opt("--shard", 0)), int(opt("--nshards", 1)), int(opt("--threads", 4)))


if __name__ == "__main__":
    main()
