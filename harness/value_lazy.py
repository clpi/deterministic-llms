"""Value amendment 6 (tasks/VALUE_AMEND_6.json, record 4995): the lazy ratchet.

  python value_lazy.py OUTFILE DIR...      DIRs hold rows already read with VALUE_AMEND_5's rule

Candidates are produced in the ratchet order and the decoder stops as soon as the
eager answer is settled (code: first candidate passing VISIBLE; math: the leading
checked answer cannot be overtaken by the candidates left). Simulated exactly from
the stored paths; theorems L1 and L2 are lean/Prlm/Lazy.lean. No model is run.
"""
import glob, json, os, sys
import value as V
import value_portfolio as P
from value_eos import starts, first_eos

EOS = {151643, 151645}


def cost(r, m, k):
    """Generated tokens of candidate (m, k), from its own start to its end of document."""
    M = r["methods"][m]
    p = M["paths"][k]
    s = starts(m, M)[k]
    e = first_eos(p, EOS)
    return max(0, (len(p) if e is None else e + 1) - s)


def lazy_code(r, order, tests):
    tok = 0
    for n, (m, k) in enumerate(order):
        tok += cost(r, m, k)
        if P.mbpp_visible(r, tests, m, k):
            return r["methods"][m]["full"][k], (m, k), tok, n + 1
    return r["methods"]["prlm-0"]["full"][0], ("prlm-0", 0), tok, len(order)


def eager_code(r, order, tests):
    for m, k in order:
        if P.mbpp_visible(r, tests, m, k):
            return r["methods"][m]["full"][k], (m, k)
    return r["methods"]["prlm-0"]["full"][0], ("prlm-0", 0)


def lazy_math(r, order, dec):
    tok, pool, counts = 0, [], {}
    for n, (m, k) in enumerate(order):
        tok += cost(r, m, k)
        a = r["methods"][m]["answers"][k]
        if a is not None and P.equations_ok(dec(r["methods"][m]["paths"][k])):
            pool.append(a)
            key = round(a, 6)
            counts[key] = counts.get(key, 0) + 1
        left = len(order) - n - 1
        if counts:
            top = sorted(counts.values(), reverse=True)
            lead, second = top[0], (top[1] if len(top) > 1 else 0)
            if lead > second + left:
                return V.vote(pool), tok, n + 1
    pred = V.vote(pool) if pool else r["methods"]["prlm-0"]["answers"][0]
    return pred, tok, len(order)


def eager_math(r, order, dec):
    ans = [r["methods"][m]["answers"][k] for m, k in order]
    ok = [a is not None and P.equations_ok(dec(r["methods"][m]["paths"][k])) for (m, k), a in zip(order, ans)]
    pool = [a for a, o in zip(ans, ok) if o]
    return V.vote(pool) if pool else r["methods"]["prlm-0"]["answers"][0]


def main(outfile, dirs):
    rows = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
            if "again" in os.path.abspath(p):
                continue
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    rows.setdefault((r.get("task", "gsm8k"), r["inputs"]["repo"]), {})[r["index"]] = r
    tests, out = None, {}
    for (task, repo), R in sorted(rows.items()):
        idx = sorted(R)
        code = task == "mbpp"
        if code and tests is None:
            import value_code as C
            tests = {q["index"]: q for q in C.mbpp_load(os.environ.get("VALUE_MBPP_DATA"))}
        dec = None if code else P.tokenizer(repo)     # code rows never decode (review: no tokenizer download for MBPP)
        key = "pass" if code else "correct"
        n = P.names(task)
        steps = P.ratchet_steps(task)
        bases = ("greedy", n["SAMPLES"], n["BRANCH"])
        base = {m: [R[i]["methods"][m][key] for i in idx] for m in bases}
        btok = {m: sum(R[i]["methods"][m]["gen_tokens"] for i in idx) / len(idx) for m in bases}
        res = {"task": task, "problems": len(idx),
               "baselines": {m: {"accuracy": round(sum(base[m]) / len(idx), 4), "mean_gen_tokens": round(btok[m], 1)} for m in bases}}
        variants = [("lazy-R4", steps[4])] if code else [("lazy-R2", steps[2]), ("lazy-R4", steps[4])]
        for name, order in variants:
            acc, toks, evals, same = [], [], [], 0
            for i in idx:
                r = R[i]
                if code:
                    ok, sel, t, e = lazy_code(r, order, tests)
                    eok, esel = eager_code(r, order, tests)
                    same += int(sel == esel)
                else:
                    pred, t, e = lazy_math(r, order, dec)
                    ep = eager_math(r, order, dec)
                    # exact vote keys (review: V.same's 1e-6 tolerance could merge two distinct keys)
                    same += int((pred is None and ep is None) or (pred is not None and ep is not None and round(pred, 6) == round(ep, 6)))
                    ok = V.same(pred, r["gold"])
                acc.append(bool(ok)); toks.append(t); evals.append(e)
            mt = sum(toks) / len(idx)
            eager_tok = sum(sum(cost(R[i], m, k) for m, k in order) for i in idx) / len(idx)
            row = {"accuracy": round(sum(acc) / len(idx), 4), "mean_gen_tokens": round(mt, 1),
                   "eager_mean_gen_tokens": round(eager_tok, 1), "mean_candidates": round(sum(evals) / len(idx), 2),
                   "candidates": len(order), "L3_equal_to_eager": same, "L3_holds": same == len(idx), "vs": {}}
            for m in bases:
                v = V.verdict(acc, base[m])
                v["bar_3pt"] = v["verdict"] == "WIN" and v["diff"] >= 0.03
                v["token_ratio"] = round(mt / btok[m], 3) if btok[m] else None
                v["better_on_both"] = v["verdict"] == "WIN" and mt < btok[m]
                row["vs"][m] = v
            res[name] = row
        out[f"{task}:{repo}"] = res
    json.dump(out, open(outfile, "w"), indent=1)
    print(json.dumps(out))
    bad = [k for k, r in out.items() for v in ("lazy-R2", "lazy-R4") if v in r and not r[v]["L3_holds"]]
    if bad:
        raise SystemExit(f"L3 violated (lazy selection differs from eager) on: {bad}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
