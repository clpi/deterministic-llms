"""Value amendment 8 (tasks/VALUE_AMEND_8.json, record 4997): math vote rules against sc-8.

  python value_vote.py OUTFILE DIR...      DIRs hold the regenerated rows of VALUE_AMEND_5

One rule is selected on DEV (even indices of the 0.6B and 1.7B 400-problem rows, pooled) and
tested against sc-8 on TEST (odd indices of those rows, all of 4B and 14B). Every rule is
reported on every row. Tokens: each candidate's own generated tokens to its end of document
(value_lazy.cost); V8 pays for the prlm-0 family only on problems where the sc-8 vote is tied.
No model is run.
"""
import glob, json, os, sys
import value as V
import value_portfolio as P
from value_lazy import cost, eager_math

RULES = ("V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8")
SELECTABLE = ("V1", "V2", "V3", "V4", "V5", "V7", "V8")


def tied(answers):
    counts = {}
    for a in answers:
        if a is not None:
            counts[round(a, 6)] = counts.get(round(a, 6), 0) + 1
    if not counts:
        return set()
    top = max(counts.values())
    keys = [k for k, c in counts.items() if c == top]
    return set(keys) if len(keys) > 1 else set()


def checked(r, cands, dec):
    out = []
    for m, k in cands:
        a = r["methods"][m]["answers"][k]
        out.append(a if a is not None and P.equations_ok(dec(r["methods"][m]["paths"][k])) else None)
    return out


def filt(cands_ans, checked_ans):
    return checked_ans if any(a is not None for a in checked_ans) else cands_ans


def rules(r, dec, steps):
    S = [("sc-8", k) for k in range(8)]
    one = [("sample", 0)]
    fam = [("prlm-0", 0)] + [("margin-branch-8", k) for k in range(1, 8)]
    ans = lambda cs: [r["methods"][m]["answers"][k] for m, k in cs]
    tok = lambda cs: sum(cost(r, m, k) for m, k in cs)
    sa, oa, fa = ans(S), ans(one), ans(fam)
    sc, oc, fc = checked(r, S, dec), checked(r, one, dec), checked(r, fam, dec)
    fam_maj = V.vote(fa)
    fam_maj_c = V.vote(filt(fa, fc))
    out = {}
    out["V0"] = (V.vote(sa), tok(S))
    out["V1"] = (V.vote(filt(sa, sc)), tok(S))
    out["V2"] = (V.vote(sa + oa), tok(S + one))
    out["V3"] = (V.vote(filt(sa + oa, sc + oc)), tok(S + one))
    out["V4"] = (V.vote(sa + [fam_maj]), tok(S + fam))
    out["V5"] = (V.vote(filt(sa, sc) + [fam_maj_c]), tok(S + fam))
    out["V6"] = (eager_math(r, steps[2], dec), tok(steps[2]))
    w, first = {}, {}
    for i, (a, wt) in enumerate([(a, 1.0) for a in sa] + [(fa[0], 1.0)] + [(a, 1 / 8) for a in fa[1:]]):
        if a is None:
            continue
        key = round(a, 6)
        w[key] = w.get(key, 0.0) + wt
        first.setdefault(key, i)
    out["V7"] = (max(w, key=lambda k: (w[k], -first[k])) if w else None, tok(S + fam))
    t = tied(sa)
    if t:
        pick = next((x for x in (fam_maj, fa[0]) if x is not None and round(x, 6) in t), None)
        out["V8"] = (pick if pick is not None else V.vote(sa), tok(S + fam))
    else:
        out["V8"] = (V.vote(sa), tok(S))
    return {k: (V.same(a, r["gold"]), c) for k, (a, c) in out.items()}


def stats(res, idx):
    if not idx:
        return None
    return {k: {"accuracy": round(sum(res[i][k][0] for i in idx) / len(idx), 4),
                "mean_gen_tokens": round(sum(res[i][k][1] for i in idx) / len(idx), 1)} for k in RULES}


def main(outfile, dirs):
    rows = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    t = r.get("task", "gsm8k")
                    if t != "mbpp":
                        rows.setdefault((t, r["inputs"]["repo"]), {})[r["index"]] = r
    steps = P.ratchet_steps("gsm8k")
    per, out = {}, {"rows": {}}
    for (task, repo), R in sorted(rows.items()):
        dec = P.tokenizer(repo)
        per[(task, repo)] = {i: rules(R[i], dec, steps) for i in sorted(R)}
    dev_keys = [k for k in per if k[0] == "gsm8k"]
    dev = [(k, i) for k in dev_keys for i in per[k] if i % 2 == 0]
    dev_score = {r: (sum(per[k][i][r][0] for k, i in dev), -sum(per[k][i][r][1] for k, i in dev)) for r in SELECTABLE}
    chosen = max(SELECTABLE, key=lambda r: (dev_score[r][0], dev_score[r][1], -RULES.index(r)))
    out["dev_problems"] = len(dev)
    out["dev_correct"] = {r: dev_score[r][0] for r in SELECTABLE}
    out["dev_correct_references"] = {r: sum(per[k][i][r][0] for k, i in dev) for r in ("V0", "V6")}
    out["selected"] = chosen
    pooled_a, pooled_b = [], []
    for k, res in per.items():
        idx = sorted(res)
        test = [i for i in idx if i % 2 == 1] if k[0] == "gsm8k" else idx
        a = [res[i][chosen][0] for i in test]
        b = [res[i]["V0"][0] for i in test]
        pooled_a += a; pooled_b += b
        v = V.verdict(a, b)
        v["bar_3pt"] = v["verdict"] == "WIN" and v["diff"] >= 0.03
        ta = sum(res[i][chosen][1] for i in test) / len(test)
        tb = sum(res[i]["V0"][1] for i in test) / len(test)
        v["token_ratio"] = round(ta / tb, 3) if tb else None
        v["better_on_both"] = v["verdict"] == "WIN" and ta < tb
        out["rows"][f"{k[0]}:{k[1]}"] = {"problems": len(idx), "test_problems": len(test), "all": stats(res, idx),
                                         "dev": stats(res, [i for i in idx if i % 2 == 0]) if k[0] == "gsm8k" else None,
                                         "test": stats(res, test), "selected_vs_sc8_test": v}
    pv = V.verdict(pooled_a, pooled_b)
    pv["bar_3pt"] = pv["verdict"] == "WIN" and pv["diff"] >= 0.03
    out["pooled_test"] = {"problems": len(pooled_a), **pv}
    rv = [r["selected_vs_sc8_test"] for r in out["rows"].values()]
    out["M1"] = any(x["bar_3pt"] for x in rv) and not any(x["verdict"] == "LOSS" for x in rv)
    out["M2"] = pv["verdict"] == "WIN"
    json.dump(out, open(outfile, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
