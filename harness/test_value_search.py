"""Checks for value_search.py (amendment 10) on synthetic rows: F0 is the plain sc-8 vote,
top-eta keeps ceil(eta*N) of all traces, Borda follows (N-r+1)^p, the x2 weight needs two
formats, and the whole search runs and is deterministic. No model is run."""
import json, os, random, sys, tempfile
import value as V
import value_search as S

C = lambda x: {k: x for k in S.CONF}


def unit():
    tr = [(1.0, C(1), "cot", 5), (2.0, C(9), "cot", 5), (1.0, C(2), "cot", 5)]
    assert S.aggregate(tr, ("maj",)) == 1.0
    assert S.aggregate(tr, ("w", "mean")) == 2.0                     # 9 > 1 + 2
    assert S.aggregate(tr, ("top", "mean", 0.1)) == 2.0              # keeps ceil(0.3) = 1 trace
    assert S.aggregate(tr, ("borda", "mean", 1.0)) == 1.0            # 2.0 gets 3, 1.0 gets 1 + 2: tie, first index wins
    tr2 = [(None, C(99), "cot", 5), (3.0, C(1), "cot", 5), (4.0, C(2), "pal", 5), (3.0, C(1), "pal", 5)]
    assert S.aggregate(tr2, ("top", "mean", 0.1)) is None             # the top trace has no answer
    assert S.aggregate(tr2, ("maj",)) == 3.0
    assert S.aggregate(tr2, ("w", "mean", "x2")) == 3.0               # 3.0 in both formats: (1+1)*2 = 4 > 2
    assert S.aggregate([(None, None, "cot", 0)], ("maj",)) is None
    for _ in range(2000):                                            # majority equals V.vote
        a = [random.choice([None, 1.0, 2.0, 3.0]) for _ in range(8)]
        t = [(x, C(random.random()), "cot", 1) for x in a]
        assert S.aggregate(t, ("maj",)) == (None if V.vote(a) is None else round(V.vote(a), 6))


def synth(d):
    rng = random.Random(7)
    for sub in ("cot", "pal", "conf"):
        os.makedirs(os.path.join(d, sub))
    for repo in ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base"):
        cot, pal, conf = [], [], []
        for i in range(60):
            g = float(i)
            def ans(p):
                return g if rng.random() < p else float(rng.choice([i + 1, i + 2]))
            def path():
                return [rng.randrange(100, 200) for _ in range(rng.randrange(5, 15))]
            meth = {}
            for m, n in (("sc-8", 8), ("prlm-0", 1), ("greedy", 1), ("sample", 1), ("margin-branch-8", 8)):
                a = [ans(0.55) for _ in range(n)]
                meth[m] = {"answers": a, "paths": [path() for _ in range(n)]}
            meth["margin-branch-8"]["branch_points"] = [[0, 2]] * 7
            pa = [ans(0.5) for _ in range(8)]
            pm = {"pal-8": {"answers": pa, "paths": [path() for _ in range(8)]},
                  "pal-prlm-0": {"answers": [pa[0]], "paths": [path()]}, "pal-greedy": {"answers": [pa[0]], "paths": [path()]}}
            cot.append({"index": i, "gold": g, "methods": meth, "inputs": {"repo": repo}, "task": "gsm8k"})
            pal.append({"index": i, "gold": g, "methods": pm, "inputs": {"repo": repo}, "task": "pal"})
            cs = lambda A: [C(1.0 + (0.5 if a == g else 0.0) + rng.random()) for a in A]
            conf.append({"index": i, "repo": repo, "amendment": "VALUE_AMEND_10",
                         "cot": {m: cs(meth[m]["answers"]) for m in meth}, "pal": {m: cs(pm[m]["answers"]) for m in pm}})
        tag = repo.split("/")[1]
        for sub, rows in (("cot", cot), ("pal", pal), ("conf", conf)):
            with open(os.path.join(d, sub, f"{tag}.jsonl"), "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))


def run():
    with tempfile.TemporaryDirectory() as d:
        synth(d)
        outs = []
        for k in range(2):
            o = os.path.join(d, f"o{k}.json")
            S.main(o, os.path.join(d, "conf"), os.path.join(d, "cot"), os.path.join(d, "pal"))
            outs.append(open(o).read())
        assert outs[0] == outs[1], "not deterministic"
        r = json.loads(outs[0])
        assert r["problems"] == 120 and r["dev_problems"] == 60
        for row in r["rows"].values():
            assert row["test_problems"] == 30
        # with confidence planted on correct traces, a confidence rule must beat plain majority on DEV
        assert not r["selected"]["8"].endswith(":maj"), r["selected"]
        print("ok", r["selected"], r["S1_primary"])


if __name__ == "__main__":
    unit()
    run()
