"""Output structure as a parametric, certified, reversible decision
(tasks/STRUCT_CONTRACT.json; proofs in lean/Prlm/Struct.lean).

  python3 structure.py CONTRACT_ID  (the contract names harness/struct.py; renamed before any run because it shadowed the standard struct module)
"""
import json, os, re, sys, time
import numpy as np
from common import ROOT, Log, Model, encode, decode, qscores, sha256_file

NAMES = ["essay", "song", "steps", "demo", "table"]
FTOK = [encode(" essay")[0], encode(" song")[0], encode(" list")[0], encode(" demo")[0], encode(" table")[0]]
LEX = [{"essay", "explain", "discuss", "analysis", "argue"},
       {"song", "poem", "lyrics", "verse", "rhyme"},
       {"steps", "how", "list", "guide", "checklist"},
       {"demo", "interactive", "script", "code", "build"},
       {"compare", "table", "versus", "vs", "chart"}]
UNIT = 64
DEV_TOPICS = ["volcanoes", "chess", "jazz", "photosynthesis", "bitcoin", "the moon", "coffee", "sailing", "origami",
              "black holes", "gardening", "the roman empire", "neural networks", "surfing", "bread baking", "glaciers",
              "electric cars", "opera", "bees", "climbing"]
TEST_TOPICS = ["earthquakes", "poker", "blues music", "fermentation", "ethereum", "mars", "tea", "rowing", "knitting",
               "quasars", "composting", "ancient egypt", "decision trees", "skateboarding", "pasta making", "deserts",
               "bicycles", "ballet", "ants", "hiking"]
DEV_T = [["write an essay about {X}", "explain {X} in depth", "discuss the history of {X}"],
         ["write a song about {X}", "a short poem on {X}", "lyrics about {X}"],
         ["steps to learn {X}", "how do I get into {X}", "a checklist for {X}"],
         ["build an interactive demo of {X}", "write a script that simulates {X}", "show me code for {X}"],
         ["compare {X} options in a table", "{X} versus the alternatives", "a chart of {X} facts"]]
TEST_T = [["give me an analysis of {X}", "argue for and against {X}", "a long-form piece on {X}"],
          ["a verse about {X}", "make a rhyme about {X}", "a ballad about {X}"],
          ["a guide to {X}", "list what I need for {X}", "walk me through starting {X}"],
          ["an interactive {X} explorer", "a demo of {X} I can click", "a playground for {X}"],
          ["a table of {X} facts", "compare kinds of {X}", "{X} side by side"]]
DEV_AMB = ["tell me about {X}", "{X}?", "what is {X}", "I want to understand {X}"]
TEST_AMB = ["talk to me about {X}", "{X}, please", "teach me {X}", "I'm curious about {X}"]
USERS = {"E": 0, "D": 3, "S": 1}
SEPS = {0: 628, 1: 198, 2: 198, 3: 198, 4: 198}
GRAMMAR = {0: (3, 32), 1: (4, 8), 2: (4, 12), 3: (6, 10), 4: (4, 12)}
CUE = {0: "Paragraph", 1: "Line", 2: "Step", 3: "Code line", 4: "Row"}


def requests(topics, tmpls, amb):
    out = []
    for f, ts in enumerate(tmpls):
        for ti, t in enumerate(ts):
            for x in topics:
                out.append({"r": t.format(X=x), "gold": f, "kind": "explicit", "tmpl": t, "topic": x})
    for t in amb:
        for x in topics:
            out.append({"r": t.format(X=x), "gold": None, "kind": "ambiguous", "tmpl": t, "topic": x})
    return out


def kw(r):
    w = set(re.findall(r"[a-z]+", r.lower()))
    return [len(w & L) for L in LEX]


def base(M, r):
    lg = M.logits(encode("Request: " + r + "\nThe best format for the answer is a"))
    q = qscores(lg)
    return [int(q[t]) for t in FTOK]


def scores(c, wkw, b):
    return [c["base"][f] + wkw * UNIT * c["kw"][f] + b[f] for f in range(5)]


def pick(s):
    v = max(range(5), key=lambda f: (s[f], -f))
    m = s[v] - max(s[f] for f in range(5) if f != v)
    return v, m


def need(p, sp, f, sf):
    return sf - sp + (1 if f < p else 0)


def dstar(p, s):
    return max([0] + [need(p, s[p], f, s[f]) for f in range(5) if f != p])


def gold(c, u):
    return c["gold"] if c["kind"] == "explicit" else USERS[u]


def acc(cs, wkw, b, u, kind):
    sel = [c for c in cs if c["kind"] == kind]
    ok = [pick(scores(c, wkw, b))[0] == gold(c, u) for c in sel]
    errs = {}
    for c, o in zip(sel, ok):
        if not o:
            errs[c["tmpl"]] = errs.get(c["tmpl"], 0) + 1
    return round(sum(ok) / len(sel), 4), sum(ok), len(sel), errs


def bias(p, d):
    b = [0] * 5
    b[p] = d
    return b


# ---------------------------------------------------------------- rendering
def gen_segment(M, prefix, n, banned):
    seg = []
    for _ in range(n):
        q = qscores(M.logits(prefix + seg))
        for t in banned:
            q[t] = np.iinfo(np.int64).min
        v = int(np.argmax(q))
        seg.append(v)
    return seg


def render_reply(M, r, f, mode, seg0_prompt=None, keep=None):
    n, L = GRAMMAR[f]
    sep = SEPS[f]
    banned = {198, 628, 50256}
    head = encode("Request: " + r + "\nFormat: " + NAMES[f] + "\n\n")
    segs = []
    calls0 = M.forwards
    for i in range(n):
        if keep is not None and i in keep:
            segs.append(keep[i])
            continue
        if mode == "chained":
            pre = head + sum((s + [sep] for s in segs), [])
            if i == 0 and seg0_prompt is not None:
                pre = encode("Request: " + seg0_prompt + "\nFormat: " + NAMES[f] + "\n\n")
        else:
            rr = seg0_prompt if (i == 0 and seg0_prompt is not None) else r
            pre = encode("Request: " + rr + "\nFormat: " + NAMES[f] + "\n\n" + CUE[f] + " " + str(i + 1) + ":")
        segs.append(gen_segment(M, pre, L, banned))
    toks = [FTOK[f]] + sum((s + [sep] for s in segs), [])
    return segs, toks, M.forwards - calls0


def parse(toks):
    f = FTOK.index(toks[0])
    sep = SEPS[f]
    segs, acc_ = [], []
    for t in toks[1:]:
        if t == sep:
            segs.append(acc_)
            acc_ = []
        else:
            acc_.append(t)
    return f, segs


def main(cid):
    L = Log()
    prem = [cid]
    t0 = time.perf_counter()
    M = Model(threads=2)
    dev = requests(DEV_TOPICS, DEV_T, DEV_AMB)
    test = requests(TEST_TOPICS, TEST_T, TEST_AMB)
    for c in dev + test:
        c["base"], c["kw"] = base(M, c["r"]), kw(c["r"])
    L.add("struct:declared", ["declared"], premises=prem,
          receipt={"struct_sha256": sha256_file(os.path.abspath(__file__)),
                   "struct_lean_sha256": sha256_file(os.path.join(ROOT, "lean", "Prlm", "Struct.lean")),
                   "format_tokens": FTOK, "dev": len(dev), "test": len(test), "score_seconds": round(time.perf_counter() - t0, 1),
                   "base_sha256": __import__("hashlib").sha256(json.dumps([c["base"] for c in dev + test]).encode()).hexdigest()})
    # fit w_kw on DEV explicit
    curve = {lv: acc(dev, lv, [0] * 5, "E", "explicit")[0] for lv in range(65)}
    best = max(curve.values())
    wkw = min(lv for lv, a in curve.items() if a == best)
    fit = {"w_kw_level": wkw, "dev_explicit_acc": best, "dev_explicit_acc_model_only": curve[0],
           "test_explicit_model_only": acc(test, 0, [0] * 5, "E", "explicit"),
           "test_explicit_fitted": acc(test, wkw, [0] * 5, "E", "explicit")}
    L.add("struct:fit", ["measured", "struct"], premises=prem, receipt=fit)
    print("fit", json.dumps(fit), flush=True)
    devA = [c for c in dev if c["kind"] == "ambiguous"]
    testA = [c for c in test if c["kind"] == "ambiguous"]
    users = {}
    for u, p in USERS.items():
        s0 = [scores(c, wkw, [0] * 5) for c in devA]
        D = max(dstar(p, s) for s in s0)
        won = lambda d: sum(pick(scores(c, wkw, bias(p, d)))[0] == p for c in devA)
        G = max(dstar(p, scores(c, wkw, [0] * 5)) for c in testA)
        R = {"structure": NAMES[p], "delta": D, "delta_logits": round(D / 256, 3),
             "dev_events_won_at_delta": f"{won(D)}/{len(devA)}",
             "dev_events_won_at_delta_minus_1": f"{won(D - 1)}/{len(devA)}" if D > 0 else "n/a"}
        for name, d in [("update", D), ("none", 0), ("under", max(D - 1, 0)), ("over", D + 4096), ("GAMED", G)]:
            R[name] = {"delta": d, "test_explicit": acc(test, wkw, bias(p, d), u, "explicit"),
                       "test_ambiguous": acc(test, wkw, bias(p, d), u, "ambiguous")}
        rng = np.random.default_rng(2)
        rp = rng.integers(0, 5, len(test))
        R["random"] = {"test_explicit": round(float(np.mean([rp[i] == gold(c, u) for i, c in enumerate(test) if c["kind"] == "explicit"])), 4),
                       "test_ambiguous": round(float(np.mean([rp[i] == gold(c, u) for i, c in enumerate(test) if c["kind"] == "ambiguous"])), 4)}
        stream = {}
        for k in (1, 2, 4, 8, 16, 40, 80):
            dk = max(dstar(p, s) for s in s0[:k])
            stream[k] = {"delta": dk, "test_explicit": acc(test, wkw, bias(p, dk), u, "explicit")[0],
                         "test_ambiguous": acc(test, wkw, bias(p, dk), u, "ambiguous")[0]}
        R["stream"] = stream
        # certified-stable decisions under the update (Core.stable_of_bound), checked by recompute
        stable = recheck = 0
        for c in test:
            v, m = pick(scores(c, wkw, [0] * 5))
            if 2 * D < m:
                stable += 1
                recheck += pick(scores(c, wkw, bias(p, D)))[0] == v
        R["certified_stable"] = {"skipped_recompute": stable, "of": len(test), "unchanged_on_recompute": recheck}
        users[u] = R
        L.add(f"struct:user:{u}", ["measured", "struct"], premises=prem, receipt=R)
        print(u, json.dumps({k: v for k, v in R.items() if k != "stream"}), flush=True)
    # rendering, reversibility and incremental resynthesis
    items = []
    for f in range(5):
        ex = [c for c in test if c["kind"] == "explicit" and c["gold"] == f]
        items += [(ex[(i * 23) % len(ex)], "E") for i in range(6)]
    for i, u in enumerate(["D"] * 4 + ["S"] * 3 + ["E"] * 3):
        items.append((testA[(i * 17) % len(testA)], u))
    rend = []
    for c, u in items:
        p = USERS[u]
        f = pick(scores(c, wkw, bias(p, users[u]["delta"])))[0]
        out = {"request": c["r"], "user": u, "structure": NAMES[f]}
        for mode in ("decomposed", "chained"):
            segs, toks, calls = render_reply(M, c["r"], f, mode)
            pf, ps = parse(toks)
            out[mode] = {"calls": calls, "round_trip": pf == f and ps == segs,
                         "text_retokenizes": encode(decode(toks[1:])) == toks[1:]}
            edited = c["r"].replace(c["topic"], c["topic"] + " for beginners")
            n = len(segs)
            keep = {i: segs[i] for i in range(1, n)} if mode == "decomposed" else None
            segs2, toks2, calls2 = render_reply(M, c["r"], f, mode, seg0_prompt=edited, keep=keep)
            if mode == "decomposed":
                fresh, _, _ = render_reply(M, c["r"], f, mode, seg0_prompt=edited)
                out[mode]["edit_calls"] = calls2
                out[mode]["edit_equals_full_rebuild"] = segs2 == fresh
                out[mode]["untouched_bitwise"] = all(segs2[i] == segs[i] for i in range(1, n))
            else:
                out[mode]["edit_calls"] = calls2
                out[mode]["later_segments_changed"] = sum(segs2[i] != segs[i] for i in range(1, n))
            if mode == "decomposed":
                out["text"] = decode(toks[1:])[:400]
        rend.append(out)
        print("render", c["r"], NAMES[f], out["decomposed"]["round_trip"], flush=True)
    summ = {"replies": len(rend), "round_trip": sum(r["decomposed"]["round_trip"] and r["chained"]["round_trip"] for r in rend),
            "text_retokenizes_decomposed": sum(r["decomposed"]["text_retokenizes"] for r in rend),
            "decomposed_edit_calls": sum(r["decomposed"]["edit_calls"] for r in rend),
            "chained_edit_calls": sum(r["chained"]["edit_calls"] for r in rend),
            "decomposed_untouched_bitwise": sum(r["decomposed"]["untouched_bitwise"] for r in rend),
            "decomposed_edit_equals_full_rebuild": sum(r["decomposed"]["edit_equals_full_rebuild"] for r in rend),
            "chained_later_segments_changed": sum(r["chained"]["later_segments_changed"] for r in rend),
            "structures": {n: sum(r["structure"] == n for r in rend) for n in NAMES}}
    L.add("struct:render", ["measured", "struct"], premises=prem, receipt={"summary": summ, "replies": rend})
    print("render", json.dumps(summ), flush=True)


if __name__ == "__main__" and sys.argv[1] not in ("gated", "dup"):
    main(int(sys.argv[1]))


# ---------------------------------------------------------------- amendment 1: gated bias axis
def gscores(c, wkw, p, d):
    s = [c["base"][f] + wkw * UNIT * c["kw"][f] for f in range(5)]
    if not any(c["kw"]):
        s[p] += d
    return s


def gacc(cs, wkw, p, d, u, kind):
    sel = [c for c in cs if c["kind"] == kind]
    ok = [pick(gscores(c, wkw, p, d))[0] == gold(c, u) for c in sel]
    errs = {}
    for c, o in zip(sel, ok):
        if not o:
            errs[c["tmpl"]] = errs.get(c["tmpl"], 0) + 1
    return round(sum(ok) / len(sel), 4), sum(ok), len(sel), errs


def gated(cid):
    L = Log()
    prem = [cid]
    M = Model(threads=2)
    dev = requests(DEV_TOPICS, DEV_T, DEV_AMB)
    test = requests(TEST_TOPICS, TEST_T, TEST_AMB)
    for c in dev + test:
        c["base"], c["kw"] = base(M, c["r"]), kw(c["r"])
    bh = __import__("hashlib").sha256(json.dumps([c["base"] for c in dev + test]).encode()).hexdigest()
    curve = {lv: acc(dev, lv, [0] * 5, "E", "explicit")[0] for lv in range(65)}
    wkw = min(lv for lv, a in curve.items() if a == max(curve.values()))
    devA = [c for c in dev if c["kind"] == "ambiguous"]
    testA = [c for c in test if c["kind"] == "ambiguous"]
    devX = [c for c in dev if c["kind"] == "explicit"]
    out = {"base_sha256": bh, "w_kw_level": wkw, "dev_explicit_lexicon_free": sum(not any(c["kw"]) for c in devX),
           "test_explicit_lexicon_free": sum(not any(c["kw"]) for c in test if c["kind"] == "explicit"), "users": {}}
    for u, p in USERS.items():
        D = max(dstar(p, gscores(c, wkw, p, 0)) for c in devA)
        G = max(dstar(p, gscores(c, wkw, p, 0)) for c in testA)
        won = lambda d: sum(pick(gscores(c, wkw, p, d))[0] == p for c in devA)
        R = {"structure": NAMES[p], "delta": D, "dev_events_won_at_delta": f"{won(D)}/{len(devA)}",
             "dev_events_won_at_delta_minus_1": f"{won(D - 1)}/{len(devA)}",
             "dev_explicit_unchanged": sum(pick(gscores(c, wkw, p, D))[0] == pick(gscores(c, wkw, p, 0))[0] for c in devX)}
        for name, d in [("update", D), ("none", 0), ("under", max(D - 1, 0)), ("over", D + 4096), ("GAMED", G)]:
            R[name] = {"delta": d, "test_explicit": gacc(test, wkw, p, d, u, "explicit"),
                       "test_ambiguous": gacc(test, wkw, p, d, u, "ambiguous")}
        out["users"][u] = R
        print(u, json.dumps(R), flush=True)
    L.add("struct:gated", ["measured", "struct"], premises=prem, receipt=out)


if __name__ == "__main__" and sys.argv[1] == "gated":
    gated(int(sys.argv[2]))


# ---------------------------------------------------------------- amendment 2: cost of decomposition
def dup_cost(cid):
    L = Log()
    M = Model(threads=2)
    test = requests(TEST_TOPICS, TEST_T, TEST_AMB)
    dev = requests(DEV_TOPICS, DEV_T, DEV_AMB)
    for c in dev + test:
        c["base"], c["kw"] = base(M, c["r"]), kw(c["r"])
    curve = {lv: acc(dev, lv, [0] * 5, "E", "explicit")[0] for lv in range(65)}
    wkw = min(lv for lv, a in curve.items() if a == max(curve.values()))
    devA = [c for c in dev if c["kind"] == "ambiguous"]
    testA = [c for c in test if c["kind"] == "ambiguous"]
    delta = {u: max(dstar(p, scores(c, wkw, [0] * 5)) for c in devA) for u, p in USERS.items()}
    items = []
    for f in range(5):
        ex = [c for c in test if c["kind"] == "explicit" and c["gold"] == f]
        items += [(ex[(i * 23) % len(ex)], "E") for i in range(6)]
    for i, u in enumerate(["D"] * 4 + ["S"] * 3 + ["E"] * 3):
        items.append((testA[(i * 17) % len(testA)], u))
    res = {m: {"segments": 0, "duplicates": 0, "bigram_distinct": []} for m in ("decomposed", "chained")}
    for c, u in items:
        f = pick(scores(c, wkw, bias(USERS[u], delta[u])))[0]
        for m in res:
            segs, toks, _ = render_reply(M, c["r"], f, m)
            res[m]["segments"] += len(segs)
            res[m]["duplicates"] += sum(any(segs[i] == segs[j] for j in range(i)) for i in range(len(segs)))
            body = sum(segs, [])
            bg = list(zip(body, body[1:]))
            res[m]["bigram_distinct"].append(len(set(bg)) / max(1, len(bg)))
    for m in res:
        res[m]["bigram_distinct"] = round(float(np.mean(res[m]["bigram_distinct"])), 4)
    L.add("struct:dup", ["measured", "struct"], premises=[cid], receipt=res)
    print("dup", json.dumps(res), flush=True)


if __name__ == "__main__" and sys.argv[1] == "dup":
    dup_cost(int(sys.argv[2]))
