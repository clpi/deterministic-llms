"""Value amendment 10 (tasks/VALUE_AMEND_10.json, record 4999): the state-of-the-art floor and
the decoder search.

  python value_search.py OUTFILE CONFDIR COTDIR PALDIR

Floor (8 generations): F0 self-consistency, F1-F4 DeepConf (Fu et al. 2025), F5 self-certainty
Borda (Kang et al. 2025), F6 program-aided self-consistency. The floor on a row is the best of
them on that row's TEST problems. The search enumerates pools x aggregators in the fixed order
of rule_space(), selects on DEV (even indices, 0.6B and 1.7B pooled) within each compute class,
and tests on TEST (odd indices). No model is run.
"""
import glob, json, math, os, sys
import value as V
from value_lazy import cost

CONF = ("mean", "bottom10", "lowest", "tail", "selfcert")
ETAS = (0.1, 0.5, 0.9)
POWERS = (0.3, 1.0)
LEAST = {k: -1e9 for k in CONF}


def load(d, pred):
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
        if "again" in os.path.abspath(p):
            continue
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                k = pred(r)
                if k:
                    rows[k] = r
    return rows


def traces(cot, pal, conf):
    """Each pool as a list of (answer, confidences, format, tokens), in a fixed order."""
    def T(row, conf_rec, m, k, fmt, tok):
        c = (conf_rec.get(m) or [None] * (k + 1))[k] if conf_rec else None
        return (row["methods"][m]["answers"][k], c, fmt, tok)
    cc, pc = conf.get("cot", {}), conf.get("pal", {})
    C8 = [T(cot, cc, "sc-8", k, "cot", cost(cot, "sc-8", k)) for k in range(8)]
    P8 = [T(pal, pc, "pal-8", k, "pal", len(pal["methods"]["pal-8"]["paths"][k])) for k in range(8)]
    FAM = [T(cot, cc, "prlm-0", 0, "cot", cost(cot, "prlm-0", 0))] + \
          [T(cot, cc, "margin-branch-8", k, "cot", cost(cot, "margin-branch-8", k)) for k in range(1, 8)]
    return {"cot8": C8, "pal8": P8, "mix8": C8[:4] + P8[:4], "fam8": FAM, "cot8+pal8": C8 + P8, "cot8+fam8": C8 + FAM}


CLASS = {"cot8": 8, "pal8": 8, "mix8": 8, "fam8": 8, "cot8+pal8": 16, "cot8+fam8": 16}


def aggregate(tr, agg):
    kind = agg[0]
    xf = agg[-1] == "x2"
    items = [(i, a, c, f) for i, (a, c, f, _) in enumerate(tr) if a is not None]
    if not items:
        return None
    if kind != "maj":
        cname = agg[1]
        if any(c is None for _, _, c, _ in items):      # a trace without a confidence (empty path) counts as least confident
            items = [(i, a, c if c is not None else LEAST, f) for i, a, c, f in items]
        val = lambda c: c[cname]
    if kind == "maj":
        w = [(i, a, 1.0, f) for i, a, _, f in items]
    elif kind == "w":
        w = [(i, a, val(c), f) for i, a, c, f in items]
    elif kind == "top":                                 # DeepConf: keep the top eta of all traces, then vote the answered ones
        keep = max(1, math.ceil(agg[2] * len(tr)))
        allc = [(i, c if c is not None else LEAST) for i, (_, c, _, _) in enumerate(tr)]
        kept = {i for i, _ in sorted(allc, key=lambda t: (-val(t[1]), t[0]))[:keep]}
        w = [(i, a, val(c), f) for i, a, c, f in items if i in kept]
        if not w:
            return None
    elif kind == "borda":
        ranked = sorted(items, key=lambda t: (-val(t[2]), t[0]))
        n = len(tr)
        w = sorted((i, a, (n - r) ** agg[2], f) for r, (i, a, c, f) in enumerate(ranked))
    score, first, fmts = {}, {}, {}
    for i, a, x, f in w:
        k = round(a, 6)
        score[k] = score.get(k, 0.0) + x
        first.setdefault(k, i)
    if xf:
        for _, a, _, f in items:
            fmts.setdefault(round(a, 6), set()).add(f)
        score = {k: v * (2.0 if len(fmts.get(k, ())) > 1 else 1.0) for k, v in score.items()}
    return max(score, key=lambda k: (score[k], -first[k]))


def rule_space():
    aggs = [("maj",)] + [("w", c) for c in CONF] + [("top", c, e) for c in CONF for e in ETAS] + \
           [("borda", c, p) for c in CONF for p in POWERS]
    out = []
    for pool in ("cot8", "pal8", "mix8", "fam8", "cot8+pal8", "cot8+fam8"):
        for a in aggs:
            out.append((pool, a))
            if pool in ("mix8", "cot8+pal8"):
                out.append((pool, a + ("x2",)))
    return out


FLOOR = {"F0": ("cot8", ("maj",)), "F1": ("cot8", ("w", "mean")), "F2": ("cot8", ("top", "bottom10", 0.9)),
         "F3": ("cot8", ("top", "lowest", 0.1)), "F4": ("cot8", ("top", "tail", 0.9)),
         "F5": ("cot8", ("borda", "selfcert", 0.3)), "F6": ("pal8", ("maj",))}
FLOOR16 = {k: ("cot8+pal8", a) for k, (_, a) in FLOOR.items() if k != "F6"}


def name(rule):
    pool, a = rule
    return pool + ":" + "/".join(str(x) for x in a)


def main(outfile, confdir, cotdir, paldir):
    conf = load(confdir, lambda r: (r["repo"], r["index"]) if r.get("amendment") == "VALUE_AMEND_10" else None)
    cot = load(cotdir, lambda r: (r["inputs"]["repo"], r["index"]) if r.get("task", "gsm8k") == "gsm8k" else None)
    pal = load(paldir, lambda r: (r["inputs"]["repo"], r["index"]) if r.get("task") == "pal" else None)
    keys = sorted(k for k in conf if k in cot and k in pal)
    space = rule_space()
    rules = {name(r): r for r in space}
    rules.update({f: r for f, r in FLOOR.items()})
    rules.update({f + "@16": r for f, r in FLOOR16.items()})
    res = {}                                   # key -> rule name -> (correct, tokens)
    for k in keys:
        tr = traces(cot[k], pal[k], conf[k])
        g = cot[k]["gold"]
        res[k] = {}
        for nm, (pool, a) in rules.items():
            t = tr[pool]
            res[k][nm] = (V.same(aggregate(t, a), g), sum(x[3] for x in t))
    repos = sorted({k[0] for k in keys})
    dev = [k for k in keys if k[1] % 2 == 0]
    test = {r: [k for k in keys if k[0] == r and k[1] % 2 == 1] for r in repos}
    out = {"problems": len(keys), "dev_problems": len(dev), "rows": {}, "selected": {}}
    for cls in (8, 16):
        cands = [name(r) for r in space if CLASS[r[0]] == cls]
        score = {nm: (sum(res[k][nm][0] for k in dev), -sum(res[k][nm][1] for k in dev), -i) for i, nm in enumerate(cands)}
        out["selected"][str(cls)] = max(cands, key=lambda nm: score[nm])
        out[f"dev_top10_{cls}"] = [(nm, score[nm][0]) for nm in sorted(cands, key=lambda nm: score[nm], reverse=True)[:10]]
    pooled = {"8": ([], []), "16": ([], [])}
    for r in repos:
        T = test[r]
        acc = lambda nm: sum(res[k][nm][0] for k in T) / len(T)
        tok = lambda nm: sum(res[k][nm][1] for k in T) / len(T)
        floor8 = max(FLOOR, key=lambda f: (acc(f), -tok(f)))
        floor16 = max(FLOOR16, key=lambda f: (acc(f + "@16"), -tok(f + "@16")))
        row = {"test_problems": len(T), "floor_rules_test": {f: round(acc(f), 4) for f in FLOOR},
               "floor16_rules_test": {f: round(acc(f + "@16"), 4) for f in FLOOR16},
               "best_floor8": floor8, "best_floor16": floor16 + "@16"}
        for cls, fl in (("8", floor8), ("16", floor16 + "@16")):
            sel = out["selected"][cls]
            a = [res[k][sel][0] for k in T]
            b = [res[k][fl][0] for k in T]
            v = V.verdict(a, b)
            v["bar_3pt"] = v["verdict"] == "WIN" and v["diff"] >= 0.03
            v["selected_accuracy"], v["floor_accuracy"] = round(acc(sel), 4), round(acc(fl), 4)
            v["token_ratio"] = round(tok(sel) / tok(fl), 3) if tok(fl) else None
            row[f"S_{cls}"] = v
            pooled[cls][0].extend(a); pooled[cls][1].extend(b)
        allk = [k for k in keys if k[0] == r]
        row["S4_floor_vs_F0_all"] = {f: V.verdict([res[k][f][0] for k in allk], [res[k]["F0"][0] for k in allk]) for f in FLOOR if f != "F0"}
        out["rows"][r] = row
    for cls in ("8", "16"):
        a, b = pooled[cls]
        out[f"pooled_S_{cls}"] = V.verdict(a, b) if a else None
    s8 = [out["rows"][r]["S_8"] for r in repos]
    out["S1_primary"] = bool(out["pooled_S_8"]) and out["pooled_S_8"]["verdict"] == "WIN" and not any(v["verdict"] == "LOSS" for v in s8)
    out["S2"] = out["S1_primary"] and any(v["bar_3pt"] for v in s8)
    s16 = out["pooled_S_16"]
    out["S3"] = bool(s16) and s16["verdict"] == "WIN"
    json.dump(out, open(outfile, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:5])
