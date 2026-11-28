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
CERT = ("cert_share", "cert_low")                          # amendment 11
CONF11 = CONF + CERT
ETAS = (0.1, 0.5, 0.9)
POWERS = (0.3, 1.0)
LEAST = {k: -1e9 for k in CONF11}


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


def rule_space(conf=CONF):
    aggs = [("maj",)] + [("w", c) for c in conf] + [("top", c, e) for c in conf for e in ETAS] + \
           [("borda", c, p) for c in conf for p in POWERS]
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


def uses(rule, feats):
    a = rule[1]
    return len(a) > 1 and a[1] in feats


def main(outfile, confdir, cotdir, paldir):
    conf = load(confdir, lambda r: (r["repo"], r["index"]) if r.get("amendment") == "VALUE_AMEND_10" else None)
    cot = load(cotdir, lambda r: (r["inputs"]["repo"], r["index"]) if r.get("task", "gsm8k") == "gsm8k" else None)
    pal = load(paldir, lambda r: (r["inputs"]["repo"], r["index"]) if r.get("task") == "pal" else None)
    keys = sorted(k for k in conf if k in cot and k in pal)
    space, space11 = rule_space(), rule_space(CONF11)
    rules = {name(r): r for r in space11}
    rules.update({f: r for f, r in FLOOR.items()})
    rules.update({f + "@16": r for f, r in FLOOR16.items()})
    res = {}                                   # key -> rule name -> (correct, tokens)
    for k in keys:
        tr = traces(cot[k], pal[k], conf[k])
        g = cot[k]["gold"]
        res[k] = {nm: (V.same(aggregate(tr[pool], a), g), sum(x[3] for x in tr[pool])) for nm, (pool, a) in rules.items()}
    repos = sorted({k[0] for k in keys})
    dev = [k for k in keys if k[1] % 2 == 0]
    test = {r: [k for k in keys if k[0] == r and k[1] % 2 == 1] for r in repos}
    alltest = [k for r in repos for k in test[r]]

    def pick(cands, on):
        sc = {nm: (sum(res[k][nm][0] for k in on), -sum(res[k][nm][1] for k in on), -i) for i, nm in enumerate(cands)}
        return max(cands, key=lambda nm: sc[nm]), sc

    acc = lambda nm, T: sum(res[k][nm][0] for k in T) / len(T)
    tok = lambda nm, T: sum(res[k][nm][1] for k in T) / len(T)
    floor = {}
    for r in repos:
        floor[(r, "8")] = max(FLOOR, key=lambda f: (acc(f, test[r]), -tok(f, test[r])))
        floor[(r, "16")] = max([f + "@16" for f in FLOOR16], key=lambda f: (acc(f, test[r]), -tok(f, test[r])))

    def against(sel, other=None, cls="8"):
        """sel vs other (default: the per-row best floor) on every row's TEST problems and pooled."""
        out, pa, pb = {}, [], []
        for r in repos:
            T = test[r]
            fl = other or floor[(r, cls)]
            a, b = [res[k][sel][0] for k in T], [res[k][fl][0] for k in T]
            v = V.verdict(a, b)
            v["bar_3pt"] = v["verdict"] == "WIN" and v["diff"] >= 0.03
            v["selected_accuracy"], v["other_accuracy"], v["other"] = round(acc(sel, T), 4), round(acc(fl, T), 4), fl
            v["token_ratio"] = round(tok(sel, T) / tok(fl, T), 3) if tok(fl, T) else None
            out[r] = v
            pa += a; pb += b
        out["pooled"] = V.verdict(pa, pb)
        return out

    out = {"problems": len(keys), "dev_problems": len(dev), "rows": {}, "selected": {}, "amend11": {}}
    for r in repos:
        out["rows"][r] = {"test_problems": len(test[r]),
                          "floor_rules_test": {f: round(acc(f, test[r]), 4) for f in FLOOR},
                          "floor16_rules_test": {f: round(acc(f + "@16", test[r]), 4) for f in FLOOR16},
                          "best_floor8": floor[(r, "8")], "best_floor16": floor[(r, "16")],
                          "S4_floor_vs_F0_all": {f: V.verdict([res[k][f][0] for k in keys if k[0] == r],
                                                              [res[k]["F0"][0] for k in keys if k[0] == r]) for f in FLOOR if f != "F0"}}
    # amendment 10: pre-registered space
    for cls in ("8", "16"):
        cands = [name(x) for x in space if CLASS[x[0]] == int(cls)]
        sel, sc = pick(cands, dev)
        out["selected"][cls] = sel
        out[f"dev_top10_{cls}"] = [(nm, sc[nm][0]) for nm in sorted(cands, key=lambda nm: sc[nm], reverse=True)[:10]]
        out[f"S_{cls}"] = against(sel, cls=cls)
        out[f"gamed_test_pick_{cls}"] = {"rule": pick(cands, alltest)[0], "vs_floor": against(pick(cands, alltest)[0], cls=cls)}
    rows8 = [out["S_8"][r] for r in repos]
    out["S1_primary"] = out["S_8"]["pooled"]["verdict"] == "WIN" and not any(v["verdict"] == "LOSS" for v in rows8)
    out["S2"] = out["S1_primary"] and any(v["bar_3pt"] for v in rows8)
    out["S3"] = out["S_16"]["pooled"]["verdict"] == "WIN"
    # amendment 11: certificate features added
    A = out["amend11"]
    c8 = [name(x) for x in space11 if CLASS[x[0]] == 8]
    A["selected_8"], sc = pick(c8, dev)
    A["dev_top10_8"] = [(nm, sc[nm][0]) for nm in sorted(c8, key=lambda nm: sc[nm], reverse=True)[:10]]
    A["vs_floor"] = against(A["selected_8"])
    rows = [A["vs_floor"][r] for r in repos]
    A["A1_primary"] = A["vs_floor"]["pooled"]["verdict"] == "WIN" and not any(v["verdict"] == "LOSS" for v in rows)
    cert = [name(x) for x in space11 if CLASS[x[0]] == 8 and uses(x, CERT)]
    publ = [name(x) for x in space11 if CLASS[x[0]] == 8 and uses(x, CONF)]
    A["best_cert_dev"], A["best_published_dev"] = pick(cert, dev)[0], pick(publ, dev)[0]
    A["cert_vs_published"] = against(A["best_cert_dev"], A["best_published_dev"])
    A["A2"] = A["cert_vs_published"]["pooled"]["verdict"] == "WIN"
    A["gamed_test_pick_8"] = {"rule": pick(c8, alltest)[0], "vs_floor": against(pick(c8, alltest)[0])}
    json.dump(out, open(outfile, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:5])
