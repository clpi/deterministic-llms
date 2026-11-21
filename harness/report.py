"""Rebuilds every reported number from out/log.jsonl alone (fails if the
hash chain is broken) and writes out/REPORT.md and out/front-*.txt."""
import json, os, re, subprocess, sys
from collections import defaultdict
import numpy as np
from common import ROOT, LOG, verify_log, decode
import axes as AX

n_rec, head = verify_log()
recs = [json.loads(l) for l in open(LOG)]
T = json.load(open(os.path.join(ROOT, "tasks", "tasks.json")))
items = {t["id"]: t for t in T["t1"] + T["t2"] + T["t3"]}
out = defaultdict(dict)  # (system) -> {(item, run): receipt}
for r in recs:
    if r["head"].startswith("out:"):
        x = r["receipt"]
        out[x["system"]][(x["item"], x["run"])] = x
other = {r["head"]: r["receipt"] for r in recs if not r["head"].startswith("out:")}
LEAN = "/tmp/claude-0/prlm"
lines = []
P = lines.append
rng = np.random.default_rng(0)


def boot(diff):
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0:
        return (0, 0, 0)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    return diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def verdict(d):
    m, lo, hi = d
    if lo > 0:
        return "higher"
    if hi < 0:
        return "lower"
    return "tie"


def words(ids):
    return [re.sub(r"[^a-z]", "", w.lower()) for w in decode(ids).split()]


def ngr(x, n):
    return [tuple(x[i:i + n]) for i in range(len(x) - n + 1)]


def t1m(g, it):
    ref = it["reference"]
    match = sum(int(a == b) for a, b in zip(g, ref))
    from collections import Counter
    cg, cr = Counter(g), Counter(ref)
    ov = sum((cg & cr).values())
    f1 = 0 if ov == 0 else 2 * ov / (len(g) + len(ref))
    tri = ngr(g, 3)
    rep3 = sum(1 for i, t in enumerate(tri) if t in tri[:i]) / max(1, len(tri))
    bi = ngr(g, 2)
    d2 = len(set(bi)) / max(1, len(bi))
    return {"ref_match": match, "ref_f1": f1, "rep3": rep3, "distinct2": d2}


def per_prompt(system, ids, fn, runs=None):
    """Mean of fn over runs, per prompt."""
    res = {}
    for i in ids:
        vals = [fn(x["gen"], items[i]) for (it, r), x in out[system].items()
                if it == i and (runs is None or r in runs)]
        if vals:
            res[i] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    return res


def ms_per_tok(system, prefix, runs=None):
    v = [x["wall"] * 1000 / max(1, len(x["gen"])) for (i, r), x in out[system].items()
         if i.startswith(prefix) and (runs is None or r in runs) and x.get("forwards", 1) > 0]
    return float(np.median(v)) if v else float("nan")


P(f"# PRLM report (rebuilt from the log)\n\nLog: {n_rec} records, chain head {head[:16]}.\n")

# ---------------------------------------------------------------- T1
try:
    t1ids = [t["id"] for t in T["t1"] if (t["id"], 0) in out["prlm-0"] and (t["id"], 0) in out["GAMED-oracle"]]
    SYS1 = ["samp-t1.0", "samp-t0.7", "samp-p0.9", "greedy-f32", "prlm-0", "prlm-theta", "ctl-null", "GAMED-oracle"]
    m1 = {s: per_prompt(s, t1ids, t1m, runs=None if s.startswith("samp") else [0]) for s in SYS1}
    P("## T1 continuation (35 prompts, 24 tokens)\n")
    P("| system | ref_match /24 | ref_f1 | rep3 | distinct2 | ms/token | calls/output |")
    P("| --- | --- | --- | --- | --- | --- | --- |")
    for s in SYS1:
        if not m1[s]:
            continue
        avg = {k: np.mean([m1[s][i][k] for i in m1[s]]) for k in ["ref_match", "ref_f1", "rep3", "distinct2"]}
        calls = np.mean([x["forwards"] for (i, r), x in out[s].items() if i.startswith("t1")])
        P(f"| {s} | {avg['ref_match']:.2f} | {avg['ref_f1']:.3f} | {avg['rep3']:.3f} | {avg['distinct2']:.3f} | {ms_per_tok(s, 't1', None if s.startswith('samp') else [0]):.1f} | {calls:.1f} |")
    P("\nPaired differences against prlm-0 (mean, 95% bootstrap interval over prompts):\n")
    for s in ["greedy-f32", "prlm-theta", "samp-t1.0", "samp-t0.7", "samp-p0.9"]:
        for k in ["ref_match", "ref_f1", "rep3", "distinct2"]:
            common = [i for i in t1ids if i in m1[s] and i in m1["prlm-0"]]
            d = boot([m1[s][i][k] - m1["prlm-0"][i][k] for i in common])
            P(f"- {s} − prlm-0, {k}: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] → {s} {verdict(d)}")
    same = sum(int(out["greedy-f32"][(i, 0)]["gen"] == out["prlm-0"][(i, 0)]["gen"]) for i in t1ids)
    cert = sum(x.get("certified", 0) for (i, r), x in out["prlm-0"].items() if i.startswith("t1") and r == 0)
    steps = sum(len(x["gen"]) for (i, r), x in out["prlm-0"].items() if i.startswith("t1") and r == 0)
    nullsame = sum(int(out["ctl-null"][(i, 0)]["gen"] == out["prlm-0"][(i, 0)]["gen"]) for i in t1ids)
    P(f"\nprlm-0 equals greedy-f32 on {same}/{len(t1ids)} prompts; {cert}/{steps} steps certified (rest accepted ties). Do-nothing control equals prlm-0 on {nullsame}/{len(t1ids)}.\n")


except Exception as e:
    P(f'\n(section T1 incomplete: {type(e).__name__} {e})\n')
# ---------------------------------------------------------------- T2
try:
    t2ids = [t["id"] for t in T["t2"] if (t["id"], 0) in out["GAMED-oracle"]]


    def t2acc(g, it):
        w = decode(g).strip().split()
        first = w[0].strip(".,;:!?\"'()") if w else ""
        return {"acc": float(first == it["answer"])}


    P("## T2 closed cloze (40 items)\n")
    P("| system | accuracy | per-seed range |")
    P("| --- | --- | --- |")
    for s in ["samp-t1.0", "samp-t0.7", "samp-p0.9", "greedy-f32", "prlm-0", "ctl-null", "GAMED-oracle", "alt-beam4", "alt-bestfirst", "alt-ngram", "alt-rules"]:
        if s not in out:
            continue
        pp = per_prompt(s, t2ids, t2acc)
        if not pp:
            continue
        acc = np.mean([pp[i]["acc"] for i in pp])
        rngs = ""
        if s.startswith("samp"):
            seeds = [np.mean([t2acc(out[s][(i, r)]["gen"], items[i])["acc"] for i in t2ids]) for r in range(5)]
            rngs = f"{min(seeds):.3f} to {max(seeds):.3f}"
        P(f"| {s} | {acc:.3f} | {rngs} |")
    d = boot([t2acc(out['greedy-f32'][(i,0)]['gen'], items[i])['acc'] - np.mean([t2acc(out['samp-p0.9'][(i,r)]['gen'], items[i])['acc'] for r in range(5)]) for i in t2ids])
    P(f"\ngreedy-f32 − samp-p0.9 accuracy: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] → {verdict(d)}\n")


except Exception as e:
    P(f'\n(section T2 incomplete: {type(e).__name__} {e})\n')
# ---------------------------------------------------------------- T3
try:
    P("## T3 control (20 prompts per kind, 24 tokens)\n")


    def t3ok(g, it):
        k = it["kind"]
        if k == "ban":
            return float(not any(w in set(it["words"]) for w in words(g)))
        if k == "topic":
            return float(any(w in set(it["train_words"]) for w in words(g)))
        if k == "end":
            return float(any(any(ch in decode([t]) for ch in ".!?") for t in g[:it["within"]]))


    def t3test(g, it):
        return float(any(w in set(it["test_words"]) for w in words(g)))


    P("| kind | system | success | held-out hit (topic) | NLL/token under base |")
    P("| --- | --- | --- | --- | --- |")
    for kind in ["ban", "topic", "end"]:
        ids = [t["id"] for t in T["t3"] if t["kind"] == kind]
        syss = ["samp-p0.9", "greedy-f32", "prlm-0", "ctl-null"] + sorted(s for s in out if s.startswith("prlm-theta:")) + ["GAMED-oracle"]
        for s in syss:
            rows = [(i, r, x) for (i, r), x in out[s].items() if i in ids]
            if not rows:
                continue
            ok = np.mean([t3ok(x["gen"], items[i]) for i, r, x in rows])
            ho = np.mean([t3test(x["gen"], items[i]) for i, r, x in rows]) if kind == "topic" else float("nan")
            nl = np.mean([x["nll"] for i, r, x in rows])
            tag = " (CORRECTED: sign bug, boosted banned words)" if s in ("prlm-theta:ban", "prlm-theta:ban-soft") else ""
            P(f"| {kind} | {s}{tag} | {ok:.2f} | {'' if kind != 'topic' else f'{ho:.2f}'} | {nl:.2f} |")
        # best of 5 sampling with a checker
        tries, succ = [], 0
        for i in ids:
            k = next((r + 1 for r in range(5) if t3ok(out["samp-p0.9"][(i, r)]["gen"], items[i])), None)
            succ += k is not None
            tries.append(k or 5)
        P(f"| {kind} | samp-p0.9 best-of-5 with checker | {succ/len(ids):.2f} |  | mean tries {np.mean(tries):.2f} (×24 calls) |")
    P("")


except Exception as e:
    P(f'\n(section T3 incomplete: {type(e).__name__} {e})\n')
# ---------------------------------------------------------------- adaptive
try:
    for c in T["corpus"]:
        if len(c["ids"]) >= 56:
            items[f"dev/{c['module']}"] = {"id": f"dev/{c['module']}", "prompt": c["ids"][:32], "reference": c["ids"][32:56]}
    devids = sorted({i for s in out if s.startswith("dev:") for (i, r) in out[s]})
    if devids:
        P("## On-the-fly weighting (post hoc; chosen on DEV, every candidate scored on TEST)\n")
        P("| candidate | DEV ref_match | DEV rep3 | TEST ref_match | TEST rep3 | TEST distinct2 |")
        P("| --- | --- | --- | --- | --- | --- |")
        names = ["greedy"] + sorted(s[4:] for s in out if s.startswith("dev:") and s != "dev:greedy")
        for nm in names:
            dv = per_prompt("dev:" + nm, devids, t1m)
            ts = per_prompt("greedy-f32" if nm == "greedy" else "adapt:" + nm, t1ids, t1m, runs=[0])
            f = lambda d, k: np.mean([d[i][k] for i in d]) if d else float("nan")
            P(f"| {nm} | {f(dv,'ref_match'):.2f} | {f(dv,'rep3'):.3f} | {f(ts,'ref_match'):.2f} | {f(ts,'rep3'):.3f} | {f(ts,'distinct2'):.3f} |")
        for nm in names[1:]:
            ts = per_prompt("adapt:" + nm, t1ids, t1m, runs=[0])
            for k in ["ref_match", "rep3"]:
                common = [i for i in t1ids if i in ts and i in m1["prlm-0"]]
                d = boot([ts[i][k] - m1["prlm-0"][i][k] for i in common])
                P(f"- TEST {nm} − prlm-0, {k}: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] → {verdict(d)}")
        P("")
except Exception as e:
    P(f"\n(section adaptive incomplete: {type(e).__name__} {e})\n")
# ---------------------------------------------------------------- reproducibility
try:
    P("## Reproducibility (T1)\n")


    def distinct(system, runs):
        c = []
        for i in t1ids:
            gs = {tuple(out[system][(i, r)]["gen"]) for r in runs if (i, r) in out[system]}
            if gs:
                c.append(len(gs))
        return np.mean(c) if c else float("nan"), max(c) if c else 0


    for s, runs, note in [("prlm-0", [0, 1, 2, 3, 4], "runs 1 to 3 from an empty cache, run 4 at 1 thread"),
                          ("greedy-f32", [0, 1, 2, 3], "runs 1 to 3 from an empty cache"),
                          ("samp-p0.9", [0, 1, 2, 3, 4], "seeds 0 to 4 (unseeded serving)"),
                          ("samp-p0.9-seed0-rerun", [1, 2, 3], "seed 0 rerun three times")]:
        m, mx = distinct(s, runs)
        P(f"- {s}: mean distinct outputs per prompt {m:.2f}, max {mx} ({note})")
    if "repro:logits-threads" in other:
        x = other["repro:logits-threads"]
        P(f"- logits bitwise equal at 1 and 4 threads: {x['bitwise_equal']}/{x['prompts']} prompts")
    s0 = sum(int(out["samp-p0.9"].get((i, 0), {}).get("gen") == out["samp-p0.9-seed0-rerun"].get((i, 1), {}).get("gen")) for i in t1ids)
    P(f"- seeded sampling seed 0 equals its rerun on {s0}/{len(t1ids)} prompts\n")


except Exception as e:
    P(f'\n(section reproducibility incomplete: {type(e).__name__} {e})\n')
# ---------------------------------------------------------------- alternatives
try:
    P("## Alternative constructions\n")
    P("| system | items | certified | mean calls | mean NLL of output (beam/best-first) | equals greedy |")
    P("| --- | --- | --- | --- | --- | --- |")
    for s in ["alt-beam4", "alt-bestfirst"]:
        rows = list(out[s].items())
        if not rows:
            continue
        c = sum(int(bool(x.get("certified"))) for k, x in rows)
        calls = np.mean([x["forwards"] for k, x in rows])
        eq = sum(int(x["gen"] == out["alt-greedy-short"][k]["gen"]) for k, x in rows)
        nll = [x["nll"] if x.get("nll") is not None else -x["score"] for k, x in rows if x.get("nll") is not None or x.get("score") is not None]
        P(f"| {s} | {len(rows)} | {c} | {calls:.1f} | {np.mean(nll) if nll else float('nan'):.2f} | {eq}/{len(rows)} |")
    gs = np.mean([x["forwards"] for k, x in out["alt-greedy-short"].items()]) if out["alt-greedy-short"] else 0
    P(f"| alt-greedy-short | {len(out['alt-greedy-short'])} | n/a | {gs:.1f} | | |")
    cd = [k for k, x in out["alt-bestfirst"].items() if x.get("certified") and x["gen"] != out["alt-greedy-short"][k]["gen"]]
    P(f"\nBest-first certified a sequence different from greedy on {len(cd)} items: each is, by its certificate, the most likely sequence of its length, so greedy's is less likely there ({', '.join(k[0] for k in cd[:8])}{' ...' if len(cd) > 8 else ''}). Beam-4's NLL column is its own summed score; beam certified only {sum(int(bool(x.get('certified'))) for x in out['alt-beam4'].values())} items because its bound (the best pruned prefix) is loose.")
    if out["alt-ngram"]:
        ng = per_prompt("alt-ngram", t1ids, t1m)
        P(f"\nTrigram automaton on T1: ref_match {np.mean([ng[i]['ref_match'] for i in ng]):.2f}/24, ref_f1 {np.mean([ng[i]['ref_f1'] for i in ng]):.3f}, rep3 {np.mean([ng[i]['rep3'] for i in ng]):.3f}, ms/token {ms_per_tok('alt-ngram','t1') if False else np.median([x['wall']*1000/24 for (i,r),x in out['alt-ngram'].items() if i.startswith('t1')]):.3f}.")
    if out["alt-rules"]:
        rows = [(k, x) for k, x in out["alt-rules"].items() if k[0].startswith("t1")]
        fired = sum(x.get("fired", 0) for k, x in rows)
        tot = sum(len(x["gen"]) for k, x in rows)
        agree = sum(int(x["gen"] == out["greedy-f32"][(k[0], 0)]["gen"]) for k, x in rows)
        tokagree = sum(sum(int(a == b) for a, b in zip(x["gen"], out["greedy-f32"][(k[0], 0)]["gen"])) for k, x in rows)
        rb = other.get("rules:built", {})
        P(f"Rule table ({rb.get('rules')} rules): fired on {fired}/{tot} T1 steps (calls saved), output equals greedy on {agree}/{len(rows)} prompts, {tokagree}/{tot} tokens equal at position.\n")


except Exception as e:
    P(f'\n(section alternatives incomplete: {type(e).__name__} {e})\n')
# ---------------------------------------------------------------- sweeps
try:
    P("## Weight sweeps on T3 topic prompts (dose response, reweighting cost)\n")
    for axis in ["topic", "embdir"]:
        P(f"### {axis}\n")
        P("| weight (logits) | hit TRAIN words | hit HELD-OUT words | NLL/token |")
        P("| --- | --- | --- | --- |")
        grid = sorted({float(s.split(":")[2]) for s in out if s.startswith(f"sweep:{axis}:")})
        for w in grid:
            s = f"sweep:{axis}:{w:g}" if f"sweep:{axis}:{w:g}" in out else [k for k in out if k.startswith(f"sweep:{axis}:") and float(k.split(':')[2]) == w][0]
            rows = [(i, x) for (i, r), x in out[s].items()]
            P(f"| {w:g} | {np.mean([t3ok(x['gen'], items[i]) for i, x in rows]):.2f} | {np.mean([t3test(x['gen'], items[i]) for i, x in rows]):.2f} | {np.mean([x['nll'] for i, x in rows]):.2f} |")
        costs = [v for k, v in other.items() if k.startswith(f"sweep-cost:{axis}:")]
        if costs:
            P(f"\nIncremental (content-addressed reuse) vs scratch: {sum(c['calls_incremental'] for c in costs)} vs {sum(c['calls_scratch'] for c in costs)} model calls; identical outputs {sum(c['identical'] for c in costs)}/{sum(c['of'] for c in costs)}.\n")


except Exception as e:
    P(f'\n(section sweeps incomplete: {type(e).__name__} {e})\n')
# ---------------------------------------------------------------- replay
try:
    P("## Lean replay of recorded steps\n")
    for k, v in other.items():
        if k.startswith("replay:"):
            P(f"- {k[7:]}: exit {v['exit']}, steps {v['steps']}, harness pick = proved pick {v['agree']}, rescored pick = harness recompute {v['agreeRe']}, rescored = recomputed {v['reEqRec']}, certified {v['certified']}, cone box held {v['boundStable']} (correct {v['boundStableRight']}), pick changed by the weight change {v['changed']} ({v['seconds']} s)")


except Exception as e:
    P(f'\n(section replay incomplete: {type(e).__name__} {e})\n')
# ---------------------------------------------------------------- Pareto front (T1, minimization)
try:
    P("\n## Pareto front over T1 systems (minimize; Lean `prlm front`)\n")
    P("Objectives: (24 − ref_match)×100, rep3×1000, distinct outputs per prompt ×100, ms/token ×10, calls per token ×100.\n")
    vec_lines = []
    for s in ["samp-t1.0", "samp-t0.7", "samp-p0.9", "greedy-f32", "prlm-0", "prlm-theta", "alt-ngram", "alt-rules"]:
        mm = per_prompt(s, t1ids, t1m, runs=None if s.startswith("samp") else [0])
        if not mm:
            continue
        rm = np.mean([mm[i]["ref_match"] for i in mm])
        r3 = np.mean([mm[i]["rep3"] for i in mm])
        dist = 5 if s.startswith("samp") else 1
        if s.startswith("samp"):
            dist = distinct(s, [0, 1, 2, 3, 4])[0]
        ms = np.median([x["wall"] * 1000 / 24 for (i, r), x in out[s].items() if i.startswith("t1") and r == 0])
        calls = np.mean([x["forwards"] / 24 for (i, r), x in out[s].items() if i.startswith("t1") and r == 0])
        vec_lines.append(f"{s} {int(round((24-rm)*100))} {int(round(r3*1000))} {int(round(dist*100))} {int(round(ms*10))} {int(round(calls*100))}")
    fp = os.path.join(ROOT, "out", "front-t1.txt")
    open(fp, "w").write("\n".join(vec_lines) + "\n")
    for l in vec_lines:
        P(f"    {l}")
    # contract: latency within 10% is a tie, so a second front with ms/token
    # replaced by its 10% bucket floor(log(ms)/log(1.1))
    import math
    vb = []
    for l in vec_lines:
        f = l.split()
        ms = int(f[4]) / 10
        f[4] = str(int(math.log(ms) / math.log(1.1)) if ms > 0 else 0)
        vb.append(" ".join(f))
    fb = os.path.join(ROOT, "out", "front-t1-ties.txt")
    open(fb, "w").write("\n".join(vb) + "\n")
    if os.path.exists(LEAN):
        rb = subprocess.run([LEAN, "front", fb], capture_output=True, text=True)
        P("\nWith latency as 10% buckets (the contract's tie rule):\n")
        for l in rb.stdout.strip().splitlines():
            P(f"    {l}")
    if os.path.exists(LEAN):
        r = subprocess.run([LEAN, "front", fp], capture_output=True, text=True)
        P("\nFront members and the rows that minimize Chebyshev weights solved from each:\n")
        for l in r.stdout.strip().splitlines():
            P(f"    {l}")
    open(os.path.join(ROOT, "out", "REPORT.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

except Exception as e:
    P(f'\n(section Pareto front (T1, minimization) incomplete: {type(e).__name__} {e})\n')
