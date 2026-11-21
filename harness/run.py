"""Contract run. Every output is a record in out/log.jsonl (hash-chained).
Phases: declare, t1, t2, t3, repro, alt, sweep, replay. report.py rebuilds
every number from the log alone.

Usage: python3 run.py PHASE [PHASE ...]
"""
import json, os, subprocess, sys, time
import numpy as np
from common import *
import axes as AX, decoders as D

T = json.load(open(os.path.join(ROOT, "tasks", "tasks.json")))
L = Log()
M = Model(threads=4)
SCR = "/tmp/claude-0/replay"
os.makedirs(SCR, exist_ok=True)
LEAN = "/tmp/claude-0/prlm"

SAMPLERS = {"samp-t1.0": dict(temp=1.0), "samp-t0.7": dict(temp=0.7), "samp-p0.9": dict(temp=1.0, top_p=0.9)}
DECL = {
    "t1": {"prlm-theta": dict(rep=1.5, noloop=4.0)},
    "t3": {"ban": dict(ban=-4000.0), "ban-soft": dict(ban=-8.0), "topic": dict(topic=4.0),
           "embdir": dict(embdir=0.02), "embdir-rand": dict(embdir=0.02), "end": dict(end=4.0)},
    "controls": {"null": dict(null=5.0), "oracle": dict(oracle=5.0)},
    "sweep": {"topic": [0, 1, 2, 3, 4, 6, 8], "embdir": [0, 0.005, 0.01, 0.02, 0.04]},
    "alt": {"beam_B": 4, "bestfirst_budget": 200, "alt_steps": 8, "ngram": "trigram backoff on TRAIN docs",
            "rules": "k=3, min_support=2, from greedy on TRAIN docs"},
}


def rec(system, item, run, gen, info, extra=None, colors=()):
    r = {"system": system, "item": item, "run": run, "gen": [int(t) for t in gen],
         "wall": round(info.get("wall", 0.0), 6), "forwards": info.get("forwards", 0)}
    if "certs" in info:
        cs = info["certs"]
        r["certified"] = sum(1 for c in cs if c[2])
        r["min_margin"] = int(min(c[1] for c in cs))
    for k in ("certified", "score", "nll", "calls", "fired"):
        if k in info and k not in r:
            r[k] = info[k]
    if extra:
        r.update(extra)
    L.add(f"out:{system}:{item}:{run}", ["output", system.split(":")[0]] + list(colors), receipt=r)


def fresh():
    M.cache.clear()
    return M.forwards


def run_one(fn, *a, **kw):
    f0 = fresh()
    g, info = fn(*a, **kw)
    info["forwards"] = M.forwards - f0
    return g, info


def hidden_for(item):
    h = {}
    if "reference" in item:
        h["reference"] = item["reference"]
    if "answer" in item:
        h["answer_tok"] = encode(" " + item["answer"])[0]
    if item.get("kind") == "topic":
        h["test_tokens"] = AX.word_first_tokens(item["test_words"])
    return h


def prlm_run(item, steps, theta, active, replay=None, delta=None, embseed=None):
    req = AX.Request(item, hidden_for(item), embseed=embseed)
    f0 = fresh()
    g, info = D.prlm(M, item["prompt"], steps, req, theta, active, replay=replay, delta=delta)
    info["forwards"] = M.forwards - f0
    return g, info


def phase_declare():
    path = os.path.join(ROOT, "out", "DECLARED.json")
    with open(path, "w") as f:
        json.dump(DECL, f, sort_keys=True, indent=1)
    L.add("declared-weights", ["declared"], premises=[3, 4],
          receipt={"file": "out/DECLARED.json", "sha256": sha256_file(path),
                   "axes_sha256": sha256_file(os.path.join(ROOT, "harness", "axes.py")),
                   "decoders_sha256": sha256_file(os.path.join(ROOT, "harness", "decoders.py")),
                   "run_sha256": sha256_file(os.path.join(ROOT, "harness", "run.py")),
                   "lean_core_sha256": sha256_file(os.path.join(ROOT, "lean", "Prlm", "Core.lean"))})


def phase_t1():
    rp = D.Replay(os.path.join(SCR, "t1-prlm0.bin"))
    dl = AX.weights(**DECL["t1"]["prlm-theta"])
    for it in T["t1"]:
        p = it["prompt"]
        for name, kw in SAMPLERS.items():
            for s in range(5):
                g, info = run_one(D.sample, M, p, 24, seed=s, **kw)
                rec(name, it["id"], s, g, info)
        g, info = run_one(D.greedy, M, p, 24)
        rec("greedy-f32", it["id"], 0, g, info)
        g, info = prlm_run(it, 24, AX.weights(), ["rep", "noloop"], replay=rp, delta=dl)
        rec("prlm-0", it["id"], 0, g, info)
        g, info = prlm_run(it, 24, dl, ["rep", "noloop"])
        rec("prlm-theta", it["id"], 0, g, info)
        g, info = prlm_run(it, 24, AX.weights(**DECL["controls"]["null"]), ["null"])
        rec("ctl-null", it["id"], 0, g, info)
        g, info = prlm_run(it, 24, AX.weights(**DECL["controls"]["oracle"]), ["oracle"])
        rec("GAMED-oracle", it["id"], 0, g, info, colors=["gamed"])
        print("t1", it["id"], flush=True)
    rp.close()


def phase_t2():
    for it in T["t2"]:
        p = it["prompt"]
        for name, kw in SAMPLERS.items():
            for s in range(5):
                g, info = run_one(D.sample, M, p, 4, seed=s, **kw)
                rec(name, it["id"], s, g, info)
        g, info = run_one(D.greedy, M, p, 4)
        rec("greedy-f32", it["id"], 0, g, info)
        g, info = prlm_run(it, 4, AX.weights(), [])
        rec("prlm-0", it["id"], 0, g, info)
        g, info = prlm_run(it, 4, AX.weights(**DECL["controls"]["null"]), ["null"])
        rec("ctl-null", it["id"], 0, g, info)
        g, info = prlm_run(it, 4, AX.weights(**DECL["controls"]["oracle"]), ["oracle"])
        rec("GAMED-oracle", it["id"], 0, g, info, colors=["gamed"])
    print("t2 done", flush=True)


def phase_t3():
    rp = D.Replay(os.path.join(SCR, "t3-theta.bin"))
    for it in T["t3"]:
        p, k = it["prompt"], it["kind"]
        for s in range(5):
            g, info = run_one(D.sample, M, p, 24, seed=s, top_p=0.9)
            rec("samp-p0.9", it["id"], s, g, info, extra={"nll": D.nll_under_base(M, p, g)})
        g, info = run_one(D.greedy, M, p, 24)
        rec("greedy-f32", it["id"], 0, g, info, extra={"nll": D.nll_under_base(M, p, g)})
        g, info = prlm_run(it, 24, AX.weights(), [])
        rec("prlm-0", it["id"], 0, g, info, extra={"nll": D.nll_under_base(M, p, g)})
        names = {"ban": ["ban", "ban-soft"], "topic": ["topic", "embdir", "embdir-rand"], "end": ["end"]}[k]
        for nm in names:
            th = AX.weights(**DECL["t3"][nm])
            act = [nm.split("-")[0]]
            # replay the hard filter run from theta=0 with delta=theta: rescore vs recompute
            g, info = prlm_run(it, 24, th, act, embseed=(7 if nm == "embdir-rand" else None),
                               replay=rp if nm in ("ban-soft", "topic", "end") else None,
                               delta=[-x for x in th] if nm in ("ban-soft", "topic", "end") else None)
            rec("prlm-theta:" + nm, it["id"], 0, g, info, extra={"nll": D.nll_under_base(M, p, g)})
        g, info = prlm_run(it, 24, AX.weights(**DECL["controls"]["null"]), ["null"])
        rec("ctl-null", it["id"], 0, g, info, extra={"nll": D.nll_under_base(M, p, g)})
        if k == "topic":
            g, info = prlm_run(it, 24, AX.weights(**DECL["controls"]["oracle"]), ["oracle"])
            rec("GAMED-oracle", it["id"], 0, g, info, colors=["gamed"], extra={"nll": D.nll_under_base(M, p, g)})
        print("t3", it["id"], flush=True)
    rp.close()


def phase_repro():
    # deterministic systems rerun from an empty cache; one rerun at 1 thread
    global M
    for r in range(1, 4):
        for it in T["t1"]:
            g, info = prlm_run(it, 24, AX.weights(), ["rep", "noloop"])
            rec("prlm-0", it["id"], r, g, info)
            g, info = run_one(D.greedy, M, it["prompt"], 24)
            rec("greedy-f32", it["id"], r, g, info)
            g, info = run_one(D.sample, M, it["prompt"], 24, seed=0, top_p=0.9)
            rec("samp-p0.9-seed0-rerun", it["id"], r, g, info)
    M1 = Model(threads=1)
    M0, M = M, M1
    for it in T["t1"]:
        g, info = prlm_run(it, 24, AX.weights(), ["rep", "noloop"])
        rec("prlm-0", it["id"], 4, g, info, extra={"threads": 1})
    M = M0
    # bitwise logits across thread counts at every prompt
    eq = 0
    for it in T["t1"]:
        a = M.logits(it["prompt"], use_cache=False)
        b = M1.logits(it["prompt"], use_cache=False)
        eq += int(np.array_equal(a, b))
    L.add("repro:logits-threads", ["measured"], receipt={"prompts": len(T["t1"]), "bitwise_equal": eq})
    print("repro done", flush=True)


def phase_alt():
    S = DECL["alt"]["alt_steps"]
    ng = D.NGram([c["ids"] for c in T["corpus"]])
    rt = D.RuleTable(M, [c["ids"] for c in T["corpus"]])
    L.add("rules:built", ["measured"], receipt={"rules": len(rt.rules), "k": rt.k})
    for it in T["t1"] + T["t2"]:
        p = it["prompt"]
        st = S if it["id"].startswith("t1") else 4
        g, info = run_one(D.beam_cert, M, p, st, B=DECL["alt"]["beam_B"])
        rec("alt-beam4", it["id"], 0, g, info)
        g, info = run_one(D.bestfirst_cert, M, p, st, budget=DECL["alt"]["bestfirst_budget"])
        rec("alt-bestfirst", it["id"], 0, g, info)
        g, info = run_one(D.greedy, M, p, st)
        rec("alt-greedy-short", it["id"], 0, g, info)
        g, info = ng.gen(p, 24 if it["id"].startswith("t1") else 4)
        info["forwards"] = 0
        rec("alt-ngram", it["id"], 0, g, info)
        g, info = run_one(rt.gen, M, p, 24 if it["id"].startswith("t1") else 4)
        rec("alt-rules", it["id"], 0, g, info)
        print("alt", it["id"], flush=True)


def phase_sweep():
    """Reweighting along whole generations: for each weight in the grid,
    (a) from scratch with an empty cache, (b) reusing the content-addressed
    cache of every earlier weight. Outputs must be identical; count calls."""
    items = [t for t in T["t3"] if t["kind"] == "topic"]
    for axis, grid in DECL["sweep"].items():
        for it in items:
            M.cache.clear()
            f_inc0 = M.forwards
            outs_inc = []
            for w in grid:
                req = AX.Request(it, hidden_for(it))
                g, _ = D.prlm(M, it["prompt"], 24, req, AX.weights(**{axis: w}), [axis])
                outs_inc.append(g)
            calls_inc = M.forwards - f_inc0
            calls_scr = 0
            same = 0
            for w, gi in zip(grid, outs_inc):
                g, info = prlm_run(it, 24, AX.weights(**{axis: w}), [axis])
                calls_scr += info["forwards"]
                same += int(g == gi)
                rec(f"sweep:{axis}:{w}", it["id"], 0, g, info, extra={"nll": D.nll_under_base(M, it["prompt"], g)})
            L.add(f"sweep-cost:{axis}:{it['id']}", ["measured"],
                  receipt={"grid": grid, "calls_incremental": calls_inc, "calls_scratch": calls_scr,
                           "identical": same, "of": len(grid)})
        print("sweep", axis, flush=True)


def phase_replay():
    for name in sorted(os.listdir(SCR)):
        p = os.path.join(SCR, name)
        t0 = time.perf_counter()
        out = subprocess.run([LEAN, "decide", p], capture_output=True, text=True)
        line = out.stdout.strip().splitlines()[-1]
        kv = dict(zip(line.split()[0::2], map(int, line.split()[1::2])))
        L.add(f"replay:{name}", ["lean-replay", "measured"],
              receipt={"exit": out.returncode, "file_sha256": sha256_file(p), "bytes": os.path.getsize(p),
                       "lean_exe_sha256": sha256_file(LEAN), "seconds": round(time.perf_counter() - t0, 2), **kv})
        print(name, line, flush=True)


if __name__ == "__main__":
    for ph in sys.argv[1:]:
        t0 = time.time()
        globals()["phase_" + ph]()
        L.add(f"phase:{ph}", ["phase"], receipt={"seconds": round(time.time() - t0, 1), "forwards_total": M.forwards})
