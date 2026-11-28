"""Value amendment 7 (tasks/VALUE_AMEND_7.json, record 4996): cross-CPU replay with and
without a pinned instruction set.

  python replay.py 14b OUTDIR --shard K --nshards 20 --replica a --arm pinned
  python replay.py 4b  OUTDIR --shard K --nshards 20 --replica b --arm unpinned
  python replay.py summary OUTFILE DIR...

The pin itself is applied outside this script, before anything loads: llama.cpp is built
with a fixed instruction set (replay.yml) and torch reads ATEN_CPU_CAPABILITY /
ONEDNN_MAX_CPU_ISA from the environment at import. This script records what was set and
which CPU ran it, decodes greedy and prlm-0 with the end-of-document stop of
VALUE_AMEND_5, and keeps the prlm-0 margins so a divergence can be checked against the
certificate (a step is certified when its margin exceeds 2B, B = 4, as in xhw.py).
"""
import glob, json, os, sys, time

N = 40
B = 4
PINS = ("ATEN_CPU_CAPABILITY", "ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA", "LLAMA_ISA")
KEY_FLAGS = ("avx2", "fma", "f16c", "avx512f", "avx512_bf16", "avx512_vnni", "avx_vnni", "amx_tile", "amx_bf16")


def cpu():
    model, flags = "?", []
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name") and model == "?":
                model = l.split(":", 1)[1].strip()
            if l.startswith("flags") and not flags:
                flags = l.split(":", 1)[1].split()
    except OSError:
        pass
    return {"cpu": model, "flags": sorted(f for f in flags if f in KEY_FLAGS)}


def run(kind, outdir, shard, nshards, replica, arm, threads=4):
    import value as V
    import value_eos_run as E          # patches V.run_path with the end-of-document stop
    os.makedirs(outdir, exist_ok=True)
    probs = [p for k, p in enumerate(V.load_test(None)[:N]) if k % nshards == shard]
    if kind == "14b":
        import value_llama as L
        lm = L.LlamaLM(threads)
    else:
        import value_code as C
        lm = C.HF16("Qwen/Qwen3-4B-Base", threads)
    E.patch(lm)
    path = os.path.join(outdir, f"replay-{kind}-{arm}-{replica}-shard{shard}.jsonl")
    done = {json.loads(l)["index"] for l in open(path) if l.strip()} if os.path.exists(path) else set()
    info = {"kind": kind, "arm": arm, "replica": replica, "machine": {**V.machine(), **cpu()},
            "pins": {k: os.environ.get(k) for k in PINS}, "inputs": lm.inputs, "amendment": "VALUE_AMEND_7"}
    for pr in probs:
        if pr["index"] in done:
            continue
        ids = lm.encode(V.PREFIX + f"Q: {pr['question']}\nA:")
        state = lm.extend(None, ids)
        out = {}
        for name, mode in (("greedy", "greedy"), ("prlm-0", "prlm")):
            t0 = time.perf_counter()
            p, g = V.run_path(lm, state, [], [], mode)
            dt = time.perf_counter() - t0
            out[name] = {"path": p.tokens, "answer": p.answer, "correct": V.same(p.answer, pr["gold"]),
                         "gen_tokens": g, "seconds": round(dt, 2),
                         "margins": [m for _, m in p.recs] if mode == "prlm" else None}
        r = {"index": pr["index"], "gold": pr["gold"], "methods": out, **info}
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(pr["index"], {m: int(out[m]["correct"]) for m in out}, flush=True)


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


def summary(outfile, dirs):
    rows = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "replay-*.jsonl"), recursive=True)):
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    rows[(r["kind"], r["arm"], r["index"], r["replica"])] = r
    out = {}
    for kind in ("14b", "4b"):
        for arm in ("pinned", "unpinned"):
            s = {"pairs": 0, "cross_cpu": {"pairs": 0, "greedy_equal": 0, "prlm0_equal": 0},
                 "same_cpu": {"pairs": 0, "greedy_equal": 0, "prlm0_equal": 0},
                 "certified_flips": 0, "divergences": [], "accuracy": {}, "tokens_per_s": {}}
            idx = sorted({k[2] for k in rows if k[0] == kind and k[1] == arm})
            acc = {"greedy": [], "prlm-0": []}
            speed = {}
            for i in idx:
                for rep in ("a", "b"):
                    r = rows.get((kind, arm, i, rep))
                    if r is None:
                        continue
                    for m in acc:
                        M = r["methods"][m]
                        sp = speed.setdefault(r["machine"]["cpu"], [0, 0.0])
                        sp[0] += M["gen_tokens"]; sp[1] += M["seconds"]
                a, b = rows.get((kind, arm, i, "a")), rows.get((kind, arm, i, "b"))
                if a:
                    for m in acc:
                        acc[m].append(a["methods"][m]["correct"])
                if not (a and b):
                    continue
                s["pairs"] += 1
                g = s["cross_cpu" if a["machine"]["cpu"] != b["machine"]["cpu"] else "same_cpu"]
                g["pairs"] += 1
                ge = a["methods"]["greedy"]["path"] == b["methods"]["greedy"]["path"]
                pa, pb = a["methods"]["prlm-0"], b["methods"]["prlm-0"]
                pe = pa["path"] == pb["path"]
                g["greedy_equal"] += int(ge); g["prlm0_equal"] += int(pe)
                if not pe:
                    t = first_diff(pa["path"], pb["path"])
                    ma = pa["margins"][t] if t is not None and t < len(pa["margins"]) else None
                    mb = pb["margins"][t] if t is not None and t < len(pb["margins"]) else None
                    flip = any(m is not None and m > 2 * B for m in (ma, mb))
                    s["certified_flips"] += int(flip)
                    s["divergences"].append({"index": i, "step": t, "margin_a": ma, "margin_b": mb, "certified_flip": flip,
                                             "cpu_a": a["machine"]["cpu"], "cpu_b": b["machine"]["cpu"]})
            s["accuracy"] = {m: round(sum(v) / len(v), 4) if v else None for m, v in acc.items()}
            s["problems"] = len(acc["greedy"])
            s["tokens_per_s"] = {c: round(t / sec, 2) if sec else None for c, (t, sec) in speed.items()}
            cc = s["cross_cpu"]
            s["X1_holds"] = (cc["pairs"] >= 5 and cc["greedy_equal"] == cc["pairs"] and cc["prlm0_equal"] == cc["pairs"]) \
                if arm == "pinned" else None
            s["X3_measured"] = cc["pairs"] >= 5
            out[f"{kind}:{arm}"] = s
        ctl = out[f"{kind}:unpinned"]["cross_cpu"]
        out[f"{kind}:X0_control_detects"] = ctl["greedy_equal"] < ctl["pairs"] or ctl["prlm0_equal"] < ctl["pairs"]
    json.dump(out, open(outfile, "w"), indent=1)
    print(json.dumps(out))


def main():
    a = sys.argv[1:]
    if a[0] == "summary":
        return summary(a[1], a[2:])
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    run(a[0], a[1], int(opt("--shard", 0)), int(opt("--nshards", 20)), opt("--replica", "a"), opt("--arm", "pinned"),
        int(opt("--threads", 4)))


if __name__ == "__main__":
    main()
