"""Density as the computed consequence of on-the-fly reweighting, solved on the
Pareto front (tasks/DENSITY_CONTRACT.json).

  python3 density.py CONTRACT_ID
"""
import itertools, json, os, struct, subprocess, sys, time
import numpy as np
import axes as AX
import decoders as D
from common import ROOT, Log, Model, qscores, sha256_file, encode

LEAN = "/tmp/claude-0/prlm"
SCR = "/tmp/claude-0/density"
AXN = ["rep", "noloop", "ban", "topic", "end"]
LEVELS = [0, 2, 8]
STEPS = 24
T = json.load(open(os.path.join(ROOT, "tasks", "tasks.json")))
V = 50257


def record_steps(M):
    """prlm-0 trajectories with base scores and all five axes' features."""
    out = []
    for it in T["t3"]:
        req = AX.Request(it, {})
        c, gs = list(it["prompt"]), len(it["prompt"])
        for _ in range(STEPS):
            base = qscores(M.logits(c, use_cache=False))
            ents = AX.features(c, gs, req, AXN)
            seen = set()
            if len(c) >= 2:
                a, b = c[-2], c[-1]
                seen = {c[i + 2] for i in range(len(c) - 2) if c[i] == a and c[i + 1] == b}
            out.append({"item": it, "base": base, "ents": ents, "content": list(c), "gs": gs, "loopset": seen})
            v = D.certify(base)[0]
            c.append(v)
    return out


def main(cid):
    os.makedirs(SCR, exist_ok=True)
    L = Log()
    prem = [cid]
    t0 = time.perf_counter()
    M = Model(threads=4)
    steps = record_steps(M)
    held = {it["id"]: set(AX.word_first_tokens(it["test_words"])) for it in T["t3"] if it["kind"] == "topic"}
    ban = {it["id"]: set(AX.ban_tokens(it["words"])) for it in T["t3"] if it["kind"] == "ban"}
    endt = set(AX.end_tokens())
    idx = {a: AX.IDX[a] for a in AXN}
    # per step: old order (ties to the lower id), per-axis sparse feature vectors
    pre = []
    for s in steps:
        b = s["base"]
        order = np.lexsort((np.arange(V), -b))
        rank = np.empty(V, np.int64)
        rank[order] = np.arange(V)
        fa = {a: {} for a in AXN}
        for t, ai, v in s["ents"]:
            for a in AXN:
                if idx[a] == ai:
                    fa[a][t] = fa[a].get(t, 0) + v
        pre.append((order, fa))
    L.add("density:declared", ["declared"], premises=prem,
          receipt={"density_sha256": sha256_file(os.path.abspath(__file__)), "lean_exe_sha256": sha256_file(LEAN),
                   "steps": len(steps), "record_seconds": round(time.perf_counter() - t0, 1),
                   "topic_objective": "held-out test words of each topic prompt"})
    pts, history, disagree, rows = [], [], 0, {}
    for lv in itertools.product(LEVELS, repeat=len(AXN)):
        name = "d_" + "_".join(f"{a}{x}" for a, x in zip(AXN, lv))
        delta = dict(zip(AXN, lv))
        work = moved_tot = bh = tm = em = lp = drift = 0
        for s, (order, fa) in zip(steps, pre):
            b = s["base"]
            mv = {}
            for a in AXN:
                if delta[a]:
                    w = round(256 * delta[a])
                    for t, v in fa[a].items():
                        mv[t] = mv.get(t, 0) + w * v
            moved = [t for t, x in mv.items() if x != 0]
            inS = set(moved)
            r = next(i for i, t in enumerate(order) if t not in inS)  # rank of first unmoved
            cands = [(t, int(b[t]) + mv[t]) for t in moved] + [(int(order[r]), int(b[order[r]]))]
            pick = max(cands, key=lambda x: (x[1], -x[0]))[0]       # best_split
            dense = b.copy()
            for t in moved:
                dense[t] += mv[t]
            if int(np.argmax(dense)) != pick:
                disagree += 1
            work += len(moved) + r + 1
            moved_tot += len(moved)
            it = s["item"]
            k = it["kind"]
            bh += k == "ban" and pick in ban[it["id"]]
            tm += k == "topic" and pick not in held[it["id"]]
            em += k == "end" and pick not in endt
            lp += pick in s["loopset"]
            drift += int(b[order[0]]) - int(b[pick])
        n = len(steps)
        vec = [round(work / n), bh, tm, em, lp, drift // 256]
        rows[name] = {"delta": delta, "work_mean": round(work / n, 2), "density_mean": round(moved_tot / n / V, 6),
                      "ban_hits": bh, "topic_misses": tm, "end_misses": em, "loops": lp, "drift": drift // 256}
        prev = set(x["point"] for x in history[-1]["front"]) if history else set()
        pts.append((name, vec))
        path = os.path.join(SCR, "front.txt")
        with open(path, "w") as f:
            for nm, v in pts:
                f.write(nm + " " + " ".join(str(int(x)) for x in v) + "\n")
        out = subprocess.run([LEAN, "front", path], capture_output=True, text=True)
        fr = []
        for line in out.stdout.splitlines():
            if line.startswith("front "):
                fr.append({"point": line.split()[1], "chebmin": line.split("chebmin ", 1)[1].strip("[] ").split(", ")})
        names = set(x["point"] for x in fr)
        history.append({"after": name, "exit": out.returncode, "front": fr,
                        "monotone": names <= (prev | {name})})
    L.add("density:points", ["measured", "density"], premises=prem,
          receipt={"points": rows, "sparse_vs_dense_disagreements": disagree,
                   "decisions": len(rows) * len(steps)})
    L.add("density:fronts", ["measured", "front"], premises=prem,
          receipt={"appends": len(history), "all_monotone": all(h["monotone"] for h in history),
                   "all_exit0": all(h["exit"] == 0 for h in history),
                   "front_sizes": [len(h["front"]) for h in history],
                   "final_front": history[-1]["front"]})
    # Lean replay for the final front points: theta 0, change = the point
    lean = {}
    for fpt in history[-1]["front"]:
        nm = fpt["point"]
        th = AX.weights()
        dl = AX.weights(**rows[nm]["delta"])
        path = os.path.join(SCR, nm + ".bin")
        with open(path, "wb") as f:
            for s in steps:
                base, ents = s["base"], s["ents"]
                v = D.certify(base + D.dense(ents, th))[0]
                v2 = D.certify(base + D.dense(ents, [a + b for a, b in zip(th, dl)]))[0]
                f.write(struct.pack("<II", V, len(th)))
                f.write(np.asarray(base, dtype="<i4").tobytes())
                f.write(struct.pack(f"<{len(th)}i", *th))
                f.write(struct.pack(f"<{len(dl)}i", *dl))
                f.write(struct.pack("<I", len(ents)))
                if ents:
                    f.write(np.asarray(ents, dtype="<i4").tobytes())
                f.write(struct.pack("<ii", v, v2))
        o = subprocess.run([LEAN, "decide", path], capture_output=True, text=True)
        line = o.stdout.strip().splitlines()[-1]
        kv = dict(zip(line.split()[0::2], map(int, line.split()[1::2])))
        lean[nm] = {"exit": o.returncode, **{k: kv[k] for k in ("steps", "agree", "agreeRe", "reEqRec", "sparseEq", "work", "changed")}}
        os.remove(path)
    L.add("density:lean", ["lean-replay", "measured", "density"], premises=prem, receipt=lean)
    print(json.dumps({"disagree": disagree, "front": [x["point"] for x in history[-1]["front"]],
                      "monotone": all(h["monotone"] for h in history), "lean": lean}), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
