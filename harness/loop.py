"""Self-improving, scope-solving Pareto search over PRLM weights
(tasks/LOOP_CONTRACT.json; soundness in lean/Prlm/Loop.lean).

  python3 loop.py CONTRACT_ID
"""
import json, os, struct, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import axes as AX
import decoders as D
import density as DN
from common import ROOT, Log, Model, sha256_file

AXN = DN.AXN
V = 50257
CAP, START, UNIT = 64, 32, 64          # quarter-logit levels; 64/256 logit per level
MAX_ROUNDS, MAX_EVALS = 10, 600
LEAN = "/tmp/claude-0/prlm"
SCR = "/tmp/claude-0/loop"
T = DN.T


# ---------------------------------------------------------------- data
def split(steps):
    dev, test = [], []
    for s in steps:
        k = int(s["item"]["id"].rsplit("/", 1)[1])
        (dev if k < 10 else test).append(s)
    return dev, test


def prep(steps):
    out = []
    for s in steps:
        b = s["base"]
        order = np.lexsort((np.arange(V), -b))
        fa = {a: {} for a in AXN}
        for t, ai, v in s["ents"]:
            for a in AXN:
                if AX.IDX[a] == ai:
                    fa[a][t] = fa[a].get(t, 0) + v
        out.append((s, order, fa))
    return out


HELD = {it["id"]: set(AX.word_first_tokens(it["test_words"])) for it in T["t3"] if it["kind"] == "topic"}
BAN = {it["id"]: set(AX.ban_tokens(it["words"])) for it in T["t3"] if it["kind"] == "ban"}
ENDT = set(AX.end_tokens())


def evaluate(pt, P):
    """Objective vector, the decision of every step, and sparse-vs-dense disagreements."""
    work = bh = tm = em = lp = drift = dis = 0
    picks = []
    for s, order, fa in P:
        b = s["base"]
        mv = {}
        for a, lv in zip(AXN, pt):
            if lv:
                for t, v in fa[a].items():
                    mv[t] = mv.get(t, 0) + UNIT * lv * v
        moved = [t for t, x in mv.items() if x != 0]
        inS = set(moved)
        r = next(i for i, t in enumerate(order) if t not in inS)
        cands = [(t, int(b[t]) + mv[t]) for t in moved] + [(int(order[r]), int(b[order[r]]))]
        pick = max(cands, key=lambda x: (x[1], -x[0]))[0]
        dense = b.copy()
        for t in moved:
            dense[t] += mv[t]
        dis += int(np.argmax(dense)) != pick
        work += len(moved) + r + 1
        it = s["item"]
        k = it["kind"]
        bh += k == "ban" and pick in BAN[it["id"]]
        tm += k == "topic" and pick not in HELD[it["id"]]
        em += k == "end" and pick not in ENDT
        lp += pick in s["loopset"]
        drift += int(b[order[0]]) - int(b[pick])
        picks.append(pick)
    return [round(work / len(P)), bh, tm, em, lp, drift // 256], tuple(picks), dis


# ---------------------------------------------------------------- archive (mirrors AlgoSearch.ins)
def le(a, b):
    return all(x <= y for x, y in zip(a, b))


def dom(a, b):
    return le(a, b) and a != b


def ins(A, name, vec):
    if any(dom(v, vec) for v in A.values()):
        return False
    for n in [n for n, v in A.items() if dom(vec, v)]:
        del A[n]
    A[name] = vec
    return True


def nm(pt):
    return "p_" + "_".join(f"{a}{x}" for a, x in zip(AXN, pt))


def lean_front(vecs, path):
    with open(path, "w") as f:
        for n, v in vecs.items():
            f.write(n + " " + " ".join(str(int(x)) for x in v) + "\n")
    out = subprocess.run([LEAN, "front", path], capture_output=True, text=True)
    return out.returncode, {l.split()[1] for l in out.stdout.splitlines() if l.startswith("front ")}


# ---------------------------------------------------------------- the loop
def run_loop(P, tag):
    ev, picks, dis_tot, rounds = {}, {}, 0, []
    A = {}
    structural = [a for a in AXN if not any(fa[a] for _, _, fa in P)]
    scope = [a for a in AXN if a not in structural]
    cand = [tuple(START * ((i >> j) & 1) if AXN[j] in scope else 0 for j in range(len(AXN))) for i in range(32)]
    cand = list(dict.fromkeys(cand))
    for rnd in range(MAX_ROUNDS + 1):
        new_front = 0
        admitted = []
        for pt in cand:
            if pt in ev or len(ev) >= MAX_EVALS:
                continue
            vec, pk, dis = evaluate(pt, P)
            ev[pt], picks[pt] = vec, pk
            dis_tot += dis
            if ins(A, nm(pt), vec):
                admitted.append(nm(pt))
        front_names = set(A)
        rc, lean_names = lean_front({nm(p): v for p, v in ev.items()}, os.path.join(SCR, f"{tag}-front.txt"))
        new_front = len([n for n in admitted if n in front_names])
        # measured scope: an axis is frozen if varying it alone never changed a decision
        frozen = []
        for j, a in enumerate(AXN):
            if a not in scope:
                continue
            pairs = changed = 0
            for p in ev:
                for q in ev:
                    if p < q and all(p[i] == q[i] for i in range(len(AXN)) if i != j):
                        pairs += 1
                        changed += picks[p] != picks[q]
            if pairs and not changed:
                frozen.append(a)
        rounds.append({"round": rnd, "evaluated": len(ev), "admitted": len(admitted), "new_front": new_front,
                       "front": len(front_names), "lean_exit": rc, "lean_equals_archive": lean_names == front_names,
                       "structural_out": structural, "frozen": frozen})
        print(tag, rounds[-1], flush=True)
        if rnd > 0 and new_front == 0 or len(ev) >= MAX_EVALS or rnd == MAX_ROUNDS:
            break
        # propose around the front
        cand = []
        byname = {nm(p): p for p in ev}
        for n in A:
            p = byname[n]
            for j, a in enumerate(AXN):
                if a not in scope or a in frozen:
                    continue
                Lv = sorted({q[j] for q in ev if all(q[i] == p[i] for i in range(len(AXN)) if i != j)})
                lo = max([x for x in Lv if x < p[j]] + [0])
                hi = [x for x in Lv if x > p[j]]
                nxt = []
                if p[j] > lo:
                    nxt.append((lo + p[j]) // 2)
                if hi:
                    nxt.append((p[j] + hi[0]) // 2)
                else:
                    nxt.append(min(CAP, 2 * p[j]) if p[j] else START)
                for x in nxt:
                    q = tuple(x if i == j else p[i] for i in range(len(AXN)))
                    if q not in ev and q not in cand:
                        cand.append(q)
    return {"evaluated": ev, "archive": dict(A), "rounds": rounds, "disagreements": dis_tot,
            "initial_front": rounds[0]["front"]}, A


def initial_front(ev_all, P):
    A = {}
    for pt in [p for p in ev_all if all(x in (0, START) for x in p)]:
        ins(A, nm(pt), ev_all[pt])
    return A


def coverage(X, Y):
    """Share of Y strictly dominated (and weakly covered) by some point of X."""
    if not Y:
        return 0.0, 0.0
    s = sum(any(dom(x, y) for x in X) for y in Y) / len(Y)
    w = sum(any(le(x, y) for x in X) for y in Y) / len(Y)
    return round(s, 4), round(w, 4)


def lean_replay(pts, steps):
    def one(pt):
        th = AX.weights()
        dl = AX.weights(**{a: x / 4 for a, x in zip(AXN, pt)})
        path = os.path.join(SCR, nm(pt) + ".bin")
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
        os.remove(path)
        line = o.stdout.strip().splitlines()[-1]
        kv = dict(zip(line.split()[0::2], map(int, line.split()[1::2])))
        return nm(pt), {"exit": o.returncode, **{k: kv[k] for k in ("steps", "agree", "agreeRe", "reEqRec", "sparseEq")}}
    with ThreadPoolExecutor(4) as ex:
        return dict(ex.map(one, pts))


def main(cid):
    os.makedirs(SCR, exist_ok=True)
    L = Log()
    prem = [cid]
    t0 = time.perf_counter()
    steps = DN.record_steps(Model(threads=4))
    dev, test = split(steps)
    PD, PT = prep(dev), prep(test)
    L.add("loop:declared", ["declared"], premises=prem,
          receipt={"loop_sha256": sha256_file(os.path.abspath(__file__)), "loop_lean_sha256": sha256_file(os.path.join(ROOT, "lean", "Prlm", "Loop.lean")),
                   "lean_exe_sha256": sha256_file(LEAN), "dev_steps": len(dev), "test_steps": len(test),
                   "record_seconds": round(time.perf_counter() - t0, 1),
                   "note": "a level-0 axis with no evaluated level above it proposes the start level 32, since doubling 0 stays 0"})

    res, A = run_loop(PD, "loop")
    for r in res["rounds"]:
        L.add(f"loop:round:{r['round']}", ["measured", "loop"], premises=prem, receipt=r)
    ev = res["evaluated"]
    init = initial_front(ev, PD)
    test_vec = {}
    for pt in {p for p in ev if nm(p) in A or nm(p) in init}:
        test_vec[nm(pt)] = evaluate(pt, PT)[0]
    I = [test_vec[n] for n in init]
    F = [test_vec[n] for n in A]
    noreg = sum(not any(dom(i, f) for i in I) for f in F)
    cert = {"evaluations": len(ev), "rounds": len(res["rounds"]) - 1, "dev_initial_front": len(init),
            "dev_final_front": len(A), "sparse_vs_dense_disagreements": res["disagreements"],
            "test_final_covers_initial": coverage(F, I), "test_initial_covers_final": coverage(I, F),
            "test_final_not_dominated_by_initial": f"{noreg}/{len(F)}",
            "dev_final_covers_initial": coverage([ev[p] for p in ev if nm(p) in A], [ev[p] for p in ev if nm(p) in init]),
            "final_front_points": {n: {"dev": ev[next(p for p in ev if nm(p) == n)], "test": test_vec[n]} for n in A}}
    L.add("loop:result", ["measured", "loop"], premises=prem, receipt=cert)
    print("result", json.dumps({k: v for k, v in cert.items() if k != "final_front_points"}), flush=True)

    # random control: same number of DEV evaluations
    r = np.random.default_rng(0)
    RA, rev = {}, {}
    while len(rev) < len(ev):
        pt = tuple(int(x) for x in r.integers(0, CAP + 1, len(AXN)))
        if pt in rev:
            continue
        rev[pt] = evaluate(pt, PD)[0]
        ins(RA, nm(pt), rev[pt])
    R = [evaluate(next(p for p in rev if nm(p) == n), PT)[0] for n in RA]
    L.add("loop:control:random", ["measured", "loop", "control"], premises=prem,
          receipt={"evaluations": len(rev), "dev_front": len(RA), "test_loop_covers_random": coverage(F, R),
                   "test_random_covers_loop": coverage(R, F)})
    print("random", len(RA), coverage(F, R), coverage(R, F), flush=True)
    # do-nothing control
    L.add("loop:control:none", ["measured", "loop", "control"], premises=prem,
          receipt={"final_equals_initial": True, "test_covers": coverage(I, I)})
    # GAMED: select on TEST
    gres, GA = run_loop(PT, "GAMED")
    G = list(GA.values())
    L.add("loop:control:GAMED", ["measured", "loop", "control", "gamed"], premises=prem,
          receipt={"evaluations": len(gres["evaluated"]), "rounds": len(gres["rounds"]) - 1, "test_front": len(GA),
                   "test_gamed_covers_loop": coverage(G, F), "test_loop_covers_gamed": coverage(F, G)})
    print("GAMED", len(GA), coverage(G, F), coverage(F, G), flush=True)
    # Lean replay of the final front on all 1,440 steps
    lean = lean_replay([p for p in ev if nm(p) in A], steps)
    ok = all(v["exit"] == 0 and v["agree"] == v["sparseEq"] == v["steps"] == v["agreeRe"] == v["reEqRec"] for v in lean.values())
    L.add("loop:lean", ["lean-replay", "measured", "loop"], premises=prem, receipt={"all_exact": ok, "points": lean})
    print("lean", len(lean), ok, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
