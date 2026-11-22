"""Run the self-improving loop until it certifies a fixed point
(tasks/FIXPOINT_CONTRACT.json; what the stop means: lean/Prlm/Fix.lean).

  python3 fixpoint.py CONTRACT_ID
"""
import hashlib, json, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
import loop as LP
from common import ROOT, Log, sha256_file

AXN, CAP, START = LP.AXN, LP.CAP, LP.START
SAFETY = 200_000
SCR = LP.SCR
STEPS = "/tmp/claude-0/steps_t3.pkl"
_P = None


def _init(which):
    global _P
    steps = pickle.load(open(STEPS, "rb"))
    dev, test = LP.split(steps)
    _P = LP.prep(dev if which == "dev" else test)


def _ev(pt):
    vec, picks, dis = LP.evaluate(pt, _P)
    return pt, vec, hashlib.sha256(np.asarray(picks, dtype=np.int64).tobytes()).hexdigest(), dis


class Search:
    def __init__(self, which, tag, L, prem):
        self.pool = Pool(4, initializer=_init, initargs=(which,))
        self.tag, self.L, self.prem = tag, L, prem
        self.ev, self.ph, self.A, self.dis, self.rounds = {}, {}, {}, 0, []
        steps = pickle.load(open(STEPS, "rb"))
        dev, test = LP.split(steps)
        P = LP.prep(dev if which == "dev" else test)
        self.scope = [a for a in AXN if any(fa[a] for _, _, fa in P)]

    def run(self, pts):
        pts = [p for p in dict.fromkeys(pts) if p not in self.ev]
        if len(self.ev) + len(pts) > SAFETY:
            pts = pts[:SAFETY - len(self.ev)]
        admitted = []
        for pt, vec, h, dis in self.pool.imap(_ev, pts, chunksize=16):
            self.ev[pt], self.ph[pt] = vec, h
            self.dis += dis
            if LP.ins(self.A, LP.nm(pt), vec):
                admitted.append(LP.nm(pt))
        return admitted, len(pts)

    def check(self, phase, rnd, admitted, n):
        rc, names = LP.lean_front({LP.nm(p): v for p, v in self.ev.items()}, os.path.join(SCR, f"{self.tag}-fix.txt"))
        front = set(self.A)
        r = {"phase": phase, "round": rnd, "proposed": n, "evaluated": len(self.ev), "admitted": len(admitted),
             "new_front": len([a for a in admitted if a in front]), "front": len(front), "lean_exit": rc,
             "lean_equals_archive": names == front, "disagreements_so_far": self.dis}
        self.rounds.append(r)
        print(self.tag, r, flush=True)
        if self.L:
            self.L.add(f"fixpoint:{self.tag}:{phase}:{rnd}", ["measured", "fixpoint"], premises=self.prem, receipt=r)
        return r

    def frozen(self):
        out = []
        for j, a in enumerate(AXN):
            if a not in self.scope:
                continue
            groups = {}
            for p, h in self.ph.items():
                groups.setdefault(p[:j] + p[j + 1:], set()).add(h)
            cnt, changed = {}, False
            for p in self.ph:
                k = p[:j] + p[j + 1:]
                cnt[k] = cnt.get(k, 0) + 1
            pairs = any(c > 1 for c in cnt.values())
            changed = any(len(groups[k]) > 1 for k in groups)
            if pairs and not changed:
                out.append(a)
        return out

    def phase_a(self):
        cand = [tuple(START * ((i >> j) & 1) if AXN[j] in self.scope else 0 for j in range(len(AXN))) for i in range(32)]
        rnd = 0
        while True:
            admitted, n = self.run(cand)
            r = self.check("A", rnd, admitted, n)
            if (rnd > 0 and r["new_front"] == 0) or len(self.ev) >= SAFETY:
                return
            frozen = self.frozen()
            byname = {LP.nm(p): p for p in self.ev}
            cand = []
            for nme in self.A:
                p = byname[nme]
                for j, a in enumerate(AXN):
                    if a not in self.scope or a in frozen:
                        continue
                    Lv = sorted({q[j] for q in self.ev if q[:j] + q[j + 1:] == p[:j] + p[j + 1:]})
                    lo = max([x for x in Lv if x < p[j]] + [0])
                    hi = [x for x in Lv if x > p[j]]
                    nxt = []
                    if p[j] > lo:
                        nxt.append((lo + p[j]) // 2)
                    nxt.append((p[j] + hi[0]) // 2 if hi else (min(CAP, 2 * p[j]) if p[j] else START))
                    for x in nxt:
                        q = p[:j] + (x,) + p[j + 1:]
                        if q not in self.ev:
                            cand.append(q)
            rnd += 1

    def neighbours(self, p):
        out = []
        for j, a in enumerate(AXN):
            if a not in self.scope:
                continue
            for d in (-1, 1):
                x = p[j] + d
                if 0 <= x <= CAP:
                    out.append(p[:j] + (x,) + p[j + 1:])
        return out

    def phase_b(self):
        rnd = 0
        while True:
            byname = {LP.nm(p): p for p in self.ev}
            cand = [q for nme in self.A for q in self.neighbours(byname[nme]) if q not in self.ev]
            if not cand:
                r = self.check("B", rnd, [], 0)
                return True
            if len(self.ev) >= SAFETY:
                return False
            admitted, n = self.run(cand)
            self.check("B", rnd, admitted, n)
            rnd += 1


def parse_name(n):
    out = []
    rest = n[2:].split("_")
    for a, tok in zip(AXN, rest):
        out.append(int(tok[len(a):]))
    return tuple(out)


def main(cid):
    os.makedirs(SCR, exist_ok=True)
    L = Log()
    prem = [cid]
    t0 = time.perf_counter()
    L.add("fixpoint:declared", ["declared"], premises=prem,
          receipt={"fixpoint_sha256": sha256_file(os.path.abspath(__file__)), "loop_sha256": sha256_file(LP.__file__),
                   "fix_lean_sha256": sha256_file(os.path.join(ROOT, "lean", "Prlm", "Fix.lean")),
                   "lean_exe_sha256": sha256_file(LP.LEAN), "steps_sha256": sha256_file(STEPS), "safety_cap": SAFETY,
                   "note": "steps are the cached record of density.record_steps (the same prompts, model and rule); phase A's frozen-axis check is grouped by the other axes' levels, which is the same test as PR 4's pairwise loop"})
    S = Search("dev", "dev", L, prem)
    S.phase_a()
    init = {LP.nm(p): v for p, v in S.ev.items() if all(x in (0, START) for x in p)}
    I0 = {}
    for n, v in init.items():
        LP.ins(I0, n, v)
    closed = S.phase_b()
    wall = round(time.perf_counter() - t0, 1)
    allok = all(r["lean_equals_archive"] and r["lean_exit"] == 0 for r in S.rounds)
    # TEST scoring of round-0 front, PR 4's final front, the fixed-point front
    pr4 = None
    for l in open(os.path.join(ROOT, "out", "log.jsonl")):
        r = json.loads(l)
        if r["head"] == "loop:result":
            pr4 = list(r["receipt"]["final_front_points"])
    steps = pickle.load(open(STEPS, "rb"))
    dev, test = LP.split(steps)
    PT = LP.prep(test)
    tv = {}
    names = set(S.A) | set(I0) | set(pr4)
    for n in names:
        tv[n] = LP.evaluate(parse_name(n), PT)[0]
    F, I, Q = [tv[n] for n in S.A], [tv[n] for n in I0], [tv[n] for n in pr4]
    res = {"closed": closed, "evaluations": len(S.ev), "rounds_A": sum(r["phase"] == "A" for r in S.rounds),
           "rounds_B": sum(r["phase"] == "B" for r in S.rounds), "dev_front": len(S.A), "lean_equals_archive_every_round": allok,
           "sparse_vs_dense_disagreements": S.dis, "wall_seconds": wall,
           "test_fix_covers_round0": LP.coverage(F, I), "test_round0_covers_fix": LP.coverage(I, F),
           "test_fix_covers_pr4": LP.coverage(F, Q), "test_pr4_covers_fix": LP.coverage(Q, F),
           "test_fix_not_dominated_by_round0": f"{sum(not any(LP.dom(i, f) for i in I) for f in F)}/{len(F)}",
           "test_fix_not_dominated_by_pr4": f"{sum(not any(LP.dom(q, f) for q in Q) for f in F)}/{len(F)}",
           "final_front_points": {n: {"dev": S.A[n], "test": tv[n]} for n in S.A}}
    L.add("fixpoint:result", ["measured", "fixpoint"], premises=prem, receipt=res)
    print("result", json.dumps({k: v for k, v in res.items() if k != "final_front_points"}), flush=True)
    S.pool.close()
    # random control
    rng = np.random.default_rng(1)
    rpts = set()
    while len(rpts) < len(S.ev):
        rpts.add(tuple(int(x) for x in rng.integers(0, CAP + 1, len(AXN))))
    R = Search("dev", "random", None, prem)
    R.run(sorted(rpts))
    RT = [LP.evaluate(parse_name(n), PT)[0] for n in R.A]
    R.pool.close()
    L.add("fixpoint:control:random", ["measured", "fixpoint", "control"], premises=prem,
          receipt={"evaluations": len(R.ev), "dev_front": len(R.A), "test_fix_covers_random": LP.coverage(F, RT),
                   "test_random_covers_fix": LP.coverage(RT, F)})
    L.add("fixpoint:control:none", ["measured", "fixpoint", "control"], premises=prem,
          receipt={"final_equals_initial": True, "test_covers": LP.coverage(I, I)})
    # GAMED: both phases on TEST
    G = Search("test", "GAMED", None, prem)
    G.phase_a()
    gclosed = G.phase_b()
    GV = list(G.A.values())
    G.pool.close()
    L.add("fixpoint:control:GAMED", ["measured", "fixpoint", "control", "gamed"], premises=prem,
          receipt={"closed": gclosed, "evaluations": len(G.ev), "test_front": len(G.A),
                   "test_gamed_covers_fix": LP.coverage(GV, F), "test_fix_covers_gamed": LP.coverage(F, GV)})
    print("controls done", flush=True)
    # Lean replay of the fixed-point front on all 1,440 steps
    lean = LP.lean_replay([parse_name(n) for n in S.A], steps)
    ok = all(v["exit"] == 0 and v["agree"] == v["sparseEq"] == v["steps"] == v["agreeRe"] == v["reEqRec"] for v in lean.values())
    L.add("fixpoint:lean", ["lean-replay", "measured", "fixpoint"], premises=prem, receipt={"all_exact": ok, "points": lean})
    print("lean", len(lean), ok, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
