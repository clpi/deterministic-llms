"""Certify the loop's fixed point one resolution at a time
(tasks/FIXPOINT_AMEND_1.json; proofs in lean/Prlm/Fix.lean).

  python3 fixladder.py CONTRACT_ID
"""
import json, os, pickle, sys, time
import numpy as np
import loop as LP
import fixpoint as FP
from common import ROOT, Log, sha256_file

AXN, CAP, START = LP.AXN, LP.CAP, LP.START
R0 = [64, 32, 32, 32, 32, 1024]
SAFETY = FP.SAFETY
MAX_REPLAY = 600


def res(k):
    return [max(1, x >> k) for x in R0]


def coarse(v, r):
    return [x // y for x, y in zip(v, r)]


def ins_cell(A, name, cv):
    """Archive insertion with one representative per cell (first arrival)."""
    if any(LP.dom(v, cv) or v == cv for v in A.values()):
        return False
    for n in [n for n, v in A.items() if LP.dom(cv, v)]:
        del A[n]
    A[name] = cv
    return True


class Ladder(FP.Search):
    def __init__(self, which, tag, L, prem):
        super().__init__(which, tag, L, prem)
        self.r = res(0)
        self.order = []

    def run(self, pts):
        pts = [p for p in dict.fromkeys(pts) if p not in self.ev]
        if len(self.ev) + len(pts) > SAFETY:
            pts = pts[:SAFETY - len(self.ev)]
        admitted = []
        for pt, vec, h, dis in self.pool.imap(FP._ev, pts, chunksize=16):
            self.ev[pt], self.ph[pt] = vec, h
            self.order.append(pt)
            self.dis += dis
            if ins_cell(self.A, LP.nm(pt), coarse(vec, self.r)):
                admitted.append(LP.nm(pt))
        return admitted, len(pts)

    def rebuild(self, k):
        self.r = res(k)
        self.A = {}
        for p in self.order:
            ins_cell(self.A, LP.nm(p), coarse(self.ev[p], self.r))

    def check(self, phase, rnd, admitted, n):
        rc, names = LP.lean_front({LP.nm(p): coarse(v, self.r) for p, v in self.ev.items()},
                                  os.path.join(LP.SCR, f"{self.tag}-ladder.txt"))
        front = set(self.A)
        cells = len({tuple(v) for v in self.A.values()})
        allv = {LP.nm(p): tuple(coarse(v, self.r)) for p, v in self.ev.items()}
        lean_cells = {allv[n] for n in names}
        r = {"phase": phase, "round": rnd, "resolution": self.r, "proposed": n, "evaluated": len(self.ev),
             "admitted": len(admitted), "new_front": len([a for a in admitted if a in front]), "front": len(front),
             "cells": cells, "lean_exit": rc, "lean_equals_archive": lean_cells == {tuple(v) for v in self.A.values()}, "disagreements_so_far": self.dis}
        self.rounds.append(r)
        print(self.tag, r, flush=True)
        if self.L:
            self.L.add(f"ladder:{self.tag}:{phase}:{rnd}", ["measured", "fixpoint", "ladder"], premises=self.prem, receipt=r)
        return r

    def climb(self, kmax=10):
        certified = []
        self.phase_a()
        for k in range(0, kmax + 1):
            if k > 0:
                self.rebuild(k)
            ok = self.phase_b_level(k)
            if not ok:
                break
            certified.append({"level": k, "resolution": self.r, "front": len(self.A),
                              "cells": len({tuple(v) for v in self.A.values()}), "evaluated": len(self.ev),
                              "points": sorted(self.A)})
            if self.L:
                c = dict(certified[-1]); c["points"] = len(c["points"])
                self.L.add(f"ladder:{self.tag}:certified:{k}", ["measured", "fixpoint", "ladder", "certified"],
                           premises=self.prem, receipt=c)
        return certified

    def phase_b_level(self, k):
        rnd = 0
        while True:
            byname = {LP.nm(p): p for p in self.ev}
            cand = list(dict.fromkeys(q for nme in self.A for q in self.neighbours(byname[nme]) if q not in self.ev))
            if not cand:
                self.check(f"B{k}", rnd, [], 0)
                return True
            if len(self.ev) + len(cand) > SAFETY:
                self.run(cand)
                self.check(f"B{k}", rnd, [], len(cand))
                return False
            admitted, n = self.run(cand)
            self.check(f"B{k}", rnd, admitted, n)
            rnd += 1


def main(cid):
    os.makedirs(LP.SCR, exist_ok=True)
    L = Log()
    prem = [cid]
    t0 = time.perf_counter()
    L.add("ladder:declared", ["declared"], premises=prem,
          receipt={"fixladder_sha256": sha256_file(os.path.abspath(__file__)), "fixpoint_sha256": sha256_file(FP.__file__),
                   "loop_sha256": sha256_file(LP.__file__), "fix_lean_sha256": sha256_file(os.path.join(ROOT, "lean", "Prlm", "Fix.lean")),
                   "steps_sha256": sha256_file(FP.STEPS), "R0": R0, "safety_cap": SAFETY})
    S = Ladder("dev", "dev", L, prem)
    cert = S.climb()
    S.pool.close()
    wall = round(time.perf_counter() - t0, 1)
    allok = all(r["lean_equals_archive"] and r["lean_exit"] == 0 for r in S.rounds)
    steps = pickle.load(open(FP.STEPS, "rb"))
    dev, test = LP.split(steps)
    PT = LP.prep(test)
    pr4 = None
    for l in open(os.path.join(ROOT, "out", "log.jsonl")):
        r = json.loads(l)
        if r["head"] == "loop:result":
            pr4 = list(r["receipt"]["final_front_points"])
    init = {}
    for p in S.order:
        if all(x in (0, START) for x in p):
            LP.ins(init, LP.nm(p), S.ev[p])
    fin = cert[-1] if cert else None
    out = {"certified_levels": [{k: v for k, v in c.items() if k != "points"} for c in cert], "evaluations": len(S.ev),
           "lean_equals_archive_every_round": allok, "sparse_vs_dense_disagreements": S.dis, "wall_seconds": wall,
           "exact_fixed_point": bool(cert) and cert[-1]["level"] == 10}
    if fin:
        tv = {n: LP.evaluate(FP.parse_name(n), PT)[0] for n in set(fin["points"]) | set(init) | set(pr4)}
        F = [tv[n] for n in fin["points"]]
        I = [tv[n] for n in init]
        Q = [tv[n] for n in pr4]
        out.update({"finest_level": fin["level"], "finest_resolution": fin["resolution"],
                    "test_fix_covers_round0": LP.coverage(F, I), "test_round0_covers_fix": LP.coverage(I, F),
                    "test_fix_covers_pr4": LP.coverage(F, Q), "test_pr4_covers_fix": LP.coverage(Q, F),
                    "test_fix_not_dominated_by_round0": f"{sum(not any(LP.dom(i, f) for i in I) for f in F)}/{len(F)}",
                    "test_fix_not_dominated_by_pr4": f"{sum(not any(LP.dom(q, f) for q in Q) for f in F)}/{len(F)}",
                    "finest_front_points": {n: {"dev": S.ev[FP.parse_name(n)], "test": tv[n]} for n in fin["points"]}})
    L.add("ladder:result", ["measured", "fixpoint", "ladder"], premises=prem, receipt=out)
    print("result", json.dumps({k: v for k, v in out.items() if k != "finest_front_points"}), flush=True)
    if not fin:
        return
    # random control at the finest certified resolution
    rng = np.random.default_rng(1)
    rpts = set()
    while len(rpts) < len(S.ev):
        rpts.add(tuple(int(x) for x in rng.integers(0, CAP + 1, len(AXN))))
    R = Ladder("dev", "random", None, prem)
    R.r = fin["resolution"]
    R.run(sorted(rpts))
    R.pool.close()
    RT = [LP.evaluate(FP.parse_name(n), PT)[0] for n in R.A]
    L.add("ladder:control:random", ["measured", "fixpoint", "control"], premises=prem,
          receipt={"evaluations": len(R.ev), "dev_front": len(R.A), "test_fix_covers_random": LP.coverage(F, RT),
                   "test_random_covers_fix": LP.coverage(RT, F)})
    print("random", len(R.A), LP.coverage(F, RT), LP.coverage(RT, F), flush=True)
    L.add("ladder:control:none", ["measured", "fixpoint", "control"], premises=prem,
          receipt={"final_equals_initial": True, "test_covers": LP.coverage(I, I)})
    G = Ladder("test", "GAMED", None, prem)
    gcert = G.climb(kmax=fin["level"])
    G.pool.close()
    if gcert:
        GV = [coarse(G.ev[FP.parse_name(n)], [1] * 6) for n in gcert[-1]["points"]]
        L.add("ladder:control:GAMED", ["measured", "fixpoint", "control", "gamed"], premises=prem,
              receipt={"certified_to": gcert[-1]["level"], "evaluations": len(G.ev), "test_front": len(GV),
                       "test_gamed_covers_fix": LP.coverage(GV, F), "test_fix_covers_gamed": LP.coverage(F, GV)})
        print("GAMED", len(GV), LP.coverage(GV, F), LP.coverage(F, GV), flush=True)
    rep = next((c for c in reversed(cert) if len(c["points"]) <= MAX_REPLAY), None)
    if rep:
        lean = LP.lean_replay([FP.parse_name(n) for n in rep["points"]], steps)
        ok = all(v["exit"] == 0 and v["agree"] == v["sparseEq"] == v["steps"] == v["agreeRe"] == v["reEqRec"] for v in lean.values())
        L.add("ladder:lean", ["lean-replay", "measured", "fixpoint"], premises=prem,
              receipt={"level": rep["level"], "all_exact": ok, "points": lean})
        print("lean", rep["level"], len(lean), ok, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
