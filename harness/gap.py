"""Like-for-like gap between the main ladder and its gamed control
(tasks/GAP_CONTRACT.json).

  python3 gap.py GAP_RECORD_ID
"""
import os, pickle, sys
from multiprocessing import Pool
import loop as LP
import fixpoint as FP
from common import Log

CK = os.environ.get("LADDER_CKPT_DIR", "/mnt/project-files/prlm-ckpt")
CACHE = os.path.join(CK, "gap-cache.pkl")


def score(which, pts, cache):
    todo = [p for p in pts if (which, p) not in cache]
    with Pool(4, initializer=FP._init, initargs=(which,)) as pool:
        for i, (pt, vec, _, _) in enumerate(pool.imap(FP._ev, todo, chunksize=16)):
            cache[(which, pt)] = vec
            if i % 2000 == 1999:
                pickle.dump(cache, open(CACHE + ".tmp", "wb")); os.replace(CACHE + ".tmp", CACHE)
                print(which, i + 1, "of", len(todo), flush=True)
    pickle.dump(cache, open(CACHE + ".tmp", "wb")); os.replace(CACHE + ".tmp", CACHE)


def main(cid):
    M = pickle.load(open(os.path.join(CK, "ladder-rerun.pkl"), "rb"))
    G = pickle.load(open(os.path.join(CK, "ladder-GAMED-rerun.pkl"), "rb"))
    mf = {c["level"]: [FP.parse_name(n) for n in c["points"]] for c in M["cert"] if c["level"] <= 8}
    gf = {c["level"]: [FP.parse_name(n) for n in c["points"]] for c in G["cert"] if c["level"] <= 8}
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    for p in set().union(*mf.values()):
        cache[("dev", p)] = M["ev"][p]
    for p in set().union(*gf.values()):
        cache[("test", p)] = G["ev"][p]
    score("test", sorted(set().union(*mf.values())), cache)
    score("dev", sorted(set().union(*gf.values())), cache)
    rows = []
    for k in range(9):
        mt = [cache[("test", p)] for p in mf[k]]; gt = [cache[("test", p)] for p in gf[k]]
        md = [cache[("dev", p)] for p in mf[k]]; gd = [cache[("dev", p)] for p in gf[k]]
        r = {"level": k, "main": len(mf[k]), "gamed": len(gf[k]),
             "test_gamed_covers_main": LP.coverage(gt, mt), "test_main_covers_gamed": LP.coverage(mt, gt),
             "dev_main_covers_gamed": LP.coverage(md, gd), "dev_gamed_covers_main": LP.coverage(gd, md),
             "control_test_main_self": LP.coverage(mt, mt)[0], "control_dev_gamed_self": LP.coverage(gd, gd)[0]}
        r["gap_strict"] = round(r["test_gamed_covers_main"][0] - r["dev_main_covers_gamed"][0], 4)
        rows.append(r)
        print(r, flush=True)
    L8 = rows[8]
    verdict = "symmetric" if abs(L8["gap_strict"]) <= 0.05 else ("open gap" if L8["gap_strict"] > 0.05 else "main ahead")
    Log().add("gap:result", ["measured", "fixpoint", "control"], premises=[cid, 4978],
              receipt={"levels": rows, "level8_gap_strict": L8["gap_strict"], "verdict": verdict})
    print("verdict", verdict, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
