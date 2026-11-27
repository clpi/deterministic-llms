"""The do-nothing control of tasks/GAP_CONTRACT.json as written: each front
scored against itself on its OWN split (main on DEV, gamed on TEST) must give
strict share 0. gap.py logged a different self-comparison (main on TEST,
gamed on DEV); this adds the contracted one beside it.

  python3 gap_control.py GAP_RECORD_ID GAP_RESULT_ID
"""
import os, pickle, sys
import numpy as np
import fixpoint as FP
from common import Log

CK = os.environ.get("LADDER_CKPT_DIR", "/mnt/project-files/prlm-ckpt")


def strict_self(V):
    V = np.asarray(V, dtype=np.int64)
    hit = np.zeros(len(V), dtype=bool)
    for i in range(0, len(V), 512):
        X = V[i:i + 512]                     # candidates that may dominate
        le = (X[:, None, :] <= V[None, :, :]).all(-1)
        lt = (X[:, None, :] < V[None, :, :]).any(-1)
        hit |= (le & lt).any(0)
    return round(float(hit.mean()), 4) if len(V) else 0.0


def main(cid, rid):
    cache = pickle.load(open(os.path.join(CK, "gap-cache.pkl"), "rb"))
    M = pickle.load(open(os.path.join(CK, "ladder-rerun.pkl"), "rb"))
    G = pickle.load(open(os.path.join(CK, "ladder-GAMED-rerun.pkl"), "rb"))
    rows = []
    for k in range(9):
        mf = [FP.parse_name(n) for c in M["cert"] if c["level"] == k for n in c["points"]]
        gf = [FP.parse_name(n) for c in G["cert"] if c["level"] == k for n in c["points"]]
        r = {"level": k, "main_self_dev": strict_self([M["ev"][p] for p in mf]),
             "gamed_self_test": strict_self([G["ev"][p] for p in gf])}
        rows.append(r)
        print(r, flush=True)
    ok = all(r["main_self_dev"] == 0 and r["gamed_self_test"] == 0 for r in rows)
    Log().add("gap:control-own-split", ["measured", "fixpoint", "control"], premises=[cid, rid],
              receipt={"levels": rows, "holds": ok})
    print("holds", ok)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
