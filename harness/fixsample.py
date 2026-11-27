"""Lean replay of a declared random sample of the level 10 front
(tasks/FIXPOINT_AMEND_4.json).

  python3 fixsample.py AMEND4_RECORD_ID
"""
import json, os, pickle, sys
import numpy as np
import loop as LP
import fixpoint as FP
from common import LOG, Log

N, SEED, RESULT = 600, 4972, 4972
PART = os.path.join(os.environ.get("LADDER_CKPT_DIR", "/mnt/project-files/prlm-ckpt"), "fixsample-part.jsonl")


def main(cid):
    names = None
    for line in open(LOG):
        r = json.loads(line)
        if r["id"] == RESULT:
            names = sorted(r["receipt"]["finest_front_points"])
            break
    pick = sorted(np.random.default_rng(SEED).choice(names, size=N, replace=False).tolist())
    L = Log()
    declared = [json.loads(l) for l in open(LOG) if '"ladder:lean:sample:declared"' in l]
    if declared:   # resumed after a restart: the sample was already declared
        assert declared[0]["receipt"]["points"] == pick
    else:
        L.add("ladder:lean:sample:declared", ["declared", "lean-replay"], premises=[cid, RESULT],
              receipt={"front": len(names), "n": N, "seed": SEED, "points": pick})
    steps = pickle.load(open(FP.STEPS, "rb"))
    done = {}
    if os.path.exists(PART):
        for l in open(PART):
            k, v = json.loads(l)
            done[k] = v
    todo = [n for n in pick if n not in done]
    for i in range(0, len(todo), 8):   # results kept per point, so a restart loses at most 8
        part = LP.lean_replay([FP.parse_name(n) for n in todo[i:i + 8]], steps)
        with open(PART, "a") as f:
            for k, v in part.items():
                f.write(json.dumps([k, v]) + "\n")
        done.update(part)
        print("sample", len(done), "of", N, flush=True)
    lean = {n: done[n] for n in pick}
    ok = all(v["exit"] == 0 and v["agree"] == v["sparseEq"] == v["steps"] == v["agreeRe"] == v["reEqRec"] for v in lean.values())
    L.add("ladder:lean:sample", ["lean-replay", "measured", "fixpoint"], premises=[cid, RESULT],
          receipt={"level": 10, "front": len(names), "n": len(lean), "all_exact": ok, "points": lean})
    print("sample", len(lean), ok, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
