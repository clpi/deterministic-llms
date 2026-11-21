"""On-the-fly weighting, selected on DEV and scored once on TEST.

Written at 11:50Z after 7 of 35 T1 prompts had been scored (prlm-theta
cut repetition and also cut reference match). So the candidates below are
post hoc. To keep that from leaking: candidates are compared on DEV
prompts built from TRAIN-corpus documents (never TEST), and every
candidate, chosen or not, is also scored on TEST and reported.

A content-dependent weight is still a named axis: gate(c) * phi(c, v) is a
feature of the edge c -> c.v, so every theorem in Prlm.Core applies.
"""
import json, os
import numpy as np
from common import *
import axes as AX, decoders as D
import run as R  # shares the log, the model and rec()

CANDS = {
    "noloop4": (dict(noloop=4.0), False),
    "rep0.5": (dict(rep=0.5), False),
    "rep1.5+noloop4": (dict(rep=1.5, noloop=4.0), False),
    "gated:rep1.5+noloop4": (dict(rep=1.5, noloop=4.0), True),
}


_ORIG = AX.features


def gated_features(content, gen_start, req, active):
    g = content[gen_start:]
    tri = [tuple(g[i:i + 3]) for i in range(len(g) - 2)]
    if len(set(tri)) == len(tri):  # no repeated 3-gram yet: axes off
        return []
    return _ORIG(content, gen_start, req, active)


def gen(item, theta, gated):
    req = AX.Request(item)
    f = AX.features
    if gated:
        AX.features = gated_features
    try:
        R.fresh()
        f0 = R.M.forwards
        g, info = D.prlm(R.M, item["prompt"], 24, req, AX.weights(**theta), ["rep", "noloop"])
    finally:
        AX.features = f
    info["forwards"] = R.M.forwards - f0
    return g, info


def main():
    dev = [{"id": f"dev/{c['module']}", "prompt": c["ids"][:32], "reference": c["ids"][32:56]}
           for c in R.T["corpus"] if len(c["ids"]) >= 56][:20]
    R.L.add("adapt:declared", ["declared"], receipt={
        "attempt": "2 (attempt 1 crashed with RecursionError: the gate called the patched function; its records stay in the log and are superseded key by key)",
        "candidates": {k: [v[0], v[1]] for k, v in CANDS.items()}, "dev_items": len(dev),
        "adapt_sha256": sha256_file(os.path.abspath(__file__)),
        "note": "written after 7/35 T1 prompts were scored; selection on DEV only; all candidates scored on TEST"})
    for it in dev:
        g, info = R.run_one(D.greedy, R.M, it["prompt"], 24)
        R.rec("dev:greedy", it["id"], 0, g, info)
        for name, (th, gated) in CANDS.items():
            g, info = gen(it, th, gated)
            R.rec("dev:" + name, it["id"], 0, g, info)
    for it in R.T["t1"]:
        for name, (th, gated) in CANDS.items():
            g, info = gen(it, th, gated)
            R.rec("adapt:" + name, it["id"], 0, g, info)
        print("adapt", it["id"], flush=True)


if __name__ == "__main__":
    main()
