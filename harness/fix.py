"""Correction run for a sign error in the declared ban weights.

The ban axis's feature is -1 on banned tokens. out/DECLARED.json gave it
weights -4000 and -8, so the product was +4000 and +8: the declared runs
BOOSTED banned words (outputs like ' to the the the ...'). Those records
stay in the log as they are. This run uses +4000 and +8 under new system
names and is marked as a correction. Nothing else changes.
"""
import os
from common import *
import axes as AX, decoders as D
import run as R


def main():
    R.L.add("correction:ban-sign", ["correction", "declared"], receipt={
        "corrects": "prlm-theta:ban and prlm-theta:ban-soft in the t3 phase",
        "bug": "feature -1 times declared weight -4000/-8 boosted banned tokens",
        "fix": "weights +4000 and +8 under systems prlm-theta:ban-fixed and prlm-theta:ban-soft-fixed",
        "fix_sha256": sha256_file(os.path.abspath(__file__))})
    for it in R.T["t3"]:
        if it["kind"] != "ban":
            continue
        for nm, w in (("ban-fixed", 4000.0), ("ban-soft-fixed", 8.0)):
            g, info = R.prlm_run(it, 24, AX.weights(ban=w), ["ban"])
            R.rec("prlm-theta:" + nm, it["id"], 0, g, info,
                  extra={"nll": D.nll_under_base(R.M, it["prompt"], g)}, colors=["correction"])
        print("fix", it["id"], flush=True)


if __name__ == "__main__":
    main()
