"""Audit equal-prefix GPT-2 choices separately from free generation."""
import argparse
import json
from pathlib import Path


def evaluate(base, variant):
    if not base or len(base) != len(variant):
        raise ValueError("scores must describe the same nonempty vocabulary")
    best = max(range(len(base)), key=lambda t: base[t])
    actual = max(range(len(variant)), key=lambda t: variant[t])
    delta = [b-a for a,b in zip(base, variant)]
    lo, hi = min(delta), max(delta)
    allowance = {j: base[best] - base[j] - (j < best)
                 for j in range(len(base)) if j != best}
    limit = min(allowance.values(), default=None)
    l_inf = max(abs(lo), abs(hi))
    absolute = limit is None or 2*l_inf <= limit
    interval = limit is None or hi-lo <= limit
    relative = all(delta[j]-delta[best] <= m for j,m in allowance.items())
    if relative != (best == actual) or (absolute and not interval) or (interval and not relative):
        raise RuntimeError("certificate and independent argmax disagree")
    return {"reference": best, "actual": actual, "minimum_allowance": limit,
            "minimum_drift": lo, "maximum_drift": hi,
            "symmetric_row": absolute, "interval_row": interval, "relative_row": relative}


def summarize(rows):
    fields = ("symmetric_row", "interval_row", "relative_row")
    return {k: {"steps": sum(x[k] for x in rows),
                "first_uncovered": next((i for i,x in enumerate(rows) if not x[k]), None)}
            for k in fields}


def run(output, asset_manifest):
    import numpy as np
    import platform
    import onnxruntime
    from common import MODELS, encode, qscores, sha256_file
    from models import GPT2Ort, GPT2Numpy
    from usecases import PROMPTS, STEPS, gen_A, gen_B
    if len(PROMPTS) != 40 or STEPS != 24:
        raise ValueError("not the declared 40 by 24 corpus")
    names = ("gpt2-lm-head-10.onnx", "gpt2-lm-head-10-int8.onnx", "encoder.json", "vocab.bpe")
    expected = json.loads(Path(asset_manifest).read_text())
    actual = {name:sha256_file(Path(MODELS)/name) for name in names}
    if actual != expected:
        raise ValueError("GPT-2 model or tokenizer hashes differ from requested pins")
    reference = GPT2Ort(threads=2)
    arms = [GPT2Numpy(), GPT2Ort("gpt2-lm-head-10-int8.onnx", 2, "gpt2-int8-ort")]
    records = {m.name: [] for m in arms}
    for i,prompt in enumerate(PROMPTS):
        pre = list(encode(prompt))
        if not pre or len(pre)+STEPS > 1024:
            raise ValueError("prompt outside the declared context")
        want = gen_A(reference, "prlm-0", i, pre)
        prefix = pre+want
        common = reference.all_logits(prefix)[len(pre)-1:len(pre)-1+STEPS]
        for arm in arms:
            if isinstance(arm, GPT2Numpy):
                other, _ = gen_B(arm, "prlm-0", i, pre)
                forced = arm.kv_rows(prefix, len(pre))
            else:
                other = gen_A(arm, "prlm-0", i, pre)
                forced = arm.all_logits(prefix)[len(pre)-1:len(pre)-1+STEPS]
            if len(forced) != STEPS or len(common) != STEPS:
                raise ValueError("missing teacher-forced rows")
            rows = [evaluate(qscores(ra).tolist(), qscores(rb).tolist())
                    for ra,rb in zip(common, forced)]
            if [r["reference"] for r in rows] != want:
                raise ValueError("reference batched scorer differs from free generation")
            agreement = all(r["relative_row"] for r in rows)
            # KV and batched ORT are intentionally different execution paths:
            # record disagreement with free generation, never prefilter on it.
            records[arm.name].append({"prompt_index":i, "prefix_tokens":pre,
                        "reference_tokens":want, "free_tokens":other,
                        "free_equal":want == other, "teacher_forced_equal":agreement,
                        "coverage":summarize(rows), "rows":rows})
    metrics = {}
    for arm, prompts in records.items():
        lo = min(r["minimum_drift"] for p in prompts for r in p["rows"])
        hi = max(r["maximum_drift"] for p in prompts for r in p["rows"])
        for p in prompts:
            for row in p["rows"]:
                for key,bound in (("symmetric_corpus",2*max(abs(lo),abs(hi))),
                                  ("interval_corpus",hi-lo)):
                    row[key] = row["minimum_allowance"] is None or bound <= row["minimum_allowance"]
        methods = [*prompts[0]["coverage"],"symmetric_corpus","interval_corpus"]
        metrics[arm] = {"corpus_minimum_drift":lo, "corpus_maximum_drift":hi,
            "free_equal":sum(p["free_equal"] for p in prompts),
            "teacher_forced_equal":sum(p["teacher_forced_equal"] for p in prompts),
            "coverage":{k:{"steps":sum(row[k] for p in prompts for row in p["rows"]),
                "complete_replies":sum(all(row[k] for row in p["rows"]) for p in prompts)}
                for k in methods}}
    meta={"platform":platform.platform(),"numpy":np.__version__,"onnxruntime":onnxruntime.__version__,
          "prompt_asset_sha256":sha256_file(Path(__file__).resolve().parents[1]/"tasks/usecase_prompts.json"),
          "assets":actual,
          "skipped_forwards":0}
    with Path(output).open("x") as out:
        json.dump({"run":meta,"metrics":metrics,"observations":records}, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--asset-manifest", required=True)
    args = parser.parse_args()
    run(args.output, args.asset_manifest)
