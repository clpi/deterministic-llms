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


def run(output, asset_manifest, int8_kind=None):
    import numpy as np
    import platform
    import time
    import onnxruntime
    from common import MODELS, encode, qscores, sha256_file
    from models import GPT2Ort, GPT2Numpy
    from usecases import PROMPTS, STEPS, gen_B
    if len(PROMPTS) != 40 or STEPS != 24:
        raise ValueError("not the declared 40 by 24 corpus")
    names = ["gpt2-lm-head-10.onnx", "encoder.json", "vocab.bpe"]
    if int8_kind is not None:
        names.append("gpt2-lm-head-10-int8.onnx")
    expected = json.loads(Path(asset_manifest).read_text())
    actual = {name:sha256_file(Path(MODELS)/name) for name in names}
    if actual != expected:
        raise ValueError("GPT-2 model or tokenizer hashes differ from requested pins")
    hist = "5031900c818c44a1ba8e1983785c18e78cc79451ad1a034ca3b96c05bdcafdff"
    if int8_kind == "historical" and actual[names[-1]] != hist:
        raise ValueError("not the historical INT8 artifact")
    reference = GPT2Ort(threads=2)
    numpy = GPT2Numpy()
    int8 = GPT2Ort(names[-1], 2, "gpt2-int8-ort") if int8_kind else None

    def free_step(model, pre):
        result, scores = [], []
        for _ in range(STEPS):
            row = qscores(model.logits(pre+result)).tolist()
            scores.append(row)
            result.append(max(range(len(row)), key=row.__getitem__))
        return result, scores

    def batched(model, prefix, n):
        return [qscores(x).tolist() for x in model.all_logits(prefix)[n-1:n-1+STEPS]]

    records = {"fp32_batch_vs_fp32_step": [], "fp32_numpy_kv_vs_fp32_step": []}
    if int8:
        records.update({"int8_step_vs_fp32_step": [], "int8_batch_vs_int8_step": []})
    begun = time.perf_counter()
    for i,prompt in enumerate(PROMPTS):
        pre = list(encode(prompt))
        if not pre or len(pre)+STEPS > 1024:
            raise ValueError("prompt outside the declared context")
        want, ref_scores = free_step(reference, pre)
        prefix = pre+want
        numpy_other, _ = gen_B(numpy, "prlm-0", i, pre)
        arms = [
            ("fp32_batch_vs_fp32_step", ref_scores, batched(reference, prefix, len(pre)), None),
            ("fp32_numpy_kv_vs_fp32_step", ref_scores,
             [qscores(x).tolist() for x in numpy.kv_rows(prefix, len(pre))], numpy_other),
        ]
        if int8:
            int8_other, _ = free_step(int8, pre)
            int8_common = [qscores(int8.logits(prefix[:len(pre)+j])).tolist() for j in range(STEPS)]
            arms.extend([
                ("int8_step_vs_fp32_step", ref_scores, int8_common, int8_other),
                ("int8_batch_vs_int8_step", int8_common, batched(int8, prefix, len(pre)), None),
            ])
        for name, base_rows, variant_rows, other in arms:
            if len(base_rows) != STEPS or len(variant_rows) != STEPS:
                raise ValueError("missing teacher-forced rows")
            rows = [evaluate(a,b) for a,b in zip(base_rows, variant_rows)]
            if name != "int8_batch_vs_int8_step" and [r["reference"] for r in rows] != want:
                raise ValueError("reference scorer differs from free generation")
            agreement = all(r["relative_row"] for r in rows)
            if other is not None and agreement != (want == other):
                raise ValueError("serial / KV free-generation trajectory differs from its prefix replay")
            records[name].append({"prompt_index":i, "prefix_tokens":pre,
                        "reference_tokens":want, "free_tokens":other,
                        "free_equal":None if other is None else want == other,
                        "teacher_forced_equal":agreement,
                        "coverage":summarize(rows), "rows":rows})
        print(f"completed prompt {i+1}/{len(PROMPTS)}", flush=True)
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
            "free_equal":None if prompts[0]["free_equal"] is None else sum(p["free_equal"] for p in prompts),
            "teacher_forced_equal":sum(p["teacher_forced_equal"] for p in prompts),
            "coverage":{k:{"steps":sum(row[k] for p in prompts for row in p["rows"]),
                "complete_replies":sum(all(row[k] for row in p["rows"]) for p in prompts)}
                for k in methods}}
    import importlib.metadata
    from datetime import datetime, timezone
    meta={"platform":platform.platform(),"numpy":np.__version__,"onnxruntime":onnxruntime.__version__,
          "runtime_versions":{k:importlib.metadata.version(k) for k in ("onnx","tiktoken","numpy","onnxruntime")},
          "completed_at":datetime.now(timezone.utc).isoformat(),
          "elapsed_s":time.perf_counter()-begun,"ort_cpu_intraop_threads":2,
          "int8_artifact_kind":int8_kind,"fp32_ort_calls":reference.forwards,
          "int8_ort_calls":None if int8 is None else int8.forwards,
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
    parser.add_argument("--int8-kind", choices=["historical","rebuilt"])
    args = parser.parse_args()
    run(args.output, args.asset_manifest, args.int8_kind)
