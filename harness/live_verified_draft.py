"""Paired GPT-2 verified-draft experiment; no inferred batch/serial equality."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from common import MODELS, encode, qscores, sha256_file
from models import GPT2Ort, GPT2Numpy
from usecases import PROMPTS, STEPS


def choose(row):
    scores = qscores(row)
    w = int(np.argmax(scores))
    before = int(scores[:w].max()) + 1 if w else None
    after = int(scores[w+1:].max()) if w+1 < len(scores) else None
    return [w, int(scores[w]), int(scores[w])-max(x for x in (before,after) if x is not None)]


def serial(target, pre):
    out, trace = [], []
    positions = 0
    for _ in range(STEPS):
        positions += len(pre)+len(out)
        row = choose(target.logits(pre+out))
        trace.append(row)
        out.append(row[0])
    return out, trace, positions


def make_draft(draft, pre):
    out, past = [], None
    positions = 0
    ids = np.asarray(pre,dtype=np.int64)
    for _ in range(STEPS):
        positions += len(ids)
        logits,past = draft._fwd(ids,past)
        token = choose(logits[-1])[0]
        out.append(token)
        ids = np.asarray([token],dtype=np.int64)
    return out,STEPS,positions


def verify(target, pre, proposed):
    inp = pre + proposed[:-1]
    rows = target.all_logits(inp)[len(pre)-1:]
    if len(rows) != STEPS:
        raise ValueError("wrong target cardinality")
    decisions = [choose(row) for row in rows]
    good = [x[0] for x in decisions] == proposed
    if good:
        return proposed,decisions,len(inp),False
    out,trace,positions = serial(target,pre)
    return out,trace,len(inp)+positions,True


def main(args):
    root = Path(__file__).resolve().parents[1]
    names=("gpt2-lm-head-10.onnx","encoder.json","vocab.bpe")
    pins=json.loads(args.manifest.read_text())
    asset={n:sha256_file(Path(MODELS)/n) for n in names}
    if {n:pins.get(n) for n in names} != asset:
        raise ValueError("asset identity mismatch")
    prompt_sha=sha256_file(root/"tasks/usecase_prompts.json")
    if prompt_sha!="13bff604cf086777136e80b7c41c4248e56eb797556008cac8d55e35bc0d63b8" or len(PROMPTS)!=40 or STEPS!=24:
        raise ValueError("corpus mismatch")
    ref,draft=GPT2Ort(threads=2),GPT2Numpy()
    pre=encode(PROMPTS[0])
    # Symmetric warm-up, excluded from per-prompt measurements.
    _=serial(ref,pre)
    prop,_,_=make_draft(draft,pre); _=verify(ref,pre,prop)
    rows=[]
    for i,prompt in enumerate(PROMPTS):
        pre=encode(prompt)
        def run_serial():
            before=ref.forwards
            start=time.perf_counter_ns()
            out,trace,positions=serial(ref,pre)
            ns=time.perf_counter_ns()-start
            return dict(out=out,trace=trace,positions=positions,calls=ref.forwards-before,ns=ns)
        def run_verified():
            before=ref.forwards
            start=time.perf_counter_ns()
            proposed,dc,dp=make_draft(draft,pre)
            draft_ns=time.perf_counter_ns()-start
            out,trace,positions,fallback=verify(ref,pre,proposed)
            ns=time.perf_counter_ns()-start
            return dict(out=out,trace=trace,positions=positions,calls=ref.forwards-before,ns=ns,
                        draft_ns=draft_ns,draft_calls=dc,draft_positions=dp,fallback=fallback)
        if i%2:
            v,s=run_verified(),run_serial()
        else:
            s,v=run_serial(),run_verified()
        rows.append(dict(prompt=i,prefix=pre,serial=s,verified=v,
                         reply_equal=s["out"]==v["out"],
                         certificate_equal=s["trace"]==v["trace"] and len(s["out"])==len(v["out"])==STEPS))
        print(f"prompt {i+1}/40",flush=True)
    pre=rows[0]["prefix"]
    alt=rows[0]["serial"]["out"][:]
    alt[10]=(alt[10]+1)%ref.V
    out,cert,_,refused=verify(ref,pre,alt)
    if not refused or out!=rows[0]["serial"]["out"] or cert!=rows[0]["serial"]["trace"]:
        raise ValueError("mutated-draft control failed")
    with args.output.open("x") as f:
        json.dump(dict(meta=dict(assets=asset,prompt_sha=prompt_sha,horizon=STEPS,
                                 platform=platform.platform(),ort_threads=2,mode="real-live-draft",
                                 initial_warmups=2),tamper_refused=refused,rows=rows),f,indent=2)
        f.write("\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("output",type=Path)
    main(p.parse_args())
