"""Paired sealed GPT-2 head-projection comparison with unchanged full oracle."""
import argparse
import itertools
import json
import platform
import random
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from common import encode, sha256_file
from models import GPT2Numpy,GPT2Ort
from gpt2_last_projection import identity
from live_verified_draft import choose,serial,verify
from usecases import PROMPTS,STEPS


def run_kv(model,pre,project_last):
    out,trace,past=[],[],None
    ids=np.array(pre,dtype=np.int64)
    for _ in range(STEPS):
        rows,past=model._fwd(ids,past,last_only=project_last)
        row=choose(rows[-1])
        out.append(row[0])
        trace.append(row)
        ids=np.array([row[0]],dtype=np.int64)
    return out,trace


def main(x):
    if (len(PROMPTS),STEPS)!=(40,24):
        raise ValueError("wrong sealed corpus")
    root=Path(__file__).resolve().parents[1]
    prompt_sha=sha256_file(root/"tasks/usecase_prompts.json")
    if prompt_sha!="13bff604cf086777136e80b7c41c4248e56eb797556008cac8d55e35bc0d63b8":
        raise ValueError("prompt hash drift")
    import os,importlib.metadata
    assets=Path(os.environ["PRLM_MODELS"])
    meta=json.loads(x.manifest.read_text())
    for key in ("gpt2-lm-head-10.onnx","encoder.json","vocab.bpe"):
        if identity(assets/key)!=meta[key]:
            raise ValueError("asset pin drift")
    transformed=assets/x.last
    if identity(transformed)!=x.last_sha:
        raise ValueError("transformed pin drift")
    full,last,kv=GPT2Ort(threads=2),GPT2Ort(x.last,threads=2),GPT2Numpy()
    modes=("serial_full","serial_last","kv_last","verify_full","verify_last")

    def evaluate(mode,pre):
        bfull,blast=full.forwards,last.forwards
        kv_calls=0
        pos_kv=0
        pos_ort=0
        hkv=0
        hort=0
        rejected=False
        start=perf_counter_ns()
        if mode.startswith("serial"):
            out,trace,pos_ort=serial(last if mode=="serial_last" else full,pre)
            hort=STEPS if mode=="serial_last" else pos_ort
        else:
            out,trace=run_kv(kv,pre,mode!="verify_full")
            kv_calls=STEPS
            pos_kv=len(pre)+STEPS-1
            hkv=pos_kv if mode=="verify_full" else STEPS
            if mode.startswith("verify"):
                out,trace,pos_ort,rejected=verify(full,pre,out)
                hort=pos_ort
        elapsed=perf_counter_ns()-start
        return dict(out=out,trace=trace,ns=elapsed,ort_calls=full.forwards-bfull+last.forwards-blast,
                    kv_calls=kv_calls,ort_positions=pos_ort,kv_positions=pos_kv,
                    ort_head_rows=hort,kv_head_rows=hkv,fallback=rejected)

    # Warm each mode once; order is later balanced by cyclic rotation of a shuffled base.
    warm=encode(PROMPTS[0])
    for mode in modes:
        evaluate(mode,warm)
    rng=random.Random(x.seed)
    prompts=list(range(len(PROMPTS)))
    rng.shuffle(prompts)
    order=list(modes)
    rng.shuffle(order)
    rows=[]
    for i,j in enumerate(prompts):
        pre=encode(PROMPTS[j])
        serial_order=order[i%5:]+order[:i%5]
        arms={name:evaluate(name,pre) for name in serial_order}
        rows.append(dict(prompt=j,prefix=pre,order=serial_order,arms=arms))
        print(f"run {i+1}/{len(PROMPTS)}",flush=True)
    chosen=rows[0]
    draft=chosen["arms"]["serial_full"]["out"][:]
    draft[10]=(draft[10]+1)%full.V
    out,cert,_,refused=verify(full,chosen["prefix"],draft)
    if not refused or out!=chosen["arms"]["serial_full"]["out"] or cert!=chosen["arms"]["serial_full"]["trace"]:
        raise ValueError("negative gate failed")
    with x.output.open("x") as f:
        json.dump(dict(scope=dict(platform=platform.platform(),seed=x.seed,prompt_sha=prompt_sha,
            last_sha=x.last_sha,source_assets=meta,model_versions={k:importlib.metadata.version(k) for k in
            ("onnx","onnxruntime","numpy","tiktoken")},warmups_per_arm=1,steps=STEPS),
            tamper_refused=refused,rows=rows),f,indent=2)
        f.write("\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--last",required=True)
    p.add_argument("--last-sha",required=True)
    p.add_argument("--seed",type=int,default=20261122)
    main(p.parse_args())
