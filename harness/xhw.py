"""Cross-hardware audit of the certified rule (tasks/XHW_CONTRACT.json).

  python3 xhw.py ref OUTDIR              R only: reference trajectory and score vectors
  python3 xhw.py run OUTDIR MACHINE [REFDIR] [--configs ALL|reference] [--lean]
  python3 xhw.py compare OUTFILE DIR...         score every machine and config against R

A run writes OUTDIR/<machine>-<config>.json per config: per teacher-forced
step the sha256 of the float32 logits, the certified pick and margin, and,
when REFDIR holds R's score vectors, the largest |score difference| and the
number of differing entries; plus the free generations of every method.
"""
import glob, hashlib, json, os, platform, subprocess, sys, time
import numpy as np
from common import ROOT, MODELS, MODEL_FILE, PINS, Log, encode, qscores, sha256_file
import decoders as D

PROMPTS = json.load(open(os.path.join(ROOT, "tasks", "usecase_prompts.json")))["prompts"]
STEPS = 24
V = 50257
METHODS = ["prlm-0", "greedy-f32", "samp-seeded-t0.7", "samp-seeded-t1.0"]
MODEL_SHA = "12fbb1ec2d2d70c8ebd21a290a348a4109447b98582af64c6f93b6526d5c8f35"
B = 4  # declared in the contract, from the use-case U2 record
CONFIGS = [(4, "all"), (4, "none"), (1, "all"), (1, "none")]
REF_CONFIG = "t4-all"


def cname(threads, opt):
    return f"t{threads}-{opt}"


class Ort:
    def __init__(self, threads, opt):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if opt == "all"
                                       else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
        self.s = ort.InferenceSession(os.path.join(MODELS, MODEL_FILE), so, providers=["CPUExecutionProvider"])

    def logits(self, c):
        x = np.asarray(c[-1024:], dtype=np.int64).reshape(1, 1, -1)
        return self.s.run(["output1"], {"input1": x})[0][0, 0, -1].astype(np.float32).copy()


def decide(method, row, rng):
    if method == "prlm-0":
        return D.certify(qscores(row))[0]
    if method == "greedy-f32":
        return int(np.argmax(row))
    t = 0.7 if "t0.7" in method else 1.0
    lp = D.logsoftmax(row / t)
    return int(rng.choice(V, p=np.exp(lp)))


def free(M):
    out = {}
    for m in METHODS:
        res = []
        for i, p in enumerate(PROMPTS):
            rng = np.random.default_rng(i)
            c = list(encode(p))
            n0 = len(c)
            for _ in range(STEPS):
                c.append(decide(m, M.logits(c), rng))
            res.append(c[n0:])
        out[m] = res
    return out


def machine_info(name):
    import onnxruntime, numpy, tiktoken
    cpu = platform.processor()
    try:
        if sys.platform.startswith("linux"):
            cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith(("model name", "Model")))
        elif sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return {"machine": name, "platform": platform.platform(), "arch": platform.machine(), "cpu": cpu,
            "python": platform.python_version(), "onnxruntime": onnxruntime.__version__,
            "numpy": numpy.__version__, "tiktoken": tiktoken.__version__}


def check_inputs():
    got = {"model": sha256_file(os.path.join(MODELS, MODEL_FILE))}
    for k, v in PINS.items():
        got[k] = sha256_file(os.path.join(MODELS, k))
        assert got[k] == v, f"{k} sha256 mismatch"
    assert got["model"] == MODEL_SHA, "model sha256 mismatch"
    return got


def trajectory(refdir):
    return json.load(open(os.path.join(refdir, "reference.json")))["prlm-0"]


def teacher(M, traj, ref, lean_path=None):
    rows, k = [], 0
    rp = D.Replay(lean_path) if lean_path else None
    for i, p in enumerate(PROMPTS):
        c = list(encode(p))
        for t in range(STEPS):
            f = M.logits(c)
            q = qscores(f)
            v, s, m, ok = D.certify(q)
            row = {"f": hashlib.sha256(f.tobytes()).hexdigest()[:16], "pick": v, "margin": m}
            if ref is not None:
                d = np.abs(q - ref[k].astype(np.int64))
                row["maxdiff"], row["ndiff"] = int(d.max()), int((d > 0).sum())
            if rp:
                rp.write(q, [], [], [], v, v)
            rows.append(row)
            c.append(traj[i][t])
            k += 1
    if rp:
        rp.close()
    return rows


def load_ref(refdir):
    """R's score vectors: the raw .npy on R, or the lossless packed .npz in the repo
    (row maximum as int32, distance below it as uint16)."""
    p = os.path.join(refdir, "reference-scores.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="r")
    p = os.path.join(refdir, "reference-scores.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["rowmax"][:, None].astype(np.int64) - z["below"].astype(np.int64)
    return None


def cmd_compare(outfile, dirs):
    runs = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            r = json.load(open(p))
            if "teacher" in r and "config" in r and "key" not in r:
                r["_repeat"] = "again" in os.path.relpath(p, d)
                runs.append(r)
    ref = next(r for r in runs if r["machine"]["machine"] == "R" and r["config"] == REF_CONFIG and not r["_repeat"])
    rt = ref["teacher"]
    rows = []
    for r in runs:
        tf = r["teacher"]
        rows.append({"machine": r["machine"]["machine"], "cpu": r["machine"]["cpu"], "config": r["config"], "repeat": r["_repeat"],
                     "float_bitwise_steps": sum(a["f"] == b["f"] for a, b in zip(tf, rt)),
                     "score_equal_steps": sum(a.get("maxdiff") == 0 for a in tf),
                     "max_score_diff": max(a.get("maxdiff", 0) for a in tf),
                     "pick_disagree": sum(a["pick"] != b["pick"] for a, b in zip(tf, rt)),
                     "pick_disagree_with_margin_gt_2B": sum(a["pick"] != b["pick"] and b["margin"] > 2 * B for a, b in zip(tf, rt)),
                     "free_identical": {m: sum(x == y for x, y in zip(r["free"][m], ref["free"][m])) for m in METHODS},
                     "lean": r.get("lean")})
    gamed = max(x["max_score_diff"] for x in rows)
    out = {"model": "gpt2-fp32-ort", "steps": len(rt), "declared_B": B,
           "steps_margin_gt_2B": sum(b["margin"] > 2 * B for b in rt),
           "bound_holds": all(x["max_score_diff"] <= B for x in rows),
           "x3_refuted": any(x["pick_disagree_with_margin_gt_2B"] > 0 for x in rows),
           "gamed_B_observed": gamed, "gamed_steps_margin_gt_2Bobs": sum(b["margin"] > 2 * gamed for b in rt),
           "rows": rows}
    json.dump(out, open(outfile, "w"), indent=1)
    print({k: out[k] for k in ("bound_holds", "x3_refuted", "gamed_B_observed", "steps_margin_gt_2B")})


def cmd_ref(outdir):
    os.makedirs(outdir, exist_ok=True)
    check_inputs()
    M = Ort(4, "all")
    gen = free(M)
    json.dump({"prlm-0": gen["prlm-0"], "machine": machine_info("R"), "config": REF_CONFIG},
              open(os.path.join(outdir, "reference.json"), "w"))
    traj = gen["prlm-0"]
    S = np.zeros((len(PROMPTS) * STEPS, V), dtype=np.int32)
    k = 0
    for i, p in enumerate(PROMPTS):
        c = list(encode(p))
        for t in range(STEPS):
            S[k] = qscores(M.logits(c)).astype(np.int32)
            c.append(traj[i][t])
            k += 1
    np.save(os.path.join(outdir, "reference-scores.npy"), S)
    print("ref", sha256_file(os.path.join(outdir, "reference-scores.npy")))


def cmd_run(outdir, machine, refdir, configs, lean):
    os.makedirs(outdir, exist_ok=True)
    pins = check_inputs()
    traj = trajectory(refdir)
    ref = load_ref(refdir)
    info = machine_info(machine)
    for th, opt in configs:
        cn = cname(th, opt)
        if os.path.exists(os.path.join(outdir, f"{machine}-{cn}.json")):
            print(machine, cn, "already done", flush=True)
            continue
        t0 = time.perf_counter()
        M = Ort(th, opt)
        lp = os.path.join(outdir, f"{machine}-{cn}.bin") if lean else None
        steps = teacher(M, traj, ref, lp)
        res = {"machine": info, "config": cn, "inputs": pins, "teacher": steps, "free": free(M)}
        if lp:
            exe = os.environ.get("PRLM_LEAN", os.path.join(ROOT, "lean", ".lake", "build", "bin", "prlm"))
            o = subprocess.run([exe, "decide", lp], capture_output=True, text=True)
            line = o.stdout.strip().splitlines()[-1] if o.stdout.strip() else ""
            res["lean"] = {"exit": o.returncode, "line": line}
            os.remove(lp)
        res["seconds"] = round(time.perf_counter() - t0, 1)
        json.dump(res, open(os.path.join(outdir, f"{machine}-{cn}.json"), "w"))
        print(machine, cn, res["seconds"], res.get("lean"), flush=True)


def main():
    a = sys.argv[1:]
    if a[0] == "ref":
        cmd_ref(a[1])
    elif a[0] == "run":
        configs = CONFIGS if "--configs" not in a or a[a.index("--configs") + 1] == "ALL" else [(4, "all")]
        pos = [x for x in a[1:] if not x.startswith("--") and x not in ("ALL", "reference")]
        refdir = pos[2] if len(pos) > 2 else os.path.join(ROOT, "out", "xhw")
        cmd_run(pos[0], pos[1], refdir, configs, "--lean" in a)
    elif a[0] == "compare":
        cmd_compare(a[1], a[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
