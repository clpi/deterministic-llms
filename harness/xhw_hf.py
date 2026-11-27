"""Cross-hardware audit on current open models (tasks/XHW_AMEND_1.json).
Runs in CI, where the models can be downloaded; nothing here runs on R.

  python xhw_hf.py traj KEY OUTDIR                     reference machine: commit, trajectory, score vectors
  python xhw_hf.py run KEY MACHINE REFDIR OUTDIR [--configs ALL|reference] [--lean]
  python xhw_hf.py compare OUTFILE DIR...              summary over every machine and model

Per config the run writes OUTDIR/<KEY>-<MACHINE>-<config>.json: per teacher-forced
step the sha256 of the float32 logits, the certified pick and margin, the largest
|integer score difference| from the reference and how many entries differ; the
free generations of every method; the commit and weight sha256s it ran.
"""
import glob, hashlib, json, os, platform, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROMPTS = json.load(open(os.path.join(ROOT, "tasks", "usecase_prompts.json")))["prompts"]
STEPS = 24
QSCALE = 256
B = 4
METHODS = ["prlm-0", "greedy-f32", "samp-seeded-t0.7", "samp-seeded-t1.0"]
MODELS = {
    "Q35": "Qwen/Qwen3.5-0.8B-Base",
    "Q3": "Qwen/Qwen3-0.6B-Base",
    "LFM2": "LiquidAI/LFM2-350M",
    "G4H": "ibm-granite/granite-4.0-h-350m-base",
    "MOE": "ibm-granite/granite-3.1-1b-a400m-base",
}
CONFIGS = [4, 1]


def qscores(f):
    return np.rint(f.astype(np.float64) * QSCALE).astype(np.int64)


def certify(q):
    v = int(np.argmax(q))
    s = int(q[v])
    r = q.copy()
    r[v] = np.iinfo(np.int64).min
    return v, s - int(r.max())


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fetch(key, revision=None):
    from huggingface_hub import snapshot_download, HfApi
    repo = MODELS[key]
    rev = revision or HfApi().model_info(repo).sha
    d = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
    files = {os.path.relpath(p, d): sha256_file(p) for p in sorted(glob.glob(os.path.join(d, "*.safetensors")))}
    return d, {"repo": repo, "commit": rev, "weights_sha256": files}


class LM:
    def __init__(self, d, threads):
        import torch
        from transformers import AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(d)
        kw = dict(dtype=torch.float32, attn_implementation="eager")
        try:
            from transformers import AutoModelForCausalLM
            self.m = AutoModelForCausalLM.from_pretrained(d, **kw)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText
            self.m = AutoModelForImageTextToText.from_pretrained(d, **kw)
        self.m.eval()

    def encode(self, text):
        return list(self.tok(text).input_ids)

    def start(self, ids):
        with self.torch.inference_mode():
            o = self.m(input_ids=self.torch.tensor([ids]), use_cache=True)
        return o.logits[0, -1].float().numpy().copy(), o.past_key_values

    def step(self, tok, past):
        with self.torch.inference_mode():
            o = self.m(input_ids=self.torch.tensor([[tok]]), past_key_values=past, use_cache=True)
        return o.logits[0, -1].float().numpy().copy(), o.past_key_values


def decide(method, row, rng):
    if method == "prlm-0":
        return certify(qscores(row))[0]
    if method == "greedy-f32":
        return int(np.argmax(row))
    t = 0.7 if "t0.7" in method else 1.0
    p = np.exp(logsoftmax(row / t))
    return int(rng.choice(len(row), p=p / p.sum()))


def generate(M, method, i, prompt):
    rng = np.random.default_rng(i)
    row, past = M.start(M.encode(prompt))
    out = []
    for t in range(STEPS):
        v = decide(method, row, rng)
        out.append(v)
        if t + 1 < STEPS:
            row, past = M.step(v, past)
    return out


def forced(M, traj, ref=None, lean_path=None):
    rows, k = [], 0
    lf = open(lean_path, "wb") if lean_path else None
    for i, p in enumerate(PROMPTS):
        row, past = M.start(M.encode(p))
        for t in range(STEPS):
            q = qscores(row)
            v, m = certify(q)
            r = {"f": hashlib.sha256(row.tobytes()).hexdigest()[:16], "pick": v, "margin": m}
            if ref is not None:
                d = np.abs(q - ref[k].astype(np.int64))
                r["maxdiff"], r["ndiff"] = int(d.max()), int((d > 0).sum())
            if lf:   # Main.lean step record: n k, base, theta[], delta[], nf=0, picks
                lf.write(np.asarray([len(q), 0], dtype="<u4").tobytes())
                lf.write(q.astype("<i4").tobytes())
                lf.write(np.asarray([0], dtype="<u4").tobytes())
                lf.write(np.asarray([v, v], dtype="<i4").tobytes())
            rows.append(r)
            k += 1
            if t + 1 < STEPS:
                row, past = M.step(traj[i][t], past)
    if lf:
        lf.close()
    return rows


def machine_info(name):
    import torch, transformers
    cpu = platform.processor()
    try:
        if sys.platform.startswith("linux"):
            cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith(("model name", "Model")))
        elif sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return {"machine": name, "platform": platform.platform(), "arch": platform.machine(), "cpu": cpu,
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__}


def cmd_traj(key, outdir):
    os.makedirs(outdir, exist_ok=True)
    d, inputs = fetch(key)
    M = LM(d, 4)
    traj = [generate(M, "prlm-0", i, p) for i, p in enumerate(PROMPTS)]
    rows = []
    for i, p in enumerate(PROMPTS):
        row, past = M.start(M.encode(p))
        for t in range(STEPS):
            rows.append(qscores(row).astype(np.int32))
            if t + 1 < STEPS:
                row, past = M.step(traj[i][t], past)
    np.save(os.path.join(outdir, f"{key}-ref.npy"), np.stack(rows))
    json.dump({"key": key, "inputs": inputs, "trajectory": traj, "machine": machine_info("ubuntu-24.04")},
              open(os.path.join(outdir, f"{key}-traj.json"), "w"))
    print("traj", key, inputs["commit"], flush=True)


def cmd_run(key, machine, refdir, outdir, configs, lean):
    os.makedirs(outdir, exist_ok=True)
    T = json.load(open(os.path.join(refdir, f"{key}-traj.json")))
    d, inputs = fetch(key, T["inputs"]["commit"])
    same = inputs == T["inputs"]
    rp = os.path.join(refdir, f"{key}-ref.npy")
    ref = np.load(rp, mmap_mode="r") if os.path.exists(rp) else None
    info = machine_info(machine)
    for th in configs:
        t0 = time.perf_counter()
        try:
            M = LM(d, th)
            lp = os.path.join(outdir, f"{key}-{machine}-t{th}.bin") if lean and th == 4 else None
            res = {"key": key, "machine": info, "config": f"t{th}", "inputs": inputs, "inputs_match_reference": same,
                   "teacher": forced(M, T["trajectory"], ref, lp),
                   "free": {m: [generate(M, m, i, p) for i, p in enumerate(PROMPTS)] for m in METHODS}}
            if lp:
                exe = os.environ.get("PRLM_LEAN", os.path.join(ROOT, "lean", ".lake", "build", "bin", "prlm"))
                o = subprocess.run([exe, "decide", lp], capture_output=True, text=True)
                res["lean"] = {"exit": o.returncode, "line": (o.stdout.strip().splitlines() or [""])[-1]}
                os.remove(lp)
        except Exception as e:   # reported UNMEASURED for this machine, with the error
            res = {"key": key, "machine": info, "config": f"t{th}", "error": f"{type(e).__name__}: {e}"[:2000]}
        res["seconds"] = round(time.perf_counter() - t0, 1)
        json.dump(res, open(os.path.join(outdir, f"{key}-{machine}-t{th}.json"), "w"))
        print(key, machine, th, res["seconds"], res.get("lean"), res.get("error", ""), flush=True)


def cmd_compare(outfile, dirs):
    runs = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            r = json.load(open(p))
            if "config" in r and "key" in r:
                r["_path"] = os.path.relpath(p, d)
                runs.append(r)
    out = {}
    for key in MODELS:
        R = [r for r in runs if r["key"] == key]
        ref = next((r for r in R if r["machine"]["machine"] == "ubuntu-24.04" and r["config"] == "t4"
                    and "again" not in r["_path"] and "teacher" in r), None)
        rows = []
        for r in R:
            row = {"machine": r["machine"]["machine"], "cpu": r["machine"].get("cpu"), "config": r["config"],
                   "repeat": "again" in r["_path"]}
            if "error" in r:
                row["error"] = r["error"]
                rows.append(row)
                continue
            tf = r["teacher"]
            row["inputs_match_reference"] = r.get("inputs_match_reference")
            if ref:
                rt = ref["teacher"]
                row["float_bitwise_steps"] = sum(a["f"] == b["f"] for a, b in zip(tf, rt))
                row["score_equal_steps"] = sum(a.get("maxdiff") == 0 for a in tf)
                row["max_score_diff"] = max(a.get("maxdiff", 0) for a in tf)
                row["pick_disagree"] = sum(a["pick"] != b["pick"] for a, b in zip(tf, rt))
                row["pick_disagree_with_margin_gt_2B"] = sum(a["pick"] != b["pick"] and b["margin"] > 2 * B for a, b in zip(tf, rt))
                row["steps_margin_gt_2B"] = sum(b["margin"] > 2 * B for b in rt)
                row["free_identical"] = {m: sum(x == y for x, y in zip(r["free"][m], ref["free"][m])) for m in METHODS}
            if "lean" in r:
                row["lean"] = r["lean"]
            rows.append(row)
        gamed = max((x.get("max_score_diff", 0) for x in rows), default=0)
        out[key] = {"repo": MODELS[key], "reference_found": ref is not None, "steps": len(PROMPTS) * STEPS,
                    "declared_B": B, "bound_holds": all(x.get("max_score_diff", 0) <= B for x in rows if "error" not in x),
                    "x3_refuted": any(x.get("pick_disagree_with_margin_gt_2B", 0) > 0 for x in rows),
                    "gamed_B_observed": gamed,
                    "gamed_steps_margin_gt_2Bobs": sum(b["margin"] > 2 * gamed for b in ref["teacher"]) if ref else None,
                    "rows": rows}
    json.dump(out, open(outfile, "w"), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("reference_found", "bound_holds", "x3_refuted", "gamed_B_observed")} for k, v in out.items()}))


def main():
    a = sys.argv[1:]
    if a[0] == "traj":
        cmd_traj(a[1], a[2])
    elif a[0] == "run":
        configs = [4] if "--configs" in a and a[a.index("--configs") + 1] == "reference" else CONFIGS
        pos = [x for x in a[1:] if not x.startswith("--") and x not in ("ALL", "reference")]
        cmd_run(pos[0], pos[1], pos[2], pos[3], configs, "--lean" in a)
    elif a[0] == "compare":
        cmd_compare(a[1], a[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
