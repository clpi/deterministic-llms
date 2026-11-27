"""Value amendment 2 (tasks/VALUE_AMEND_2.json, record 4991): Qwen3-14B-Base Q4_K_M
through llama.cpp on the first 100 GSM8K TEST problems, with value.py's methods.

  python value_llama.py run OUTDIR --shard K --nshards N
  python value_llama.py bar OUTFILE DIR...      the 3-point bar over every value result file

The driver (harness/value_llama.cpp, path in $VALUE_LLAMA) keeps one sequence;
paths are decoded one at a time, each starting from the saved prompt state.
"""
import glob, json, os, subprocess, sys
import numpy as np
import value as V

REPO = "mradermacher/Qwen3-14B-Base-GGUF"
FILE = "Qwen3-14B-Base.Q4_K_M.gguf"
N_14B = 100
NCTX = 2048


class LlamaLM:
    def __init__(self, threads=4, path=None):
        if path is None:
            from huggingface_hub import hf_hub_download, HfApi
            rev = HfApi().model_info(REPO).sha
            path = hf_hub_download(REPO, FILE, revision=rev)
            self.inputs = {"repo": REPO, "file": FILE, "commit": rev, "weights_sha256": {FILE: V.sha256_file(path)},
                           "engine": "llama.cpp " + os.environ.get("LLAMA_TAG", "?")}
        else:
            self.inputs = {"repo": os.path.basename(path)}
        p = path
        exe = os.environ.get("VALUE_LLAMA", "value_llama")
        self.p = subprocess.Popen([exe, p, str(threads), str(NCTX)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.V = int(self._cmd(b"V\n").readline())
        self.forwards = 0

    def _cmd(self, b):
        self.p.stdin.write(b)
        self.p.stdin.flush()
        return self.p.stdout

    def _ok(self, c):
        r = self._cmd(c).readline().strip()
        if r != b"ok":
            raise RuntimeError(f"llama driver: {c!r} -> {r!r}, exit {self.p.poll()}")

    def _decode(self, toks):
        out = self._cmd(("D %d %s\n" % (len(toks), " ".join(map(str, toks)))).encode())
        buf = out.read(4 * self.V)
        if len(buf) != 4 * self.V:
            raise RuntimeError(f"llama driver returned {len(buf)} bytes, exit {self.p.poll()}")
        self.forwards += 1
        return np.frombuffer(buf, dtype="<f4").astype(np.float32).copy()

    def encode(self, text):
        b = text.encode("utf-8")
        parts = self._cmd(b"K %d\n" % len(b) + b).readline().split()
        return [int(x) for x in parts[1:1 + int(parts[0])]]

    def decode(self, ids):
        out = self._cmd(("T %d %s\n" % (len(ids), " ".join(map(str, ids)))).encode())
        n = int(out.readline())
        return out.read(n).decode("utf-8", errors="replace")

    # value.py's model interface: one live sequence, restored to the prompt for each path
    def fork(self, state):
        self._ok(b"L\n")
        return state

    def feed(self, past, ids):
        return self._decode(list(ids)), past

    def extend(self, state, ids):
        if state is None:                 # the prompt: clear, decode, save
            self._ok(b"R\n")
            row = self._decode(list(ids))
            self._ok(b"S\n")
            return row, "prompt"
        self._ok(b"L\n")
        return self._decode(list(ids)), state


def run(outdir, shard, nshards, threads=4):
    os.makedirs(outdir, exist_ok=True)
    probs = [p for k, p in enumerate(V.load_test(None)[:N_14B]) if k % nshards == shard]
    lm = LlamaLM(threads)
    path = os.path.join(outdir, f"gsm14b-shard{shard}.jsonl")
    done = {json.loads(l)["index"] for l in open(path) if l.strip()} if os.path.exists(path) else set()
    info = {"task": "gsm14b", "machine": V.machine(), "inputs": lm.inputs}
    for pr in probs:
        if pr["index"] in done:
            continue
        r = V.solve(lm, pr["question"], pr["gold"], pr["index"])
        r.update(info)
        r["data_sha256"] = pr.get("data_sha256")
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(pr["index"], {m: int(r["methods"][m]["correct"]) for m in V.METHODS}, flush=True)


def selftest(path):
    """Driver check on a tiny model: restoring the saved prompt state gives the
    same logits as decoding straight after the prompt, and text round-trips."""
    lm = LlamaLM(2, path)
    ids = lm.encode("Once upon a time there was a little")
    row0, st = lm.extend(None, ids)
    a, _ = lm.feed(st, [int(np.argmax(row0))])
    b0, _ = lm.feed(st, [int(np.argmax(a))])
    lm.fork(st)
    a2, _ = lm.feed(st, [int(np.argmax(row0))])
    b2, _ = lm.feed(st, [int(np.argmax(a2))])
    c, _ = lm.extend(st, [int(np.argmax(row0))])
    ok = {"restore_exact": bool((a == a2).all() and (b0 == b2).all()), "extend_after_restore_exact": bool((c == a).all()),
          "decode": lm.decode(ids), "vocab": lm.V}
    print(json.dumps(ok))
    assert ok["restore_exact"] and ok["extend_after_restore_exact"] and "upon a time" in ok["decode"]


def bar(outfile, dirs):
    """VALUE_AMEND_2's bar for every (task, model) found in the result files."""
    rows = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
            if "again" in os.path.abspath(p):
                continue
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    rows.setdefault((r.get("task", "gsm8k"), r["inputs"]["repo"]), {})[r["index"]] = r
    out = {}
    for (task, repo), R in sorted(rows.items()):
        idx = sorted(R)
        if task == "mbpp":
            key, b, s, rnd = "pass", "margin-branch-8-filter", "bo8-filter", "random-branch-8-filter"
        else:
            key, b, s, rnd = "correct", "margin-branch-8", "sc-8", "random-branch-8"
        col = lambda m: [R[i]["methods"][m][key] for i in idx]
        vs, vg, vr = V.verdict(col(b), col(s)), V.verdict(col(b), col("greedy")), V.verdict(col(b), col(rnd))
        if vs["verdict"] == "LOSS" or vg["verdict"] == "LOSS":
            v = "LOSS"
        elif vs["verdict"] == "WIN" and vg["verdict"] == "WIN" and vr["verdict"] == "WIN":
            v = "DECISIVE WIN" if vs["diff"] >= 0.03 and vg["diff"] >= 0.03 else "SMALL WIN"
        elif vs["verdict"] == "WIN" or vg["verdict"] == "WIN":
            v = "SMALL WIN" if vr["verdict"] == "WIN" else "TIE (gain not from the margin)"
        else:
            v = "TIE"
        out[f"{task}:{repo}"] = {"problems": len(idx), "vs_sampling": vs, "vs_greedy": vg, "vs_random": vr, "bar": v}
    json.dump(out, open(outfile, "w"), indent=1)
    print(json.dumps(out))


def main():
    a = sys.argv[1:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    if a[0] == "run":
        run(a[1], int(opt("--shard", 0)), int(opt("--nshards", 1)), int(opt("--threads", 4)))
    elif a[0] == "selftest":
        selftest(a[1])
    elif a[0] == "bar":
        bar(a[1], a[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
