"""Inkling 2 Mini row of the cross-hardware audit (tasks/XHW_AMEND_2.json):
the smallest Unsloth GGUF through llama.cpp, on the Linux runners only.
It reuses xhw_hf.py's procedure unchanged and swaps in a llama.cpp model.

  python xhw_ink.py traj OUTDIR
  python xhw_ink.py run MACHINE REFDIR OUTDIR [--configs ALL|reference] [--lean]
  python xhw_ink.py compare OUTFILE DIR...
"""
import os, subprocess, sys
import numpy as np
import xhw_hf as H

KEY = "INK"
REPO = "unsloth/Inkling-2-Mini-GGUF"
FILE = "Mainline_Support/Inkling-2-Mini-UD-IQ1_S.gguf_file"


class LlamaLM:
    """Same interface as xhw_hf.LM, through harness/xhw_llama.cpp."""

    def __init__(self, path, threads):
        exe = os.environ.get("XHW_LLAMA", "xhw_llama")
        self.p = subprocess.Popen([exe, path, str(threads)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.V = int(self._cmd(b"V\n").readline())

    def _cmd(self, b):
        self.p.stdin.write(b)
        self.p.stdin.flush()
        return self.p.stdout

    def _logits(self, toks):
        out = self._cmd(("D %d %s\n" % (len(toks), " ".join(map(str, toks)))).encode())
        buf = out.read(4 * self.V)
        if len(buf) != 4 * self.V:
            raise RuntimeError(f"llama driver returned {len(buf)} bytes, exit {self.p.poll()}")
        return np.frombuffer(buf, dtype="<f4").astype(np.float32).copy()

    def encode(self, text):
        b = text.encode("utf-8")
        parts = self._cmd(b"K %d\n" % len(b) + b).readline().split()
        return [int(x) for x in parts[1:1 + int(parts[0])]]

    def start(self, ids):
        assert self._cmd(b"R\n").readline().strip() == b"ok"
        return self._logits(ids), None

    def step(self, tok, past):
        return self._logits([tok]), None


def fetch(key, revision=None):
    from huggingface_hub import hf_hub_download, HfApi
    rev = revision or HfApi().model_info(REPO).sha
    p = hf_hub_download(REPO, FILE, revision=rev)
    return p, {"repo": REPO, "commit": rev, "weights_sha256": {FILE: H.sha256_file(p)}}


def machine_info(name):
    import platform
    info = {"machine": name, "platform": platform.platform(), "arch": platform.machine(), "cpu": platform.processor(),
            "python": platform.python_version(), "numpy": np.__version__, "llama_cpp": os.environ.get("XHW_LLAMA_TAG")}
    try:
        info["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith(("model name", "Model")))
    except Exception:
        pass
    return info


H.MODELS = {KEY: REPO}
H.LM = LlamaLM
H.fetch = fetch
H.machine_info = machine_info


def main():
    a = sys.argv[1:]
    if a[0] == "traj":
        H.cmd_traj(KEY, a[1])
    elif a[0] == "run":
        configs = [4] if "--configs" in a and a[a.index("--configs") + 1] == "reference" else H.CONFIGS
        pos = [x for x in a[1:] if not x.startswith("--") and x not in ("ALL", "reference")]
        H.cmd_run(KEY, pos[0], pos[1], pos[2], configs, "--lean" in a)
    elif a[0] == "compare":
        H.cmd_compare(a[1], a[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
