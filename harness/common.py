"""Shared pieces: tokenizer, GPT-2 ONNX model, content-addressed logit cache,
and the hash-chained append-only log (records in the Canon.Rec shape:
id, level, colors, premises, head, receipt, prev).

Nothing here knows about axes or tasks.
"""
import hashlib, json, os, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get("PRLM_MODELS", "/tmp/claude-0/models")
LOG = os.environ.get("PRLM_LOG", os.path.join(ROOT, "out", "log.jsonl"))

MODEL_FILE = "gpt2-lm-head-10.onnx"
PINS = {  # sha256 of every input the model path depends on
    "encoder.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    "vocab.bpe": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
}
QSCALE = 256  # integer score = round(logit * QSCALE)


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- tokenizer
_enc = None


def tokenizer():
    global _enc
    if _enc is None:
        import tiktoken
        from tiktoken.load import data_gym_to_mergeable_bpe_ranks
        ranks = data_gym_to_mergeable_bpe_ranks(
            vocab_bpe_file=os.path.join(MODELS, "vocab.bpe"),
            encoder_json_file=os.path.join(MODELS, "encoder.json"),
            vocab_bpe_hash=PINS["vocab.bpe"],
            encoder_json_hash=PINS["encoder.json"],
        )
        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        _enc = tiktoken.Encoding("gpt2-local", pat_str=pat, mergeable_ranks=ranks,
                                 special_tokens={"<|endoftext|>": 50256})
    return _enc


def encode(s):
    return tokenizer().encode(s, allowed_special=set())


def decode(ids):
    return tokenizer().decode(list(ids))


# ---------------------------------------------------------------- model
class Model:
    """GPT-2 small, LM head, ONNX. logits(prefix) is a pure function of the
    token prefix; results are cached by the prefix's content hash, which is
    the hypergraph node id of that content."""

    def __init__(self, threads=4):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(MODELS, MODEL_FILE), so,
                                         providers=["CPUExecutionProvider"])
        self.cache = {}
        self.forwards = 0
        self.forward_s = 0.0
        self.threads = threads

    @staticmethod
    def key(prefix):
        return sha256_bytes(np.asarray(prefix, dtype=np.int64).tobytes())[:24]

    def logits(self, prefix, use_cache=True):
        k = self.key(prefix)
        if use_cache and k in self.cache:
            return self.cache[k]
        x = np.asarray(prefix, dtype=np.int64)[-1024:].reshape(1, 1, -1)
        t = time.perf_counter()
        out = self.sess.run(["output1"], {"input1": x})[0][0, 0, -1].copy()
        self.forward_s += time.perf_counter() - t
        self.forwards += 1
        if use_cache:
            self.cache[k] = out
        return out


def qscores(logits):
    """Integer base scores. Deterministic given the float32 bits."""
    return np.rint(logits.astype(np.float64) * QSCALE).astype(np.int64)


# ---------------------------------------------------------------- log
class Log:
    """Append-only, hash-chained. Each line carries the sha256 of the line
    before it. Records are never edited; a correction is a new record."""

    def __init__(self, path=LOG):
        self.path = path
        self.prev = "0" * 64
        self.next_id = 0
        if os.path.exists(path):
            with open(path, "rb") as f:
                for line in f:
                    self.prev = sha256_bytes(line.rstrip(b"\n"))
                    self.next_id += 1

    def add(self, head, colors=(), premises=(), receipt=None, level=0):
        rec = {"id": self.next_id, "level": level, "colors": list(colors),
               "premises": list(premises), "head": head,
               "receipt": receipt or {}, "prev": self.prev}
        line = json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()
        with open(self.path, "ab") as f:
            f.write(line + b"\n")
        self.prev = sha256_bytes(line)
        self.next_id += 1
        return rec["id"]


def verify_log(path=LOG):
    prev = "0" * 64
    n = 0
    with open(path, "rb") as f:
        for line in f:
            line = line.rstrip(b"\n")
            rec = json.loads(line)
            if rec["prev"] != prev or rec["id"] != n:
                raise SystemExit(f"log chain broken at line {n}")
            prev = sha256_bytes(line)
            n += 1
    return n, prev
