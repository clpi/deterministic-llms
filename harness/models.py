"""Model adapters for the multi-model contract (tasks/MULTI_CONTRACT.json).

Every adapter exposes the same surface, so the rule, the replay and the
skipping code never know which model they run:

  V, eos, name
  start(prompt_ids) -> content        content the rule extends
  logits(content)   -> float32[V]     next-token logits, full recompute
  all_logits(content) -> float32[n,V] row i = logits after content[:i+1],
                                      ONE forward over the whole content
  kv_rows(content, n0) -> float32[k,V] (KV models only) rows for positions
                                      n0-1 .. len-1 via the cached decode
                                      path: prefill content[:n0], then one
                                      token at a time
  forwards                            model calls so far

The two "-random" models are MECHANICS ONLY: random weights from seed 0.
"""
import os
import numpy as np

from common import MODELS, encode, decode

GPT2_V = 50257


class Ort:
    def __init__(self, path, threads):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------- GPT-2 ONNX
class GPT2Ort:
    V, eos = GPT2_V, 50256

    def __init__(self, file="gpt2-lm-head-10.onnx", threads=4, name="gpt2-fp32-ort"):
        self.name, self.file = name, os.path.join(MODELS, file)
        self.s = Ort(self.file, threads).sess
        self.forwards = 0

    def start(self, prompt):
        return list(prompt)

    def all_logits(self, c):
        self.forwards += 1
        x = np.asarray(c[-1024:], dtype=np.int64).reshape(1, 1, -1)
        return self.s.run(["output1"], {"input1": x})[0][0, 0]

    def logits(self, c):
        return self.all_logits(c)[-1].copy()


# ---------------------------------------------------------------- GPT-2 numpy + KV
def _ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.float32(0.7978845608028654) * (x + np.float32(0.044715) * x ** 3)))


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


class GPT2Numpy:
    """The pinned GPT-2 weights, read from the ONNX initializers by name,
    run in numpy float32 with a KV cache."""
    V, eos = GPT2_V, 50256

    def __init__(self, file="gpt2-lm-head-10.onnx", name="gpt2-numpy-kv"):
        import onnx
        from onnx import numpy_helper
        m = onnx.load(os.path.join(MODELS, file))
        w = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in m.graph.initializer
             if i.name.startswith("transformer.")}
        self.w, self.name, self.L, self.H = w, name, 12, 12
        self.forwards = 0

    def start(self, prompt):
        return list(prompt)

    def _fwd(self, ids, past, last_only=False):
        w, H = self.w, self.H
        p0 = 0 if past is None else past[0][0].shape[1]
        x = w["transformer.wte.weight"][ids] + w["transformer.wpe.weight"][p0:p0 + len(ids)]
        new = []
        T = len(ids)
        for l in range(self.L):
            P = f"transformer.h.{l}."
            h = _ln(x, w[P + "ln_1.weight"], w[P + "ln_1.bias"])
            qkv = h @ w[P + "attn.c_attn.weight"] + w[P + "attn.c_attn.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(T, H, 64).transpose(1, 0, 2)
            k = k.reshape(T, H, 64).transpose(1, 0, 2)
            v = v.reshape(T, H, 64).transpose(1, 0, 2)
            if past is not None:
                k = np.concatenate([past[l][0], k], axis=1)
                v = np.concatenate([past[l][1], v], axis=1)
            new.append((k, v))
            S = k.shape[1]
            att = (q @ k.transpose(0, 2, 1)) * np.float32(0.125)
            mask = np.triu(np.ones((T, S), dtype=bool), k=S - T + 1)
            att = np.where(mask, np.float32(-1e10), att)
            y = (_softmax(att) @ v).transpose(1, 0, 2).reshape(T, 768)
            x = x + (y @ w[P + "attn.c_proj.weight"] + w[P + "attn.c_proj.bias"])
            h = _ln(x, w[P + "ln_2.weight"], w[P + "ln_2.bias"])
            h = _gelu(h @ w[P + "mlp.c_fc.weight"] + w[P + "mlp.c_fc.bias"])
            x = x + (h @ w[P + "mlp.c_proj.weight"] + w[P + "mlp.c_proj.bias"])
        x = _ln(x, w["transformer.ln_f.weight"], w["transformer.ln_f.bias"])
        if last_only:
            x = x[-1:]
        return (x @ w["transformer.wte.weight"].T).astype(np.float32), new

    def all_logits(self, c):
        self.forwards += 1
        return self._fwd(np.asarray(c, dtype=np.int64), None)[0]

    def logits(self, c):
        return self.all_logits(c)[-1].copy()

    def kv_rows(self, c, n0):
        ids = np.asarray(c, dtype=np.int64)
        lg, past = self._fwd(ids[:n0], None)
        rows = [lg[-1]]
        for t in range(n0, len(c)):
            lg, past = self._fwd(ids[t:t + 1], past)
            rows.append(lg[-1])
        return np.stack(rows)[:-1] if len(c) > n0 else np.stack(rows)


# ---------------------------------------------------------------- T5-base ONNX
class T5Ort:
    V, eos = 32128, 1

    def __init__(self, threads=4, name="t5-base-ort"):
        import sentencepiece as spm
        d = os.path.join(MODELS, "t5")
        self.enc = Ort(os.path.join(d, "t5-encoder-12.onnx"), threads).sess
        self.dec = Ort(os.path.join(d, "t5-decoder-with-lm-head-12.onnx"), threads).sess
        self.sp = spm.SentencePieceProcessor(model_file=os.path.join(d, "spiece.model"))
        self.name, self.forwards, self.h = name, 0, None

    def start(self, prompt):
        """prompt: GPT-2 ids; decoded to text, re-encoded, sentinel appended."""
        ids = self.sp.encode(decode(prompt))[-500:] + [32099, 1]
        self.h = self.enc.run(None, {"input_ids": np.array([ids], dtype=np.int64)})[0]
        return [0]

    def all_logits(self, c):
        self.forwards += 1
        return self.dec.run(None, {"input_ids": np.array([c], dtype=np.int64),
                                   "encoder_hidden_states": self.h})[0][0]

    def logits(self, c):
        return self.all_logits(c)[-1].copy()


# ---------------------------------------------------------------- Llama-style (random)
def _rms(x, g, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


def _silu(x):
    return x / (1 + np.exp(-x))


class LlamaRandom:
    """MECHANICS ONLY. RoPE, GQA (8 q heads, 2 kv heads), RMSNorm, SwiGLU."""
    V, eos = GPT2_V, 50256

    def __init__(self, d=256, layers=4, hq=8, hkv=2, ffn=688, seed=0, name="llama-style-random"):
        r = np.random.default_rng(seed)
        f = np.float32
        self.d, self.L, self.hq, self.hkv, self.hd = d, layers, hq, hkv, d // hq
        g = lambda *s: (r.standard_normal(s) / np.sqrt(s[0])).astype(f)
        self.emb = r.standard_normal((self.V, d)).astype(f)
        self.layers = [dict(n1=np.ones(d, f), wq=g(d, d), wk=g(d, hkv * self.hd), wv=g(d, hkv * self.hd),
                            wo=g(d, d), n2=np.ones(d, f), w1=g(d, ffn), w3=g(d, ffn), w2=g(ffn, d))
                       for _ in range(layers)]
        self.nf = np.ones(d, f)
        self.head = (r.standard_normal((d, self.V)) * (3.0 / np.sqrt(d))).astype(f)
        inv = 1.0 / (10000 ** (np.arange(0, self.hd, 2) / self.hd))
        ang = np.outer(np.arange(2048), inv)
        self.cos, self.sin = np.cos(ang).astype(f), np.sin(ang).astype(f)
        self.name, self.forwards = name, 0

    def start(self, prompt):
        return list(prompt)

    def _rope(self, x, p0):  # x: (h, T, hd)
        T = x.shape[1]
        c, s = self.cos[p0:p0 + T], self.sin[p0:p0 + T]
        a, b = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = a * c - b * s
        out[..., 1::2] = a * s + b * c
        return out

    def _fwd(self, ids, past):
        p0 = 0 if past is None else past[0][0].shape[1]
        x = self.emb[ids]
        T, new, rep = len(ids), [], self.hq // self.hkv
        for l, W in enumerate(self.layers):
            h = _rms(x, W["n1"])
            q = (h @ W["wq"]).reshape(T, self.hq, self.hd).transpose(1, 0, 2)
            k = (h @ W["wk"]).reshape(T, self.hkv, self.hd).transpose(1, 0, 2)
            v = (h @ W["wv"]).reshape(T, self.hkv, self.hd).transpose(1, 0, 2)
            q, k = self._rope(q, p0), self._rope(k, p0)
            if past is not None:
                k = np.concatenate([past[l][0], k], axis=1)
                v = np.concatenate([past[l][1], v], axis=1)
            new.append((k, v))
            S = k.shape[1]
            kk, vv = np.repeat(k, rep, axis=0), np.repeat(v, rep, axis=0)
            att = (q @ kk.transpose(0, 2, 1)) * np.float32(1 / np.sqrt(self.hd))
            att = np.where(np.triu(np.ones((T, S), bool), k=S - T + 1), np.float32(-1e10), att)
            y = (_softmax(att) @ vv).transpose(1, 0, 2).reshape(T, self.d)
            x = x + y @ W["wo"]
            h = _rms(x, W["n2"])
            x = x + (_silu(h @ W["w1"]) * (h @ W["w3"])) @ W["w2"]
        return (_rms(x, self.nf) @ self.head).astype(np.float32), new

    def all_logits(self, c):
        self.forwards += 1
        return self._fwd(np.asarray(c, dtype=np.int64), None)[0]

    def logits(self, c):
        return self.all_logits(c)[-1].copy()

    def kv_rows(self, c, n0):
        ids = np.asarray(c, dtype=np.int64)
        lg, past = self._fwd(ids[:n0], None)
        rows = [lg[-1]]
        for t in range(n0, len(c)):
            lg, past = self._fwd(ids[t:t + 1], past)
            rows.append(lg[-1])
        return np.stack(rows)[:-1] if len(c) > n0 else np.stack(rows)


# ---------------------------------------------------------------- Mamba-style (random)
class MambaRandom:
    """MECHANICS ONLY. Selective SSM: input-dependent dt, B, C; diagonal A;
    causal depthwise conv; the cache is the constant-size (conv window, h)."""
    V, eos = GPT2_V, 50256

    def __init__(self, d=256, layers=4, expand=2, N=16, K=4, seed=0, name="mamba-style-random"):
        r = np.random.default_rng(seed)
        f = np.float32
        E = expand * d
        self.d, self.E, self.N, self.K, self.L = d, E, N, K, layers
        g = lambda *s: (r.standard_normal(s) / np.sqrt(s[0])).astype(f)
        self.emb = r.standard_normal((self.V, d)).astype(f)
        self.layers = [dict(n=np.ones(d, f), win=g(d, 2 * E), conv=(r.standard_normal((K, E)) / np.sqrt(K)).astype(f),
                            wdt=g(E, E) * f(0.1), bdt=np.full(E, -2.0, f), wB=g(E, N), wC=g(E, N),
                            Alog=np.log(np.tile(np.arange(1, N + 1, dtype=f), (E, 1))), D=np.ones(E, f),
                            wout=g(E, d)) for _ in range(layers)]
        self.nf = np.ones(d, f)
        self.head = (r.standard_normal((d, self.V)) * (3.0 / np.sqrt(d))).astype(f)
        self.name, self.forwards = name, 0

    def start(self, prompt):
        return list(prompt)

    def _fwd(self, ids, state):
        x = self.emb[ids]
        T = len(ids)
        if state is None:
            state = [(np.zeros((self.K - 1, self.E), np.float32), np.zeros((self.E, self.N), np.float32))
                     for _ in range(self.L)]
        new = []
        for l, W in enumerate(self.layers):
            win, hst = state[l]
            xz = _rms(x, W["n"]) @ W["win"]
            xi, z = xz[:, :self.E], xz[:, self.E:]
            buf = np.concatenate([win, xi], axis=0)          # (K-1+T, E)
            ys = np.empty((T, self.E), np.float32)
            A = -np.exp(W["Alog"])
            for t in range(T):
                u = _silu((buf[t:t + self.K] * W["conv"]).sum(0))
                dt = np.log1p(np.exp(u @ W["wdt"] + W["bdt"]))
                B, C = u @ W["wB"], u @ W["wC"]
                hst = np.exp(dt[:, None] * A) * hst + (dt * u)[:, None] * B[None, :]
                ys[t] = hst @ C + W["D"] * u
            new.append((buf[-(self.K - 1):].copy(), hst))
            x = x + (ys * _silu(z)) @ W["wout"]
        return (_rms(x, self.nf) @ self.head).astype(np.float32), new

    def all_logits(self, c):
        self.forwards += 1
        return self._fwd(np.asarray(c, dtype=np.int64), None)[0]

    def logits(self, c):
        return self.all_logits(c)[-1].copy()

    def kv_rows(self, c, n0):
        ids = np.asarray(c, dtype=np.int64)
        lg, st = self._fwd(ids[:n0], None)
        rows = [lg[-1]]
        for t in range(n0, len(c)):
            lg, st = self._fwd(ids[t:t + 1], st)
            rows.append(lg[-1])
        return np.stack(rows)[:-1] if len(c) > n0 else np.stack(rows)


def make(name, threads=4):
    if name == "gpt2-fp32-ort":
        return GPT2Ort(threads=threads)
    if name == "gpt2-int8-ort":
        return GPT2Ort("gpt2-lm-head-10-int8.onnx", threads, name)
    if name == "gpt2-numpy-kv":
        return GPT2Numpy()
    if name == "t5-base-ort":
        return T5Ort(threads)
    if name == "llama-style-random":
        return LlamaRandom()
    if name == "mamba-style-random":
        return MambaRandom()
    raise KeyError(name)


NAMES = ["gpt2-fp32-ort", "gpt2-int8-ort", "gpt2-numpy-kv", "t5-base-ort",
         "llama-style-random", "mamba-style-random"]
