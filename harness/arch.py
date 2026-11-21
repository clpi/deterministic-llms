"""Architecture as a parameter (tasks/ARCH_CONTRACT.json).

One implementation over an array module `xp`: JAX (jax.numpy) for training,
numpy float32 for every measured property. A config picks the layer kinds
(attn, ssm, rwkv, gru, lstm), the position scheme (learned, rope, alibi),
the KV-head count (MHA, GQA, MQA), the feed-forward (gelu, swiglu, moe,
cmix) and the norm (ln, rms). Two entry points, the only ones the
measurements use:

  full(xp, p, cfg, ids)            logits for every position, one call
  step(xp, p, cfg, cache, tok, t)  logits for one new token and the cache
                                   (KV for attention, state for SSM/RNN)
"""
import numpy as np

D, H, HD, K, N, MAXLEN = 128, 4, 32, 4, 16, 512
E = 2 * D            # SSM inner width
F = 344              # SwiGLU / expert hidden
NE = 4               # MoE experts, top-2

CONFIGS = {
    "A1": dict(name="attn-learned-mha-gelu-ln", kinds=["attn"] * 4, pos="learned", kv=4, ffn="gelu", norm="ln"),
    "A2": dict(name="attn-rope-gqa-swiglu-rms", kinds=["attn"] * 4, pos="rope", kv=2, ffn="swiglu", norm="rms"),
    "A3": dict(name="attn-rope-mqa-swiglu-rms", kinds=["attn"] * 4, pos="rope", kv=1, ffn="swiglu", norm="rms"),
    "A4": dict(name="attn-alibi-mha-gelu-ln", kinds=["attn"] * 4, pos="alibi", kv=4, ffn="gelu", norm="ln"),
    "A5": dict(name="attn-rope-gqa-moe-rms", kinds=["attn"] * 4, pos="rope", kv=2, ffn="moe", norm="rms"),
    "S1": dict(name="ssm-selective", kinds=["ssm"] * 4, pos="none", kv=0, ffn="none", norm="rms"),
    "R1": dict(name="rnn-rwkv", kinds=["rwkv"] * 4, pos="none", kv=0, ffn="cmix", norm="ln"),
    "R2": dict(name="rnn-gru", kinds=["gru"] * 4, pos="none", kv=0, ffn="gelu", norm="ln"),
    "R3": dict(name="rnn-lstm", kinds=["lstm"] * 4, pos="none", kv=0, ffn="gelu", norm="ln"),
    "H1": dict(name="hybrid-attn-ssm", kinds=["attn", "ssm", "attn", "ssm"], pos="rope", kv=2, ffn="swiglu", norm="rms"),
}


# ---------------------------------------------------------------- init
def init(cfg, V, seed=0):
    r = np.random.default_rng(seed)
    f = np.float32
    L = len(cfg["kinds"])
    g = lambda *s: (r.standard_normal(s) / np.sqrt(s[-2] if len(s) > 1 else s[0])).astype(f)
    out_scale = f(1 / np.sqrt(2 * L))
    p = {"emb": (r.standard_normal((V, D)) * (2.0 / np.sqrt(D))).astype(f), "nf_g": np.ones(D, f), "nf_b": np.zeros(D, f)}
    if cfg["pos"] == "learned":
        p["wpe"] = (r.standard_normal((MAXLEN, D)) * 0.1).astype(f)
    for l, kind in enumerate(cfg["kinds"]):
        P = f"l{l}."
        p[P + "n1_g"], p[P + "n1_b"] = np.ones(D, f), np.zeros(D, f)
        if kind == "attn":
            kv = cfg["kv"]
            p[P + "wq"], p[P + "wk"], p[P + "wv"] = g(D, D), g(D, kv * HD), g(D, kv * HD)
            p[P + "wo"] = g(D, D) * out_scale
        elif kind == "ssm":
            p[P + "win"], p[P + "conv"] = g(D, 2 * E), (r.standard_normal((K, E)) / np.sqrt(K)).astype(f)
            p[P + "wdt"], p[P + "bdt"] = g(E, E) * f(0.1), np.full(E, -2.0, f)
            p[P + "wB"], p[P + "wC"] = g(E, N), g(E, N)
            p[P + "Alog"] = np.log(np.tile(np.arange(1, N + 1, dtype=f), (E, 1)))
            p[P + "Dp"], p[P + "wout"] = np.ones(E, f), g(E, D) * out_scale
        elif kind == "rwkv":
            p[P + "mk"], p[P + "mv"], p[P + "mr"] = (r.uniform(0.3, 0.7, (3, D))).astype(f)
            p[P + "wk"], p[P + "wv"], p[P + "wr"] = g(D, D), g(D, D), g(D, D)
            p[P + "decay"] = np.linspace(-3, 1, D).astype(f)
            p[P + "bonus"] = np.full(D, 0.5, f)
            p[P + "wo"] = g(D, D) * out_scale
        elif kind == "gru":
            p[P + "W"], p[P + "U"], p[P + "b"] = g(D, 3 * D), g(D, 3 * D), np.zeros(3 * D, f)
            p[P + "bu"] = np.zeros(D, f)
        elif kind == "lstm":
            p[P + "W"], p[P + "U"] = g(D, 4 * D), g(D, 4 * D)
            p[P + "b"] = np.concatenate([np.zeros(D), np.ones(D), np.zeros(2 * D)]).astype(f)
        ffn = cfg["ffn"]
        if ffn != "none":
            p[P + "n2_g"], p[P + "n2_b"] = np.ones(D, f), np.zeros(D, f)
        if ffn == "gelu":
            p[P + "w1"], p[P + "b1"] = g(D, 4 * D), np.zeros(4 * D, f)
            p[P + "w2"], p[P + "b2"] = g(4 * D, D) * out_scale, np.zeros(D, f)
        elif ffn == "swiglu":
            p[P + "w1"], p[P + "w3"], p[P + "w2"] = g(D, F), g(D, F), g(F, D) * out_scale
        elif ffn == "moe":
            p[P + "wr"] = g(D, NE)
            p[P + "e1"] = (r.standard_normal((NE, D, F)) / np.sqrt(D)).astype(f)
            p[P + "e3"] = (r.standard_normal((NE, D, F)) / np.sqrt(D)).astype(f)
            p[P + "e2"] = (r.standard_normal((NE, F, D)) / np.sqrt(F) * out_scale).astype(f)
        elif ffn == "cmix":
            p[P + "cmk"], p[P + "cmr"] = (r.uniform(0.3, 0.7, (2, D))).astype(f)
            p[P + "ck"], p[P + "cv"], p[P + "cr"] = g(D, 4 * D), g(4 * D, D) * out_scale, g(D, D)
    return p


# ---------------------------------------------------------------- pieces
def norm(xp, x, g, b, kind):
    if kind == "ln":
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / xp.sqrt(v + 1e-5) * g + b
    return x / xp.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * g


def sigmoid(xp, x):
    return 1 / (1 + xp.exp(-x))


def silu(xp, x):
    return x * sigmoid(xp, x)


def gelu(xp, x):
    return 0.5 * x * (1 + xp.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def softplus(xp, x):
    return xp.log1p(xp.exp(-xp.abs(x))) + xp.maximum(x, 0)


def softmax(xp, x):
    x = x - x.max(-1, keepdims=True)
    e = xp.exp(x)
    return e / e.sum(-1, keepdims=True)


def rope(xp, x, pos):  # x (h, T, HD), pos (T,)
    inv = 1.0 / (10000 ** (xp.arange(0, HD, 2, dtype=xp.float32) / HD))
    ang = pos.astype(xp.float32)[:, None] * inv[None, :]
    c, s = xp.cos(ang), xp.sin(ang)
    a, b = x[..., 0::2], x[..., 1::2]
    return xp.stack([a * c - b * s, a * s + b * c], axis=-1).reshape(x.shape)


def slopes(xp):
    return xp.asarray([2 ** (-8 * (h + 1) / H) for h in range(H)], dtype=xp.float32)


def recur(xp, fn, carry, xs):
    """xs: tuple of arrays with time on axis 0."""
    if xp is np:
        ys = []
        for t in range(xs[0].shape[0]):
            carry, y = fn(carry, tuple(x[t] for x in xs))
            ys.append(y)
        return carry, np.stack(ys)
    import jax
    return jax.lax.scan(fn, carry, xs)


# ---------------------------------------------------------------- mixers, full path
def attn_full(xp, p, P, cfg, x):
    T, kv = x.shape[0], cfg["kv"]
    q = (x @ p[P + "wq"]).reshape(T, H, HD).transpose(1, 0, 2)
    k = (x @ p[P + "wk"]).reshape(T, kv, HD).transpose(1, 0, 2)
    v = (x @ p[P + "wv"]).reshape(T, kv, HD).transpose(1, 0, 2)
    pos = xp.arange(T)
    if cfg["pos"] == "rope":
        q, k = rope(xp, q, pos), rope(xp, k, pos)
    k, v = xp.repeat(k, H // kv, axis=0), xp.repeat(v, H // kv, axis=0)
    att = (q @ k.transpose(0, 2, 1)) * (HD ** -0.5)
    if cfg["pos"] == "alibi":
        att = att - slopes(xp)[:, None, None] * (pos[:, None] - pos[None, :]).astype(xp.float32)[None]
    att = xp.where(pos[None, :] > pos[:, None], -1e9, att)
    y = (softmax(xp, att) @ v).transpose(1, 0, 2).reshape(T, D)
    return y @ p[P + "wo"]


def ssm_cell(xp, p, P):
    A = -xp.exp(p[P + "Alog"])

    def fn(h, inp):
        u, dt, B, C = inp
        h = xp.exp(dt[:, None] * A) * h + (dt * u)[:, None] * B[None, :]
        return h, h @ C + p[P + "Dp"] * u
    return fn


def ssm_full(xp, p, P, cfg, x):
    T = x.shape[0]
    xz = x @ p[P + "win"]
    xi, z = xz[:, :E], xz[:, E:]
    buf = xp.concatenate([xp.zeros((K - 1, E), dtype=xp.float32), xi], axis=0)
    u = silu(xp, sum(buf[k:k + T] * p[P + "conv"][k] for k in range(K)))
    dt = softplus(xp, u @ p[P + "wdt"] + p[P + "bdt"])
    B, C = u @ p[P + "wB"], u @ p[P + "wC"]
    _, y = recur(xp, ssm_cell(xp, p, P), xp.zeros((E, N), dtype=xp.float32), (u, dt, B, C))
    return (y * silu(xp, z)) @ p[P + "wout"]


def shift(xp, x):
    return xp.concatenate([xp.zeros((1, x.shape[1]), dtype=xp.float32), x[:-1]], axis=0)


def wkv_cell(xp, p, P):
    w, u = -xp.exp(p[P + "decay"]), p[P + "bonus"]

    def fn(st, inp):
        a, b, pp = st
        k, v = inp
        ww = u + k
        q = xp.maximum(pp, ww)
        e1, e2 = xp.exp(pp - q), xp.exp(ww - q)
        out = (e1 * a + e2 * v) / (e1 * b + e2)
        ww = pp + w
        q = xp.maximum(ww, k)
        e1, e2 = xp.exp(ww - q), xp.exp(k - q)
        return (e1 * a + e2 * v, e1 * b + e2, q), out
    return fn


def wkv_init(xp):
    z = xp.zeros(D, dtype=xp.float32)
    return (z, z, z - 1e30)


def rwkv_proj(xp, p, P, x, xx):
    k = (x * p[P + "mk"] + xx * (1 - p[P + "mk"])) @ p[P + "wk"]
    v = (x * p[P + "mv"] + xx * (1 - p[P + "mv"])) @ p[P + "wv"]
    r = sigmoid(xp, (x * p[P + "mr"] + xx * (1 - p[P + "mr"])) @ p[P + "wr"])
    return k, v, r


def rwkv_full(xp, p, P, cfg, x):
    k, v, r = rwkv_proj(xp, p, P, x, shift(xp, x))
    _, o = recur(xp, wkv_cell(xp, p, P), wkv_init(xp), (k, v))
    return (r * o) @ p[P + "wo"]


def gru_cell(xp, p, P):
    def fn(h, inp):
        (gx,) = inp
        gh = h @ p[P + "U"]
        z = sigmoid(xp, gx[:D] + gh[:D])
        r = sigmoid(xp, gx[D:2 * D] + gh[D:2 * D])
        n = xp.tanh(gx[2 * D:] + r * (gh[2 * D:] + p[P + "bu"]))
        h = (1 - z) * n + z * h
        return h, h
    return fn


def lstm_cell(xp, p, P):
    def fn(st, inp):
        h, c = st
        (gx,) = inp
        g = gx + h @ p[P + "U"]
        i, f, gg, o = sigmoid(xp, g[:D]), sigmoid(xp, g[D:2 * D]), xp.tanh(g[2 * D:3 * D]), sigmoid(xp, g[3 * D:])
        c = f * c + i * gg
        h = o * xp.tanh(c)
        return (h, c), h
    return fn


def rnn_full(xp, p, P, cfg, x, kind):
    gx = x @ p[P + "W"] + p[P + "b"]
    z = xp.zeros(D, dtype=xp.float32)
    if kind == "gru":
        _, y = recur(xp, gru_cell(xp, p, P), z, (gx,))
    else:
        _, y = recur(xp, lstm_cell(xp, p, P), (z, z), (gx,))
    return y


# ---------------------------------------------------------------- feed-forward
def ffn_apply(xp, p, P, cfg, x, xx=None):
    f = cfg["ffn"]
    if f == "gelu":
        return gelu(xp, x @ p[P + "w1"] + p[P + "b1"]) @ p[P + "w2"] + p[P + "b2"]
    if f == "swiglu":
        return (silu(xp, x @ p[P + "w1"]) * (x @ p[P + "w3"])) @ p[P + "w2"]
    if f == "moe":
        lg = x @ p[P + "wr"]                                  # (T, NE)
        kk = cfg.get("topk", 2)
        thr = xp.sort(lg, axis=-1)[..., NE - kk:NE - kk + 1]
        lg = xp.where(lg >= thr, lg, -1e9)
        wt = softmax(xp, lg)                                  # zero outside top-2
        h = silu(xp, xp.einsum("td,edf->tef", x, p[P + "e1"])) * xp.einsum("td,edf->tef", x, p[P + "e3"])
        y = xp.einsum("tef,efd->ted", h, p[P + "e2"])
        return (wt[..., None] * y).sum(1)
    if f == "cmix":
        k = (x * p[P + "cmk"] + xx * (1 - p[P + "cmk"])) @ p[P + "ck"]
        r = sigmoid(xp, (x * p[P + "cmr"] + xx * (1 - p[P + "cmr"])) @ p[P + "cr"])
        return r * ((xp.maximum(k, 0) ** 2) @ p[P + "cv"])
    raise KeyError(f)


# ---------------------------------------------------------------- full path
def full(xp, p, cfg, ids):
    x = p["emb"][ids]
    if cfg["pos"] == "learned":
        x = x + p["wpe"][:x.shape[0]]
    for l, kind in enumerate(cfg["kinds"]):
        P = f"l{l}."
        h = norm(xp, x, p[P + "n1_g"], p[P + "n1_b"], cfg["norm"])
        if kind == "attn":
            x = x + attn_full(xp, p, P, cfg, h)
        elif kind == "ssm":
            x = x + ssm_full(xp, p, P, cfg, h)
        elif kind == "rwkv":
            x = x + rwkv_full(xp, p, P, cfg, h)
        else:
            x = x + rnn_full(xp, p, P, cfg, h, kind)
        if cfg["ffn"] != "none":
            h = norm(xp, x, p[P + "n2_g"], p[P + "n2_b"], cfg["norm"])
            x = x + ffn_apply(xp, p, P, cfg, h, shift(xp, h) if cfg["ffn"] == "cmix" else None)
    x = norm(xp, x, p["nf_g"], p["nf_b"], cfg["norm"])
    return x @ p["emb"].T


# ---------------------------------------------------------------- incremental path (numpy)
def new_cache(cfg):
    c = []
    for kind in cfg["kinds"]:
        z = np.zeros(D, np.float32)
        s = {"kind": kind}
        if kind == "attn":
            s["k"], s["v"] = np.zeros((cfg["kv"], 0, HD), np.float32), np.zeros((cfg["kv"], 0, HD), np.float32)
        elif kind == "ssm":
            s["win"], s["h"] = np.zeros((K - 1, E), np.float32), np.zeros((E, N), np.float32)
        elif kind == "rwkv":
            s["xx"], s["wkv"] = z, wkv_init(np)
        elif kind == "gru":
            s["h"] = z
        elif kind == "lstm":
            s["h"] = (z, z)
        if cfg["ffn"] == "cmix":
            s["cxx"] = z
        c.append(s)
    return c


def step(p, cfg, cache, tok, t):
    xp = np
    x = p["emb"][tok]
    if cfg["pos"] == "learned":
        x = x + p["wpe"][t]
    for l, s in enumerate(cache):
        P, kind = f"l{l}.", s["kind"]
        h = norm(xp, x, p[P + "n1_g"], p[P + "n1_b"], cfg["norm"])
        if kind == "attn":
            kv = cfg["kv"]
            q = (h @ p[P + "wq"]).reshape(H, 1, HD)
            k = (h @ p[P + "wk"]).reshape(kv, 1, HD)
            v = (h @ p[P + "wv"]).reshape(kv, 1, HD)
            if cfg["pos"] == "rope":
                q, k = rope(xp, q, np.array([t])), rope(xp, k, np.array([t]))
            s["k"] = np.concatenate([s["k"], k], axis=1)
            s["v"] = np.concatenate([s["v"], v], axis=1)
            kk, vv = np.repeat(s["k"], H // kv, axis=0), np.repeat(s["v"], H // kv, axis=0)
            att = (q @ kk.transpose(0, 2, 1)) * (HD ** -0.5)
            if cfg["pos"] == "alibi":
                att = att - slopes(xp)[:, None, None] * (t - np.arange(t + 1)).astype(np.float32)[None, None, :]
            y = (softmax(xp, att) @ vv).transpose(1, 0, 2).reshape(D)
            x = x + y @ p[P + "wo"]
        elif kind == "ssm":
            xz = h @ p[P + "win"]
            xi, z = xz[:E], xz[E:]
            buf = np.concatenate([s["win"], xi[None]], axis=0)
            u = silu(xp, sum(buf[k:k + 1] * p[P + "conv"][k] for k in range(K))[0])
            dt = softplus(xp, u @ p[P + "wdt"] + p[P + "bdt"])
            s["h"], y = ssm_cell(xp, p, P)(s["h"], (u, dt, u @ p[P + "wB"], u @ p[P + "wC"]))
            s["win"] = buf[1:]
            x = x + (y * silu(xp, z)) @ p[P + "wout"]
        elif kind == "rwkv":
            k, v, r = rwkv_proj(xp, p, P, h, s["xx"])
            s["wkv"], o = wkv_cell(xp, p, P)(s["wkv"], (k, v))
            s["xx"] = h
            x = x + (r * o) @ p[P + "wo"]
        else:
            gx = h @ p[P + "W"] + p[P + "b"]
            cell = gru_cell(xp, p, P) if kind == "gru" else lstm_cell(xp, p, P)
            s["h"], y = cell(s["h"], (gx,))
            x = x + y
        if cfg["ffn"] != "none":
            h = norm(xp, x, p[P + "n2_g"], p[P + "n2_b"], cfg["norm"])
            if cfg["ffn"] == "cmix":
                y = ffn_apply(xp, p, P, cfg, h[None], s["cxx"][None])[0]
                s["cxx"] = h
            else:
                y = ffn_apply(xp, p, P, cfg, h[None])[0]
            x = x + y
    x = norm(xp, x, p["nf_g"], p["nf_b"], cfg["norm"])
    return x @ p["emb"].T


# ---------------------------------------------------------------- training (JAX)
def train(cfg, V, data, steps=600, lr=3e-3, batch=16, win=128, seed=0):
    import jax, jax.numpy as jnp
    p0 = init(cfg, V, seed)
    params = {k: jnp.asarray(v) for k, v in p0.items()}
    data = np.asarray(data, np.int32)

    def loss(pr, xb):
        def one(ids):
            lg = full(jnp, pr, cfg, ids[:-1])
            lp = jax.nn.log_softmax(lg, axis=-1)
            return -jnp.take_along_axis(lp, ids[1:, None], axis=-1).mean()
        return jax.vmap(one)(xb).mean()

    @jax.jit
    def upd(pr, m, v, xb, t):
        l, gr = jax.value_and_grad(loss)(pr, xb)
        m = jax.tree_util.tree_map(lambda a, g: 0.9 * a + 0.1 * g, m, gr)
        v = jax.tree_util.tree_map(lambda a, g: 0.999 * a + 0.001 * g * g, v, gr)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - 0.9 ** t), m)
        vh = jax.tree_util.tree_map(lambda a: a / (1 - 0.999 ** t), v)
        pr = jax.tree_util.tree_map(lambda w, a, b: w - lr * a / (jnp.sqrt(b) + 1e-8), pr, mh, vh)
        return pr, m, v, l

    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    r = np.random.default_rng(seed)
    losses = []
    for t in range(1, steps + 1):
        st = r.integers(0, len(data) - win - 1, batch)
        xb = np.stack([data[s:s + win + 1] for s in st])
        params, m, v, l = upd(params, m, v, xb, float(t))
        losses.append(float(l))
    return {k: np.asarray(v, np.float32) for k, v in params.items()}, losses
