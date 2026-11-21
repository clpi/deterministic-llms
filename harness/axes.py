"""Named axes. Written AFTER tasks/tasks.json and tasks/CONTRACT.json were
frozen and logged (log records 3 and 4).

An axis maps (content, request) to a sparse feature on candidate edges
content -> token: a list of (token, value) pairs, integers. In the
hypergraph each pair is a color on that edge. The score of an edge is
base + sum over axes of weight * value (weights in units of 1/256 logit,
the same unit as the integer base score).

Every axis has a do-nothing control ("null": no pairs) and, where the
metric holds information the request does not give, a gamed control
("oracle": reads the reference, answer or held-out words). Gamed runs are
labelled GAMED and never counted.
"""
import os, re
import numpy as np
from common import tokenizer, encode, MODELS

AXES = ["rep", "noloop", "ban", "topic", "end", "embdir", "null", "oracle"]
K = len(AXES)
IDX = {a: i for i, a in enumerate(AXES)}

_tok = None


def tokstrs():
    global _tok
    if _tok is None:
        enc = tokenizer()
        raw = [enc.decode_single_token_bytes(i).decode("utf-8", "replace") for i in range(50257)]
        norm = [re.sub(r"[^a-z]", "", s.strip().lower()) for s in raw]
        _tok = (raw, norm)
    return _tok


_wte = None


def wte():
    global _wte
    if _wte is None:
        cache = "/tmp/claude-0/wte.npy"
        if os.path.exists(cache):
            _wte = np.load(cache)
        else:
            import onnx
            from onnx import numpy_helper
            m = onnx.load(os.path.join(MODELS, "gpt2-lm-head-10.onnx"))
            for t in m.graph.initializer:
                if t.name == "transformer.wte.weight":
                    _wte = numpy_helper.to_array(t).astype(np.float32)
            np.save(cache, _wte)
    return _wte


def word_first_tokens(words):
    """First token of ' word' and of 'Word' for each word."""
    out = set()
    for w in words:
        for v in (" " + w, " " + w.capitalize()):
            ids = encode(v)
            if ids:
                out.add(ids[0])
    return sorted(out)


def ban_tokens(words):
    raw, norm = tokstrs()
    ws = set(words)
    return [i for i in range(50257) if norm[i] in ws and raw[i].strip().lower() == norm[i]]


def end_tokens():
    raw, _ = tokstrs()
    return [i for i in range(50257) if any(ch in raw[i] for ch in ".!?")]


def embdir_feature(train_words, seed=None):
    """Opaque vector -> named axis: cosine of each token embedding with the
    direction mean(train-word embeddings) - mean(all embeddings), in
    thousandths, kept where |cos| >= 0.05. seed != None gives the random
    direction control (same norm, same sparsification)."""
    E = wte()
    mu = E.mean(0)
    if seed is None:
        d = E[word_first_tokens(train_words)].mean(0) - mu
    else:
        d = np.random.default_rng(seed).standard_normal(E.shape[1]).astype(np.float32)
    En = (E - mu) / np.linalg.norm(E - mu, axis=1, keepdims=True)
    cos = En @ (d / np.linalg.norm(d))
    v = np.rint(cos * 1000).astype(np.int64)
    keep = np.nonzero(np.abs(v) >= 50)[0]
    return [(int(i), int(v[i])) for i in keep]


class Request:
    """What a request tells the axes. `hidden` is what only the metric knows
    (reference, answer, held-out words); only the oracle axis reads it."""

    def __init__(self, item, hidden=None, embseed=None):
        self.item = item
        self.hidden = hidden or {}
        self.ban = ban_tokens(item.get("words", [])) if item.get("kind") == "ban" else []
        self.topic = word_first_tokens(item["train_words"]) if item.get("kind") == "topic" else []
        self.end = end_tokens() if item.get("kind") == "end" else []
        self.emb = (embdir_feature(item["train_words"], embseed)
                    if item.get("kind") == "topic" else [])


def features(content, gen_start, req, active):
    """Sparse entries (token, axis index, value) for the edges leaving
    `content`. `active` lists the axes that are switched on; an axis
    that is off contributes nothing (its weight is 0 anyway, this only
    keeps the replay files small)."""
    ents = []
    if "rep" in active:
        from collections import Counter
        for t, n in Counter(content[-64:]).items():
            ents.append((t, IDX["rep"], -n))
    if "noloop" in active and len(content) >= 3:
        a, b = content[-2], content[-1]
        seen = set()
        for i in range(len(content) - 2):
            if content[i] == a and content[i + 1] == b:
                seen.add(content[i + 2])
        for t in seen:
            ents.append((t, IDX["noloop"], -1))
    if "ban" in active:
        ents += [(t, IDX["ban"], -1) for t in req.ban]
    if "topic" in active:
        ents += [(t, IDX["topic"], 1) for t in req.topic]
    if "end" in active:
        ents += [(t, IDX["end"], 1) for t in req.end]
    if "embdir" in active:
        ents += [(t, IDX["embdir"], v) for t, v in req.emb]
    if "oracle" in active:
        pos = len(content) - gen_start
        h = req.hidden
        if "reference" in h and pos < len(h["reference"]):
            ents.append((h["reference"][pos], IDX["oracle"], 1))
        if "answer_tok" in h and pos == 0:
            ents.append((h["answer_tok"], IDX["oracle"], 1))
        if "test_tokens" in h:
            ents += [(t, IDX["oracle"], 1) for t in h["test_tokens"]]
    # "null" never contributes: the do-nothing control
    return ents


def weights(**kw):
    """Weight vector in 1/256-logit units from named logit values."""
    th = [0] * K
    for a, v in kw.items():
        th[IDX[a]] = int(round(v * 256))
    return th
