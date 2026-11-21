"""Decoders. Every one takes the same Model and prompt and returns the
generated tokens plus per-step facts. The PRLM rule here must agree
step for step with the Lean executable built from Prlm/Core.lean; every
PRLM step can be written to a replay file for that check."""
import heapq, struct, time
import numpy as np
from common import qscores
import axes as AX

V = 50257


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


# ---------------------------------------------------------------- baselines
def sample(model, prompt, steps, temp=1.0, top_p=None, seed=0):
    rng = np.random.default_rng(seed)
    c = list(prompt)
    t0 = time.perf_counter()
    for _ in range(steps):
        lp = logsoftmax(model.logits(c) / temp)
        p = np.exp(lp)
        if top_p is not None:
            order = np.argsort(-p, kind="stable")
            cs = np.cumsum(p[order])
            keep = order[: int(np.searchsorted(cs, top_p) + 1)]
            q = np.zeros_like(p)
            q[keep] = p[keep]
            p = q / q.sum()
        c.append(int(rng.choice(V, p=p)))
    return c[len(prompt):], {"wall": time.perf_counter() - t0}


def greedy(model, prompt, steps):
    c = list(prompt)
    t0 = time.perf_counter()
    for _ in range(steps):
        c.append(int(np.argmax(model.logits(c))))
    return c[len(prompt):], {"wall": time.perf_counter() - t0}


# ---------------------------------------------------------------- PRLM rule
def dense(ents, theta):
    """Per-token bonus sum_a theta_a * value, from sparse entries."""
    b = np.zeros(V, dtype=np.int64)
    for t, a, v in ents:
        if theta[a]:
            b[t] += theta[a] * v
    return b


def certify(scores):
    """Certified best-first rule over integer scores: highest score,
    lowest token id on ties; margin over the best other token."""
    v = int(np.argmax(scores))  # first index = lowest id among ties
    s = int(scores[v])
    rest = scores.copy()
    rest[v] = np.iinfo(np.int64).min
    r = int(rest.max())
    m = s - r
    return v, s, m, m > 0


class Replay:
    """Writes step records in the format Main.lean reads."""

    def __init__(self, path):
        self.f = open(path, "wb")
        self.n = 0

    def write(self, base, theta, delta, ents, pick, pick_re):
        f = self.f
        f.write(struct.pack("<II", V, len(theta)))
        f.write(np.asarray(base, dtype="<i4").tobytes())
        f.write(struct.pack(f"<{len(theta)}i", *theta))
        f.write(struct.pack(f"<{len(delta)}i", *delta))
        f.write(struct.pack("<I", len(ents)))
        if ents:
            f.write(np.asarray(ents, dtype="<i4").tobytes())
        f.write(struct.pack("<ii", pick, pick_re))
        self.n += 1

    def close(self):
        self.f.close()


def prlm(model, prompt, steps, req, theta, active, replay=None, delta=None):
    """Certified best-first generation with named axes. If `replay` is
    given, each step is written with a second weight vector theta+delta
    and the pick recomputed under it, for the Lean replay."""
    c = list(prompt)
    certs = []
    t0 = time.perf_counter()
    for _ in range(steps):
        base = qscores(model.logits(c))
        ents = AX.features(c, len(prompt), req, active)
        sc = base + dense(ents, theta)
        v, s, m, ok = certify(sc)
        certs.append((v, m, ok))
        if replay is not None:
            th2 = [a + b for a, b in zip(theta, delta)]
            v2 = certify(base + dense(ents, th2))[0]
            replay.write(base, theta, delta, ents, v, v2)
        c.append(v)
    return c[len(prompt):], {"wall": time.perf_counter() - t0, "certs": certs}


# ---------------------------------------------------------------- alternatives
def beam_cert(model, prompt, steps, B=4):
    """Beam search on summed log-probabilities. Certificate: log-probs are
    <= 0, so a pruned prefix's score bounds every extension of it; if the
    best finished beam scores at least the best pruned prefix, it is the
    most likely sequence of this length (certified optimal)."""
    beams = [(0.0, list(prompt))]
    pruned_best = -np.inf
    t0 = time.perf_counter()
    for _ in range(steps):
        cand = []
        for s, c in beams:
            lp = logsoftmax(model.logits(c))
            top = np.argpartition(-lp, B)[: B + 1]
            for t in top:
                cand.append((s + lp[t], c + [int(t)]))
            # everything outside the top B+1 of this beam is pruned too;
            # its best is the (B+2)-th child, bounded by the (B+1)-th
            pruned_best = max(pruned_best, s + np.sort(lp)[-(B + 2)])
        cand.sort(key=lambda x: (-x[0], x[1]))
        beams = cand[:B]
        if len(cand) > B:
            pruned_best = max(pruned_best, cand[B][0])
    best = beams[0]
    return best[1][len(prompt):], {"wall": time.perf_counter() - t0, "score": best[0],
                                    "certified": bool(best[0] >= pruned_best)}


def bestfirst_cert(model, prompt, steps, budget=200):
    """Uniform-cost search for the most likely sequence of `steps` tokens
    (cost = -log p, nonnegative). Children are pushed lazily in rank
    order. The first complete sequence popped is optimal (certified).
    Gives up after `budget` model calls (not certified; returns greedy)."""
    t0 = time.perf_counter()
    calls0 = model.forwards
    sorted_cache = {}
    L0 = len(prompt)

    def children(c):
        k = tuple(c)
        if k not in sorted_cache:
            nll = -logsoftmax(model.logits(c))
            order = np.argsort(nll, kind="stable")
            sorted_cache[k] = (order, nll[order])
        return sorted_cache[k]

    heap = [(0.0, tuple(prompt), -1)]  # (cost, prefix, rank of last token)
    while heap:
        cost, pre, rank = heapq.heappop(heap)
        if len(pre) - L0 == steps:
            return list(pre[L0:]), {"wall": time.perf_counter() - t0, "certified": True,
                                    "nll": cost, "calls": model.forwards - calls0}
        if model.forwards - calls0 >= budget:
            break
        if rank >= 0:  # push next sibling
            parent = pre[:-1]
            order, nl = children(list(parent))
            pc = cost - float(nl[rank])
            if rank + 1 < V:
                heapq.heappush(heap, (pc + float(nl[rank + 1]), parent + (int(order[rank + 1]),), rank + 1))
        order, nl = children(list(pre))
        heapq.heappush(heap, (cost + float(nl[0]), pre + (int(order[0]),), 0))
    g, _ = greedy(model, prompt, steps)
    return g, {"wall": time.perf_counter() - t0, "certified": False, "calls": model.forwards - calls0}


class NGram:
    """Deterministic trigram automaton with backoff: argmax of counts,
    lowest token id on ties. A lookup table, no neural model."""

    def __init__(self, docs):
        from collections import Counter, defaultdict
        self.t3 = defaultdict(Counter)
        self.t2 = defaultdict(Counter)
        self.t1 = Counter()
        for ids in docs:
            for i, t in enumerate(ids):
                self.t1[t] += 1
                if i >= 1:
                    self.t2[ids[i - 1]][t] += 1
                if i >= 2:
                    self.t3[(ids[i - 2], ids[i - 1])][t] += 1

    @staticmethod
    def _arg(cnt):
        m = max(cnt.values())
        return min(t for t, n in cnt.items() if n == m)

    def next(self, c):
        if len(c) >= 2 and (c[-2], c[-1]) in self.t3:
            return self._arg(self.t3[(c[-2], c[-1])]), 3
        if c and c[-1] in self.t2:
            return self._arg(self.t2[c[-1]]), 2
        return self._arg(self.t1), 1

    def gen(self, prompt, steps):
        c = list(prompt)
        t0 = time.perf_counter()
        for _ in range(steps):
            c.append(self.next(c)[0])
        return c[len(prompt):], {"wall": time.perf_counter() - t0}


class RuleTable:
    """Greedy decoding distilled into rules: (last k tokens) -> the
    model's greedy next token, kept only where every TRAIN occurrence of
    the key gave the same answer at least `min_support` times. At run
    time a rule fires without a model call; otherwise the model decides."""

    def __init__(self, model, docs, k=3, min_support=2):
        from collections import defaultdict, Counter
        seen = defaultdict(Counter)
        for ids in docs:
            ids = ids[:1024]
            x = np.asarray(ids, dtype=np.int64).reshape(1, 1, -1)
            lg = model.sess.run(["output1"], {"input1": x})[0][0, 0]
            g = lg.argmax(-1)
            for i in range(k - 1, len(ids)):
                seen[tuple(ids[i - k + 1:i + 1])][int(g[i])] += 1
        self.k = k
        self.rules = {key: next(iter(c)) for key, c in seen.items()
                      if len(c) == 1 and sum(c.values()) >= min_support}

    def gen(self, model, prompt, steps):
        c = list(prompt)
        fired = 0
        t0 = time.perf_counter()
        for _ in range(steps):
            key = tuple(c[-self.k:])
            if key in self.rules:
                c.append(self.rules[key])
                fired += 1
            else:
                c.append(int(np.argmax(model.logits(c))))
        return c[len(prompt):], {"wall": time.perf_counter() - t0, "fired": fired}


def nll_under_base(model, prompt, gen):
    """Mean negative log-likelihood per generated token under the
    unmodified model (one teacher-forced forward)."""
    ids = list(prompt) + list(gen)
    x = np.asarray(ids[-1024:], dtype=np.int64).reshape(1, 1, -1)
    lg = model.sess.run(["output1"], {"input1": x})[0][0, 0].astype(np.float64)
    off = len(ids) - x.shape[-1]
    tot = 0.0
    for j in range(len(prompt), len(ids)):
        row = lg[j - 1 - off]
        m = row.max()
        tot += -(row[ids[j]] - m - np.log(np.exp(row - m).sum()))
    return tot / max(1, len(gen))
