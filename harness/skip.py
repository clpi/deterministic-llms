"""Forward skipping through multi-step continuation equivalences.
Contract: tasks/SKIP_CONTRACT.json (log record 3391, written before this file).

A fact maps an equivalence class (the last k tokens of a content) to a
continuation that some witness history produced after it. Facts are
appended as histories arrive; a later history that reaches the class
attaches its own witness (the tokens it accepted and their log-probability
under its own logits). There is one log and one fact store; nothing is
overwritten except which continuation a class offers first (the latest).

verified: draft the continuation, check it with one forward over
content + draft, accept while each drafted token is the certified pick
from that forward's logits, emit the computed pick where they disagree.
unverified: emit the continuation with no model call.
Both are compared with a clean replay (one call per token).
"""
import hashlib, json, os, sys, time
import numpy as np
from common import *
import decoders as D

T = json.load(open(os.path.join(ROOT, "tasks", "tasks.json")))
L = Log()
M = Model(threads=4)
MAXN, CONT = 24, 8
RAW = None


def raw():
    global RAW
    if RAW is None:
        import axes as AX
        RAW = AX.tokstrs()[0]
    return RAW


def is_stop(t):
    s = raw()[t]
    return "\n" in s or "." in s


def all_logits(seq):
    x = np.asarray(seq[-1024:], dtype=np.int64).reshape(1, 1, -1)
    t = time.perf_counter()
    out = M.sess.run(["output1"], {"input1": x})[0][0, 0]
    M.forward_s += time.perf_counter() - t
    M.forwards += 1
    return out


def step_cert(row):
    q = qscores(row)
    v, s, m, ok = D.certify(q)
    lp = float(D.logsoftmax(row)[v])
    return v, m, ok, lp


def hrow(row):
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()[:16]


def stop_cert(gen, certs):
    if not gen:
        return None
    last = gen[-1]
    reason = "newline" if "\n" in raw()[last] else ("period" if "." in raw()[last] else "length")
    m = certs[-1][1] if certs and certs[-1] is not None else None
    ok = certs[-1][2] if certs and certs[-1] is not None else None
    return [len(gen), int(last), m, ok, reason]


def clean(prompt):
    c, gen, certs, rows = list(prompt), [], [], {}
    f0, t0 = M.forwards, time.perf_counter()
    while len(gen) < MAXN:
        row = all_logits(c)[-1]
        rows[M.key(c)] = hrow(row)
        v, m, ok, lp = step_cert(row)
        certs.append((v, m, bool(ok), lp))
        c.append(v)
        gen.append(v)
        if is_stop(v):
            break
    return gen, certs, rows, M.forwards - f0, time.perf_counter() - t0


class Facts:
    def __init__(self, k):
        self.k = k
        self.cont = {}       # class -> continuation tokens (latest witness)
        self.witness = {}    # class -> list of witness ids
        self.origin = {}     # class -> history whose continuation is offered
        self.added = 0

    def key(self, c):
        return tuple(c[-self.k:]) if len(c) >= self.k else None

    def learn(self, c, gen_start, wid):
        """Append facts for every class whose continuation ends at the
        current end of c and lies in generated tokens."""
        n = len(c)
        for j in range(1, CONT + 1):
            t = n - j
            if t < gen_start or t < self.k:
                break
            kk = tuple(c[t - self.k:t])
            cont = c[t:n]
            if len(cont) >= len(self.cont.get(kk, ())) or self.cont.get(kk, [None])[0] != cont[0]:
                if kk not in self.cont:
                    self.added += 1
                self.cont[kk] = list(cont)
                self.origin[kk] = wid
            self.witness.setdefault(kk, [])
            if not self.witness[kk] or self.witness[kk][-1] != wid:
                self.witness[kk].append(wid)


def run(prompt, facts, mode, wid, learn=True, clean_rows=None):
    c, gen, certs = list(prompt), [], []
    attached = []  # (class hash, accepted, mass)
    bitwise = [0, 0]
    f0, t0 = M.forwards, time.perf_counter()
    gs = len(prompt)
    while len(gen) < MAXN:
        kk = facts.key(c)
        draft = facts.cont.get(kk) if kk is not None else None
        stopped = False
        if draft and mode == "unverified":
            for t in draft:
                c.append(t); gen.append(t); certs.append(None)
                if learn:
                    facts.learn(c, gs, wid)
                if is_stop(t) or len(gen) >= MAXN:
                    stopped = True
                    break
            attached.append((hashlib.sha256(repr(kk).encode()).hexdigest()[:12], len(draft), None))
            if stopped:
                break
            continue
        if draft and mode == "verified":
            seq = c + draft
            lg = all_logits(seq)
            acc, mass = 0, 0.0
            for i in range(len(draft) + 1):
                row = lg[len(c) - 1]
                if clean_rows is not None and M.key(c) in clean_rows:
                    bitwise[1] += 1
                    bitwise[0] += int(hrow(row) == clean_rows[M.key(c)])
                v, m, ok, lp = step_cert(row)
                certs.append((v, m, bool(ok), lp))
                c.append(v); gen.append(v)
                if learn:
                    facts.learn(c, gs, wid)
                agree = i < len(draft) and v == draft[i]
                if agree:
                    acc += 1; mass += lp
                if is_stop(v) or len(gen) >= MAXN:
                    stopped = True
                    break
                if not agree:
                    break
            attached.append((hashlib.sha256(repr(kk).encode()).hexdigest()[:12], acc, round(mass, 6),
                             facts.origin.get(kk) == wid))
            if stopped:
                break
            continue
        row = all_logits(c)[-1]
        if clean_rows is not None and M.key(c) in clean_rows:
            bitwise[1] += 1
            bitwise[0] += int(hrow(row) == clean_rows[M.key(c)])
        v, m, ok, lp = step_cert(row)
        certs.append((v, m, bool(ok), lp))
        c.append(v); gen.append(v)
        if learn:
            facts.learn(c, gs, wid)
        if is_stop(v):
            break
    return gen, certs, M.forwards - f0, time.perf_counter() - t0, attached, bitwise


def histories():
    dev = [{"id": f"dev/{x['module']}", "prompt": x["ids"][:32]} for x in T["corpus"] if len(x["ids"]) >= 56][:20]
    return dev + [{"id": t["id"], "prompt": t["prompt"]} for t in T["t1"]]


def origin_run():
    """Addendum: verified runs again, recording whether each accepted draft
    came from the same history or an earlier one. No clean rerun: the
    clean replies are read back from the log."""
    H = histories()
    recs = [json.loads(l) for l in open(LOG)]
    ref = {r["receipt"]["history"]: r["receipt"] for r in recs if r["head"].startswith("skip:clean:")}
    for k in (2, 4, 8):
        facts = Facts(k)
        tot = {"calls": 0, "tokens": 0, "reply_mismatch": 0, "stop_mismatch": 0,
               "accepted_same_history": 0, "accepted_earlier_history": 0, "drafts": 0}
        for h in H:
            g, cs, calls, wall, att, bw = run(h["prompt"], facts, "verified", h["id"])
            tot["calls"] += calls; tot["tokens"] += len(g); tot["drafts"] += len(att)
            tot["reply_mismatch"] += int(g != ref[h["id"]]["reply"])
            tot["stop_mismatch"] += int(stop_cert(g, cs) != ref[h["id"]]["stop"])
            for a in att:
                tot["accepted_same_history" if a[3] else "accepted_earlier_history"] += a[1]
        L.add(f"skip-origin:k{k}-verified", ["measured", "skip"], premises=[3391], receipt=tot)
        print(k, tot, flush=True)


def main():
    H = histories()
    ref = {}
    for h in H:
        g, cs, rows, calls, wall = clean(h["prompt"])
        ref[h["id"]] = (g, cs, rows)
        L.add(f"skip:clean:{h['id']}", ["output", "skip", "clean"], premises=[3391],
              receipt={"history": h["id"], "reply": g, "stop": stop_cert(g, cs),
                       "margins": [x[1] for x in cs], "calls": calls, "wall": round(wall, 4)})
    print("clean done", flush=True)
    configs = [("control-empty", 4, "verified", "empty")]
    for k in (2, 4, 8):
        for mode in ("verified", "unverified"):
            configs.append((f"k{k}-{mode}", k, mode, "accumulate"))
            configs.append((f"GAMED-k{k}-{mode}", k, mode, "gamed"))
    for name, k, mode, store in configs:
        facts = Facts(k)
        tot = {"calls": 0, "clean_calls": 0, "tokens": 0, "reply_mismatch": [], "stop_mismatch": [],
               "margin_mismatch_steps": 0, "pick_mismatch_steps": 0, "bitwise": [0, 0],
               "wall": 0.0, "clean_wall": 0.0, "accepted": 0, "attached": 0}
        for h in H:
            g0, cs0, rows0 = ref[h["id"]]
            if store == "gamed":
                facts = Facts(k)
                cc = list(h["prompt"]) + g0
                for t in range(len(h["prompt"]), len(cc)):
                    kk = tuple(cc[t - k:t])
                    facts.cont.setdefault(kk, cc[t:t + CONT])
            learn = store == "accumulate"
            g, cs, calls, wall, att, bw = run(h["prompt"], facts, mode, h["id"], learn=learn, clean_rows=rows0)
            sc, sc0 = stop_cert(g, cs), stop_cert(g0, cs0)
            if g != g0:
                tot["reply_mismatch"].append(h["id"])
            if sc != sc0:
                tot["stop_mismatch"].append([h["id"], sc, sc0])
            for a, b in zip(cs, cs0):
                if a is None:
                    continue
                tot["pick_mismatch_steps"] += int(a[0] != b[0])
                tot["margin_mismatch_steps"] += int(a[0] == b[0] and a[1] != b[1])
            tot["calls"] += calls; tot["tokens"] += len(g); tot["wall"] += wall
            tot["clean_calls"] += len(g0)
            tot["bitwise"][0] += bw[0]; tot["bitwise"][1] += bw[1]
            tot["accepted"] += sum(a[1] for a in att); tot["attached"] += len(att)
            L.add(f"skip:{name}:{h['id']}", ["output", "skip", name] + (["gamed"] if store == "gamed" else []),
                  premises=[3391], receipt={"history": h["id"], "reply": g, "stop": sc, "calls": calls,
                                            "wall": round(wall, 4), "witnesses": att, "bitwise": bw,
                                            "facts_classes": len(facts.cont)})
        tot["facts_classes"] = len(facts.cont)
        L.add(f"skip-summary:{name}", ["measured", "skip"] + (["gamed"] if "GAMED" in name else []),
              premises=[3391], receipt=tot)
        print(name, json.dumps({k2: (len(v) if isinstance(v, list) and k2.endswith("mismatch") else v)
                                for k2, v in tot.items()}), flush=True)


if __name__ == "__main__":
    origin_run() if sys.argv[1:] == ["origin"] else main()
