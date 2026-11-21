"""Drafting from prompt and history n-grams (tasks/IMPROVE_CONTRACT.json,
part B). Same 55 histories, reply rule and clean replay as skip.py;
verified mode only, so the output is the rule's own generation
(Prlm/Skip.lean accept_gen) and every drafted token is checked.

The index maps every suffix of length 1..8 to its most recent
occurrence; the draft is the next 8 tokens after the longest match.
Sources: gen (generated tokens only), ctx (+ own prompt), all (+ earlier
prompts), all+corpus (+ TRAIN-corpus documents).
"""
import json, os, time
import skip as S
from common import sha256_file

KMAX, CONT = 8, 8
PREM = [4179]


class Index:
    def __init__(self):
        self.seqs = {}
        self.idx = [dict() for _ in range(KMAX + 1)]

    def add(self, sid, seq, lo, hi):
        self.seqs[sid] = seq
        for p in range(max(lo, 1), hi):
            for k in range(1, KMAX + 1):
                if p - k < 0:
                    break
                self.idx[k][tuple(seq[p - k:p])] = (sid, p)

    def draft(self, c, allowed=lambda sid, p: True):
        for k in range(min(KMAX, len(c)), 0, -1):
            hit = self.idx[k].get(tuple(c[-k:]))
            if hit and allowed(*hit):
                sid, p = hit
                d = self.seqs[sid][p:p + CONT]
                if d:
                    return d, sid, p, k
        return None


def run(h, ix, source, cur, gs_of):
    c = list(h["prompt"])
    gs = len(c)
    gs_of[cur] = gs
    ix.seqs[cur] = c
    if source != "gen":
        ix.add(cur, c, 1, gs)
    gen, certs, origin = [], [], {"same-gen": 0, "same-prompt": 0, "earlier-gen": 0, "earlier-prompt": 0, "corpus": 0}
    f0, t0 = S.M.forwards, time.perf_counter()
    while len(gen) < S.MAXN:
        if source == "ctx":
            dr = ix.draft(c, lambda sid, p: sid == cur or p >= gs_of.get(sid, 0))
        else:
            dr = ix.draft(c)
        draft, sid, p = (dr[0], dr[1], dr[2]) if dr else ([], None, None)
        lg = S.all_logits(c + draft)
        stopped = False
        for i in range(len(draft) + 1):
            row = lg[len(c) - 1]
            v, m, ok, lp = S.step_cert(row)
            certs.append((v, m, bool(ok), lp))
            c.append(v); gen.append(v)
            ix.add(cur, c, len(c) - 1, len(c))
            agree = i < len(draft) and v == draft[i]
            if agree:
                if sid == cur:
                    tag = "same-gen" if p >= gs else "same-prompt"
                elif str(sid).startswith("corpus/"):
                    tag = "corpus"
                else:
                    tag = "earlier-gen" if p + i >= gs_of.get(sid, 0) else "earlier-prompt"
                origin[tag] += 1
            if S.is_stop(v) or len(gen) >= S.MAXN:
                stopped = True
                break
            if not agree:
                break
        if stopped:
            break
    return gen, certs, S.M.forwards - f0, time.perf_counter() - t0, origin


def main():
    H = S.histories()
    recs = [json.loads(l) for l in open(S.LOG)]
    ref = {r["receipt"]["history"]: r["receipt"] for r in recs if r["head"].startswith("skip:clean:")}
    S.L.add("skip2:declared", ["declared"], premises=PREM,
            receipt={"skip2_sha256": sha256_file(os.path.abspath(__file__)), "kmax": KMAX, "cont": CONT})
    for source in ("gen", "ctx", "all", "all+corpus"):
        ix = Index()
        gs_of = {}
        if source == "all+corpus":
            for x in S.T["corpus"]:
                sid = f"corpus/{x['module']}"
                ix.add(sid, list(x["ids"]), 1, len(x["ids"]))
        tot = {"calls": 0, "tokens": 0, "reply_mismatch": [], "stop_mismatch": [], "wall": 0.0,
               "origin": {"same-gen": 0, "same-prompt": 0, "earlier-gen": 0, "earlier-prompt": 0, "corpus": 0}}
        for h in H:
            g, cs, calls, wall, org = run(h, ix, source, h["id"], gs_of)
            if g != ref[h["id"]]["reply"]:
                tot["reply_mismatch"].append(h["id"])
            if S.stop_cert(g, cs) != ref[h["id"]]["stop"]:
                tot["stop_mismatch"].append(h["id"])
            tot["calls"] += calls; tot["tokens"] += len(g); tot["wall"] += wall
            for k, v in org.items():
                tot["origin"][k] += v
            S.L.add(f"skip2:{source}:{h['id']}", ["output", "skip2", source], premises=PREM,
                    receipt={"history": h["id"], "reply": g, "stop": S.stop_cert(g, cs), "calls": calls,
                             "wall": round(wall, 4), "origin": org})
        tot["wall"] = round(tot["wall"], 3)
        S.L.add(f"skip2-summary:{source}", ["measured", "skip2"], premises=PREM, receipt=tot)
        print(source, json.dumps({k: (len(v) if k.endswith("mismatch") else v) for k, v in tot.items()}), flush=True)


if __name__ == "__main__":
    main()
