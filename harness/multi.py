"""Multi-model runs (tasks/MULTI_CONTRACT.json, log record 4410).

  python3 multi.py run [NAME ...]        all measures, logged
  python3 multi.py dump NAME THREADS OUT  clean rows of the first 8 histories
                                         in a fresh process (E1)
"""
import json, os, struct, subprocess, sys, time

if len(sys.argv) > 3 and sys.argv[1] == "dump":  # thread count must be set before numpy loads
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = sys.argv[3]

import numpy as np
from common import ROOT, Log, qscores, sha256_file
import axes as AX
import decoders as D
import models as MM

T = json.load(open(os.path.join(ROOT, "tasks", "tasks.json")))
H = T["t1"]
STEPS, KMAX, CONT, E1N = 24, 8, 8, 8
PREM = [4410]
LEAN = "/tmp/claude-0/prlm"
SCR = "/tmp/claude-0/replay3"
THETA = AX.weights()
DELTA = AX.weights(rep=1.5, noloop=4.0)
ACTIVE = ["rep", "noloop"]


def cert(row):
    v, s, m, ok = D.certify(qscores(row))
    return v, m, bool(ok)


def clean(m, it):
    c = m.start(it["prompt"])
    gs, gen, certs, rows = len(c), [], [], []
    for _ in range(STEPS):
        row = m.logits(c)
        v, mg, ok = cert(row)
        rows.append(row)
        certs.append((v, mg, ok))
        gen.append(v)
        c.append(v)
        if v == m.eos:
            break
    return {"c": c, "gs": gs, "gen": gen, "certs": certs, "rows": rows}


def cmp_rows(a, b):
    """a, b: lists/arrays of float32 rows. bitwise, integer-score, pick equal counts."""
    bw = sum(bool(np.array_equal(x, y)) for x, y in zip(a, b))
    iq = sum(bool(np.array_equal(qscores(x), qscores(y))) for x, y in zip(a, b))
    pk = sum(cert(x)[0] == cert(y)[0] for x, y in zip(a, b))
    md = max(float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in zip(a, b))
    return {"rows": len(a), "bitwise": bw, "int_scores": iq, "picks": pk, "max_abs_diff": md}


class Index:  # skip2's longest-suffix index, one history at a time ("ctx")
    def __init__(self):
        self.seq, self.idx = None, [dict() for _ in range(KMAX + 1)]

    def add(self, seq, lo, hi):
        self.seq = seq
        for p in range(max(lo, 1), hi):
            for k in range(1, KMAX + 1):
                if p - k < 0:
                    break
                self.idx[k][tuple(seq[p - k:p])] = p

    def draft(self, c):
        for k in range(min(KMAX, len(c)), 0, -1):
            p = self.idx[k].get(tuple(c[-k:]))
            if p is not None:
                d = self.seq[p:p + CONT]
                if d:
                    return d
        return []


def skip_run(m, it, mode, ref):
    """mode: ctx (verified drafting), none (do-nothing control), gamed
    (draft = the clean replay's own continuation, reads the answer)."""
    c = m.start(it["prompt"])
    gs = len(c)
    ix = Index()
    if mode == "ctx":
        ix.add(c, 1, gs)
    gen, certs = [], []
    f0 = m.forwards
    while len(gen) < STEPS:
        if mode == "ctx":
            dr = ix.draft(c)
        elif mode == "gamed":
            dr = ref["gen"][len(gen):len(gen) + CONT]
        else:
            dr = []
        lg = m.all_logits(c + dr)
        stop = False
        for i in range(len(dr) + 1):
            v, mg, ok = cert(lg[len(c) - 1])
            certs.append((v, mg, ok))
            c.append(v)
            gen.append(v)
            if mode == "ctx":
                ix.add(c, len(c) - 1, len(c))
            if v == m.eos or len(gen) >= STEPS:
                stop = True
                break
            if not (i < len(dr) and v == dr[i]):
                break
        if stop:
            break
    return {"gen": gen, "certs": certs, "calls": m.forwards - f0}


def dense(ents, theta, V):
    out = np.zeros(V, dtype=np.int64)
    for t, a, v in ents:
        out[t] += theta[a] * v
    return out


def write_replay(path, m, hist):
    with open(path, "wb") as f:
        for h in hist:
            for i, row in enumerate(h["rows"]):
                c = h["c"][:h["gs"] + i]
                base = qscores(row)
                ents = AX.features(c, h["gs"], None, ACTIVE)
                v = D.certify(base + dense(ents, THETA, m.V))[0]
                th2 = [a + b for a, b in zip(THETA, DELTA)]
                v2 = D.certify(base + dense(ents, th2, m.V))[0]
                f.write(struct.pack("<II", m.V, len(THETA)))
                f.write(np.asarray(base, dtype="<i4").tobytes())
                f.write(struct.pack(f"<{len(THETA)}i", *THETA))
                f.write(struct.pack(f"<{len(DELTA)}i", *DELTA))
                f.write(struct.pack("<I", len(ents)))
                if ents:
                    f.write(np.asarray(ents, dtype="<i4").tobytes())
                f.write(struct.pack("<ii", v, v2))


def lean(path):
    out = subprocess.run([LEAN, "decide", path], capture_output=True, text=True)
    line = out.stdout.strip().splitlines()[-1]
    return out.returncode, dict(zip(line.split()[0::2], map(int, line.split()[1::2])))


def tamper(src, dst):
    """First record only, harness pick +1: the Lean replay must exit 1."""
    b = open(src, "rb").read()
    n, k = struct.unpack_from("<II", b, 0)
    o = 8 + 4 * n + 8 * k
    nf = struct.unpack_from("<I", b, o)[0]
    o += 4 + 12 * nf
    v, v2 = struct.unpack_from("<ii", b, o)
    rec = bytearray(b[:o + 8])
    struct.pack_into("<ii", rec, o, v + 1, v2)
    open(dst, "wb").write(bytes(rec))


def dump(name, threads, out):
    m = MM.make(name, threads=int(threads))
    rows = [np.stack(clean(m, it)["rows"]) for it in H[:E1N]]
    np.savez(out, *rows)


def run(names):
    os.makedirs(SCR, exist_ok=True)
    L = Log()
    L.add("multi:declared", ["declared"], premises=PREM,
          receipt={"multi_sha256": sha256_file(os.path.abspath(__file__)),
                   "models_sha256": sha256_file(os.path.join(ROOT, "harness", "models.py")),
                   "lean_exe_sha256": sha256_file(LEAN), "names": names})
    ref = None
    for name in names:
        t0 = time.perf_counter()
        m = MM.make(name)
        R = {"model": name, "V": m.V}
        hist = [clean(m, it) for it in H]
        R["clean_calls"] = sum(len(h["gen"]) for h in hist)
        R["tokens"] = R["clean_calls"]
        R["certified"] = sum(ok for h in hist for _, _, ok in h["certs"])
        R["clean_wall"] = round(time.perf_counter() - t0, 2)

        # E1: repeat in process, then fresh processes at 1 and 4 threads
        again = [clean(m, it)["rows"] for it in H[:E1N]]
        base = [r for h in hist[:E1N] for r in h["rows"]]
        R["E1_inprocess"] = cmp_rows(base, [r for rs in again for r in rs])
        for th in (1, 4):
            out = os.path.join(SCR, f"{name}-t{th}.npz")
            subprocess.run([sys.executable, os.path.abspath(__file__), "dump", name, str(th), out], check=True)
            z = np.load(out)
            R[f"E1_fresh_t{th}"] = cmp_rows(base, [r for i in range(E1N) for r in z[f"arr_{i}"]])

        # E3: one forward over the final sequence, and the cached path
        full, kv, rows = [], [], []
        for it, h in zip(H, hist):
            m.start(it["prompt"])  # T5 keeps the history's encoder state
            n = len(h["gen"])
            full += list(m.all_logits(h["c"])[h["gs"] - 1:h["gs"] - 1 + n])
            rows += h["rows"]
            if hasattr(m, "kv_rows"):
                kv += list(m.kv_rows(h["c"], h["gs"]))
        R["E3_batch_vs_step"] = cmp_rows(rows, full)
        if kv:
            R["E3_kv_vs_recompute"] = cmp_rows(rows, kv)

        # E2: Lean replay of the clean trajectories, plus a tamper control
        path = os.path.join(SCR, f"{name}.bin")
        write_replay(path, m, hist)
        rc, kvs = lean(path)
        R["E2_lean"] = {"exit": rc, "file_sha256": sha256_file(path), **kvs}
        tamper(path, path + ".tampered")
        R["E2_tamper_exit"] = lean(path + ".tampered")[0]

        # E4: verified drafting vs clean, with both controls
        for mode in ("ctx", "none", "gamed"):
            rm = cm = calls = 0
            mism = []
            for it, h in zip(H, hist):
                s = skip_run(m, it, mode, h)
                calls += s["calls"]
                if s["gen"] != h["gen"]:
                    rm += 1
                    mism.append(it["id"])
                if s["certs"] != h["certs"]:
                    cm += 1
            R[f"E4_{mode}"] = {"calls": calls, "reply_mismatch": rm, "cert_mismatch": cm, "mismatched": mism}

        # E5: drift vs fp32, teacher-forced on the fp32 trajectories
        if name == "gpt2-fp32-ort":
            ref = hist
        if ref is not None and name.startswith("gpt2"):
            pk = iq = okq = n = 0
            seq_eq = sum(h["gen"] == r["gen"] for h, r in zip(hist, ref))
            for r in ref:
                for i, rrow in enumerate(r["rows"]):
                    row = m.logits(r["c"][:r["gs"] + i])
                    a, b = cert(row), r["certs"][i]
                    pk += a[0] == b[0]
                    okq += a[2] == b[2]
                    iq += bool(np.array_equal(qscores(row), qscores(rrow)))
                    n += 1
            R["E5_drift"] = {"steps": n, "picks": pk, "int_scores": iq, "certified_flag": okq,
                             "sequences_equal": seq_eq, "histories": len(ref)}
        R["wall"] = round(time.perf_counter() - t0, 2)
        L.add(f"multi:{name}", ["measured", "multi"], premises=PREM, receipt=R)
        print(name, json.dumps({k: v for k, v in R.items()}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        run(sys.argv[2:] or MM.NAMES)
