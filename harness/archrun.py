"""Architecture-parametric runs (tasks/ARCH_CONTRACT.json).

  python3 archrun.py run [KEY ...]               random and trained weights, logged
  python3 archrun.py dump KEY WEIGHTS THREADS OUT fresh-process rows (X1)
"""
import hashlib, json, os, subprocess, sys, time

if len(sys.argv) > 4 and sys.argv[1] == "dump":
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = sys.argv[4]

import numpy as np
import arch
from common import ROOT, Log, sha256_file
import multi as MU

T = json.load(open(os.path.join(ROOT, "tasks", "tasks.json")))
TYPES = sorted({t for x in T["corpus"] for t in x["ids"]})
V = len(TYPES) + 1
UNK = V - 1
TO = {t: i for i, t in enumerate(TYPES)}
DATA = [TO[t] for x in T["corpus"] for t in x["ids"]]
SCR = "/tmp/claude-0/arch"
PREM = []  # set to the contract record id at run time


class ArchModel:
    eos = -1

    def __init__(self, key, params, tag):
        self.cfg, self.p, self.V = arch.CONFIGS[key], params, V
        self.name, self.forwards = f"{key}:{tag}", 0

    def start(self, prompt):
        return [TO.get(t, UNK) for t in prompt]

    def all_logits(self, c):
        self.forwards += 1
        return arch.full(np, self.p, self.cfg, np.asarray(c, dtype=np.int64))

    def logits(self, c):
        return self.all_logits(c)[-1].copy()

    def kv_rows(self, c, n0):
        cache, rows = arch.new_cache(self.cfg), []
        for i, tok in enumerate(c[:-1]):
            r = arch.step(self.p, self.cfg, cache, int(tok), i)
            if i >= n0 - 1:
                rows.append(r)
        return np.stack(rows)


def params_for(key, tag):
    if tag == "random":
        return arch.init(arch.CONFIGS[key], V)
    z = np.load(os.path.join(SCR, f"{key}-trained.npz"))
    return {k: z[k] for k in z.files}


def psha(p):
    h = hashlib.sha256()
    for k in sorted(p):
        h.update(k.encode())
        h.update(np.ascontiguousarray(p[k]).tobytes())
    return h.hexdigest()


def dump(key, tag, out):
    m = ArchModel(key, params_for(key, tag), tag)
    np.savez(out, *[np.stack(MU.clean(m, it)["rows"]) for it in MU.H[:MU.E1N]])


def to_gpt2(gen):
    return [TYPES[t] if t < len(TYPES) else -1 for t in gen]


def t1m(g, it):  # report.py's T1 metrics, same definitions
    from collections import Counter
    ref = it["reference"]
    match = sum(int(a == b) for a, b in zip(g, ref))
    ov = sum((Counter(g) & Counter(ref)).values())
    f1 = 0 if ov == 0 else 2 * ov / (len(g) + len(ref))
    ngr = lambda x, n: [tuple(x[i:i + n]) for i in range(len(x) - n + 1)]
    tri, bi = ngr(g, 3), ngr(g, 2)
    rep3 = sum(1 for i, t in enumerate(tri) if t in tri[:i]) / max(1, len(tri))
    return {"ref_match": match, "ref_f1": f1, "rep3": rep3, "distinct2": len(set(bi)) / max(1, len(bi))}


def measure(key, tag, params):
    m = ArchModel(key, params, tag)
    R = {"config": key, "name": arch.CONFIGS[key]["name"], "weights": tag, "V": V, "params_sha256": psha(params),
         "n_params": int(sum(v.size for v in params.values()))}
    t0 = time.perf_counter()
    hist = [MU.clean(m, it) for it in MU.H]
    R["steps"] = sum(len(h["gen"]) for h in hist)
    R["certified"] = sum(ok for h in hist for _, _, ok in h["certs"])
    base = [r for h in hist[:MU.E1N] for r in h["rows"]]
    R["X1_inprocess"] = MU.cmp_rows(base, [r for it in MU.H[:MU.E1N] for r in MU.clean(m, it)["rows"]])
    for th in (1, 4):
        out = os.path.join(SCR, f"{key}-{tag}-t{th}.npz")
        subprocess.run([sys.executable, os.path.abspath(__file__), "dump", key, tag, str(th), out], check=True)
        z = np.load(out)
        R[f"X1_fresh_t{th}"] = MU.cmp_rows(base, [r for i in range(MU.E1N) for r in z[f"arr_{i}"]])
    rows, full, inc = [], [], []
    for h in hist:
        n = len(h["gen"])
        rows += h["rows"]
        full += list(m.all_logits(h["c"])[h["gs"] - 1:h["gs"] - 1 + n])
        inc += list(m.kv_rows(h["c"], h["gs"]))
    R["X3_full_vs_step"] = MU.cmp_rows(rows, full)
    R["X3_incremental_vs_step"] = MU.cmp_rows(rows, inc)
    path = os.path.join(SCR, f"{key}-{tag}.bin")
    MU.write_replay(path, m, hist)
    rc, kv = MU.lean(path)
    R["X2_lean"] = {"exit": rc, "file_sha256": sha256_file(path), **kv}
    MU.tamper(path, path + ".tampered")
    R["X2_tamper_exit"] = MU.lean(path + ".tampered")[0]
    for mode in ("ctx", "none", "gamed"):
        rm = cm = calls = 0
        mism = []
        for it, h in zip(MU.H, hist):
            s = MU.skip_run(m, it, mode, h)
            calls += s["calls"]
            if s["gen"] != h["gen"]:
                rm += 1
                mism.append(it["id"])
            cm += s["certs"] != h["certs"]
        R[f"X4_{mode}"] = {"calls": calls, "reply_mismatch": rm, "cert_mismatch": cm, "mismatched": mism}
    if tag == "trained":
        q = [t1m(to_gpt2(h["gen"]), it) for it, h in zip(MU.H, hist)]
        R["Q_t1"] = {k: round(float(np.mean([x[k] for x in q])), 4) for k in q[0]}
        R["Q_unk_emitted"] = sum(t == UNK for h in hist for t in h["gen"])
    R["wall"] = round(time.perf_counter() - t0, 2)
    return R


def run(keys, contract_id):
    os.makedirs(SCR, exist_ok=True)
    prem = [contract_id]
    L = Log()
    L.add("arch:declared", ["declared"], premises=prem,
          receipt={"arch_sha256": sha256_file(os.path.join(ROOT, "harness", "arch.py")),
                   "archrun_sha256": sha256_file(os.path.abspath(__file__)),
                   "multi_sha256": sha256_file(os.path.join(ROOT, "harness", "multi.py")),
                   "V": V, "corpus_tokens": len(DATA), "keys": keys})
    for key in keys:
        R = measure(key, "random", arch.init(arch.CONFIGS[key], V))
        L.add(f"arch:{key}:random", ["measured", "arch"], premises=prem, receipt=R)
        print(json.dumps(R), flush=True)
        t0 = time.perf_counter()
        p, losses = arch.train(arch.CONFIGS[key], V, DATA)
        np.savez(os.path.join(SCR, f"{key}-trained.npz"), **p)
        tr = {"steps": len(losses), "loss_first": round(losses[0], 4), "loss_last50_mean": round(float(np.mean(losses[-50:])), 4),
              "seconds": round(time.perf_counter() - t0, 1), "params_sha256": psha(p)}
        L.add(f"arch:{key}:train", ["trained", "arch"], premises=prem, receipt=tr)
        R = measure(key, "trained", p)
        R["train"] = tr
        L.add(f"arch:{key}:trained", ["measured", "arch"], premises=prem, receipt=R)
        print(json.dumps(R), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3], sys.argv[5])
    else:
        cid = int(sys.argv[2])
        run(sys.argv[3:] or list(arch.CONFIGS), cid)
