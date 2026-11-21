"""Use cases that determinism unlocks, measured against greedy and sampled
decoding on sealed prompts (tasks/USECASE_CONTRACT.json).

  python3 usecases.py CONTRACT_ID
  python3 usecases.py gen OUTFILE      (one fresh process: every method, backend A)
"""
import json, os, subprocess, sys, time
import numpy as np
from common import ROOT, Log, encode, qscores, sha256_file
import decoders as D
from models import GPT2Ort, GPT2Numpy

PROMPTS = json.load(open(os.path.join(ROOT, "tasks", "usecase_prompts.json")))["prompts"]
STEPS = 24
V = 50257
METHODS = ["prlm-0", "greedy-f32", "samp-seeded-t0.7", "samp-seeded-t1.0", "samp-unseeded-t1.0"]
LEAN = "/tmp/claude-0/prlm"


def softmax_t(row, t):
    lp = D.logsoftmax(row / t)
    return np.exp(lp)


def decide(method, row, rng):
    if method == "prlm-0":
        return D.certify(qscores(row))[0]
    if method == "greedy-f32":
        return int(np.argmax(row))
    t = 0.7 if "t0.7" in method else 1.0
    return int(rng.choice(V, p=softmax_t(row, t)))


def rng_for(method, i):
    if method.startswith("samp-unseeded"):
        return np.random.default_rng()          # fresh entropy, as usually served
    return np.random.default_rng(i)


def gen_A(M, method, i, prompt):
    rng = rng_for(method, i)
    c = list(prompt)
    for _ in range(STEPS):
        c.append(decide(method, M.logits(c), rng))
    return c[len(prompt):]


def gen_B(N, method, i, prompt):
    """Backend B: numpy float32 with a KV cache, one new token per call."""
    rng = rng_for(method, i)
    ids = np.asarray(prompt, dtype=np.int64)
    lg, past = N._fwd(ids, None)
    out, rows = [], []
    for _ in range(STEPS):
        row = lg[-1]
        rows.append(row)
        v = decide(method, row, rng)
        out.append(v)
        lg, past = N._fwd(np.asarray([v], dtype=np.int64), past)
    return out, rows


def verify(M, method, i, prompt, out):
    """One batched forward over prompt+out, then the method's rule at every step."""
    if method.startswith("samp-unseeded"):
        return None
    rows = M.all_logits(list(prompt) + list(out))[len(prompt) - 1:len(prompt) - 1 + len(out)]
    rng = rng_for(method, i)
    return all(decide(method, rows[t], rng) == out[t] for t in range(len(out)))


def tamper(M, prompt, out):
    c = list(prompt) + list(out[:12])
    q = qscores(M.logits(c))
    q[out[12]] = np.iinfo(np.int64).min      # the best token other than the honest one
    alt = int(np.argmax(q))
    return out[:12] + [alt] + out[13:]


def metrics(gen):
    tri = [tuple(gen[i:i + 3]) for i in range(len(gen) - 2)]
    bi = [tuple(gen[i:i + 2]) for i in range(len(gen) - 1)]
    rep3 = sum(1 for i, t in enumerate(tri) if t in tri[:i]) / max(1, len(tri))
    return rep3, len(set(bi)) / max(1, len(bi))


def gen_all(outfile):
    M = GPT2Ort(threads=2)
    res = {m: [gen_A(M, m, i, encode(p)) for i, p in enumerate(PROMPTS)] for m in METHODS}
    json.dump(res, open(outfile, "w"))


def main(cid):
    L = Log()
    prem = [cid]
    scr = "/tmp/claude-0/usecases"
    os.makedirs(scr, exist_ok=True)
    L.add("usecase:declared", ["declared"], premises=prem,
          receipt={"usecases_sha256": sha256_file(os.path.abspath(__file__)),
                   "prompts_sha256": sha256_file(os.path.join(ROOT, "tasks", "usecase_prompts.json"))})
    runs = []
    for k in range(2):
        f = os.path.join(scr, f"run{k}.json")
        subprocess.run([sys.executable, os.path.abspath(__file__), "gen", f], check=True)
        runs.append(json.load(open(f)))
    A = runs[0]
    M = GPT2Ort(threads=2)
    P = [encode(p) for p in PROMPTS]
    R = {m: {} for m in METHODS}
    # U4: rerun identical
    for m in METHODS:
        R[m]["U4_rerun_identical"] = sum(a == b for a, b in zip(runs[0][m], runs[1][m]))
    # U1: verify in one call; tamper detection; controls
    for m in METHODS:
        if m.startswith("samp-unseeded"):
            R[m]["U1"] = {"verifiable": False, "verify_calls": None, "honest_accepted": 0, "tamper_detected": 0}
            continue
        c0 = M.forwards
        acc = sum(bool(verify(M, m, i, P[i], A[m][i])) for i in range(len(P)))
        calls = (M.forwards - c0) / len(P)
        det = sum(not verify(M, m, i, P[i], tamper(M, P[i], A[m][i])) for i in range(len(P)))
        R[m]["U1"] = {"verifiable": True, "verify_calls": calls, "regenerate_calls": STEPS,
                      "honest_accepted": acc, "tamper_detected": det}
    controls = {"GAMED_accept_all": {"honest_accepted": len(P), "tamper_detected": 0, "verify_calls": 0},
                "do_nothing": {"honest_accepted": None, "tamper_detected": 0, "verify_calls": 0}}
    # U2: backend B
    N = GPT2Numpy()
    maxdiff, margins = 0, []
    for m in METHODS:
        same = 0
        for i in range(len(P)):
            outB, rowsB = gen_B(N, m, i, P[i])
            if m.startswith("samp-unseeded"):
                continue
            same += outB == A[m][i]
            if m == "prlm-0":
                rowsA = M.all_logits(P[i] + A[m][i])[len(P[i]) - 1:len(P[i]) - 1 + STEPS]
                if outB == A[m][i]:
                    for ra, rb in zip(rowsA, rowsB):
                        qa, qb = qscores(ra), qscores(np.asarray(rb))
                        maxdiff = max(maxdiff, int(np.abs(qa - qb).max()))
                        margins.append(D.certify(qa)[2])
        R[m]["U2_backend_B_identical"] = None if m.startswith("samp-unseeded") else same
    R["prlm-0"]["U2_margin_cert"] = {"max_abs_score_diff_A_vs_B": maxdiff,
                                     "steps_margin_above_2x_diff": int(sum(mg > 2 * maxdiff for mg in margins)),
                                     "steps": len(margins)}
    # U3: cache on a Zipf workload
    rng = np.random.default_rng(3)
    w = 1 / np.arange(1, len(P) + 1) ** 1.1
    work = rng.choice(len(P), size=2000, p=w / w.sum())
    distinct = len(set(work.tolist()))
    for m in METHODS:
        det = not m.startswith("samp-unseeded")
        R[m]["U3"] = {"requests": 2000, "distinct_prompts": distinct,
                      "calls_no_cache": 2000 * STEPS, "calls_with_cache": distinct * STEPS if det else None,
                      "cache_preserves_outputs": det}
    # U5: cost of determinism
    for m in METHODS:
        mt = [metrics(g) for g in A[m]]
        R[m]["U5"] = {"rep3": round(float(np.mean([x[0] for x in mt])), 4), "distinct2": round(float(np.mean([x[1] for x in mt])), 4)}
    # Pareto front over methods (Lean and Python)
    pts = {}
    for m in METHODS:
        u1 = R[m]["U1"]
        vc = 99 if not u1["verifiable"] else u1["verify_calls"]
        ha = 0 if not u1["verifiable"] else 100 * u1["honest_accepted"] / len(P)
        td = 0 if not u1["verifiable"] else 100 * u1["tamper_detected"] / len(P)
        b = R[m]["U2_backend_B_identical"]
        bi = 0 if b is None else 100 * b / len(P)
        cc = R[m]["U3"]["calls_with_cache"] or R[m]["U3"]["calls_no_cache"]
        pts[m] = [int(round(vc)), int(round(100 - ha)), int(round(100 - td)), int(round(100 - bi)),
                  int(round(100 - 100 * R[m]["U4_rerun_identical"] / len(P))), int(cc),
                  int(round(1000 * R[m]["U5"]["rep3"])), int(round(1000 * (1 - R[m]["U5"]["distinct2"])))]
    path = os.path.join(scr, "front.txt")
    with open(path, "w") as f:
        for m, v in pts.items():
            f.write(m + " " + " ".join(map(str, v)) + "\n")
    o = subprocess.run([LEAN, "front", path], capture_output=True, text=True)
    lean_front = sorted(l.split()[1] for l in o.stdout.splitlines() if l.startswith("front "))
    dom = lambda a, b: all(x <= y for x, y in zip(a, b)) and a != b
    py_front = sorted(m for m in pts if not any(dom(pts[n], pts[m]) for n in pts))
    L.add("usecase:result", ["measured", "usecase"], premises=prem,
          receipt={"methods": R, "controls": controls, "pareto_points": pts,
                   "pareto_objectives": ["verify_calls", "100-honest_accept%", "100-tamper_detect%", "100-backendB_identical%",
                                         "100-rerun_identical%", "cache_calls", "rep3x1000", "(1-distinct2)x1000"],
                   "front_lean": lean_front, "front_python": py_front, "lean_exit": o.returncode})
    print(json.dumps({"R": R, "front": lean_front, "py": py_front}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen_all(sys.argv[2])
    else:
        main(int(sys.argv[1]))
