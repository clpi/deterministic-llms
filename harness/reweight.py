"""Exact sparse reweighting on real steps (tasks/IMPROVE_CONTRACT.json, part A).
Regenerates replay files and runs the Lean executable, which checks the
proved best_split rule against a full recompute at every step and counts
the work."""
import json, os, subprocess, time
from common import *
import axes as AX, decoders as D

T = json.load(open(os.path.join(ROOT, "tasks", "tasks.json")))
L = Log()
M = Model(threads=4)
SCR = "/tmp/claude-0/replay2"
os.makedirs(SCR, exist_ok=True)
LEAN = "/tmp/claude-0/prlm"
PREM = [4179]


def regen(name, items, delta_kw, active):
    path = os.path.join(SCR, name + ".bin")
    rp = D.Replay(path)
    dl = AX.weights(**delta_kw)
    for it in items:
        M.cache.clear()
        req = AX.Request(it)
        D.prlm(M, it["prompt"], 24, req, AX.weights(), active, replay=rp, delta=dl)
    rp.close()
    t0 = time.perf_counter()
    out = subprocess.run([LEAN, "decide", path], capture_output=True, text=True)
    line = out.stdout.strip().splitlines()[-1]
    kv = dict(zip(line.split()[0::2], map(int, line.split()[1::2])))
    L.add(f"sparse:{name}", ["lean-replay", "measured", "sparse"], premises=PREM,
          receipt={"exit": out.returncode, "file_sha256": sha256_file(path), "bytes": os.path.getsize(path),
                   "lean_exe_sha256": sha256_file(LEAN), "delta": delta_kw, "seconds": round(time.perf_counter() - t0, 2), **kv})
    print(name, line, flush=True)
    return path


if __name__ == "__main__":
    t3 = {k: [t for t in T["t3"] if t["kind"] == k] for k in ("topic", "end", "ban")}
    regen("t1-theta", T["t1"], dict(rep=1.5, noloop=4.0), ["rep", "noloop"])
    regen("t3-topic", t3["topic"], dict(topic=4.0), ["topic"])
    regen("t3-end", t3["end"], dict(end=4.0), ["end"])
    regen("t3-ban", t3["ban"], dict(ban=8.0), ["ban"])
