"""Tables for the multi-model, architecture-parametric and density runs,
rebuilt from out/log.jsonl alone. Writes out/REPORT-multi.md."""
import json, os
from common import ROOT, LOG, verify_log

n, head = verify_log()
recs = [json.loads(l) for l in open(LOG)]
lines = [f"# Multi-model, architecture and density report (rebuilt from the log)\n\nLog: {n} records, chain head {head[:16]}.\n"]
P = lines.append


def f(x):
    return f"{x['bitwise']}/{x['int_scores']}/{x['picks']} of {x['rows']}"


multi = [r["receipt"] for r in recs if r["head"].startswith("multi:") and "model" in r["receipt"]]
if multi:
    P("## Six models (tasks/MULTI_CONTRACT.json)\n")
    P("Cells are bitwise / equal integer scores / equal picks.\n")
    P("| model | repeat in process | fresh 1 thread | fresh 4 threads | one call vs per step | cached vs recompute | Lean agree, sparseEq | tamper exit |")
    P("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for R in multi:
        kv = f(R["E3_kv_vs_recompute"]) if "E3_kv_vs_recompute" in R else "n/a"
        P(f"| {R['model']} | {f(R['E1_inprocess'])} | {f(R['E1_fresh_t1'])} | {f(R['E1_fresh_t4'])} | {f(R['E3_batch_vs_step'])} | {kv} | {R['E2_lean']['agree']}, {R['E2_lean']['sparseEq']} of {R['E2_lean']['steps']} | {R['E2_tamper_exit']} |")
    P("\n| model | clean calls | drafting calls | reply mismatches | certificate mismatches | do-nothing calls | GAMED calls | drift vs fp32 picks | equal sequences |")
    P("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for R in multi:
        d = R.get("E5_drift")
        dd = f"{d['picks']}/{d['steps']}" if d else "n/a"
        ds = f"{d['sequences_equal']}/{d['histories']}" if d else "n/a"
        P(f"| {R['model']} | {R['clean_calls']} | {R['E4_ctx']['calls']} | {R['E4_ctx']['reply_mismatch']} | {R['E4_ctx']['cert_mismatch']} | {R['E4_none']['calls']} | {R['E4_gamed']['calls']} | {dd} | {ds} |")

arch = [r["receipt"] for r in recs if r["head"].startswith("arch:") and r["head"].endswith((":random", ":trained"))]
if arch:
    P("\n## Architecture as a parameter (tasks/ARCH_CONTRACT.json)\n")
    P("| config | weights | repeat / fresh 1t / fresh 4t | full call vs step | incremental vs step | Lean agree, sparseEq | tamper | drafting calls (clean 840), reply / cert mismatches |")
    P("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for R in arch:
        x = R["X4_ctx"]
        P(f"| {R['config']} {R['name']} | {R['weights']} | {f(R['X1_inprocess'])}; {f(R['X1_fresh_t1'])}; {f(R['X1_fresh_t4'])} | {f(R['X3_full_vs_step'])} | {f(R['X3_incremental_vs_step'])} | {R['X2_lean']['agree']}, {R['X2_lean']['sparseEq']} of {R['X2_lean']['steps']} | {R['X2_tamper_exit']} | {x['calls']}, {x['reply_mismatch']} / {x['cert_mismatch']} |")
    P("\nTrained tiny models (600 Adam steps on the 15,685-token TRAIN corpus), T1 with report.py's metrics:\n")
    P("| config | params | final train loss | ref_match /24 | ref_f1 | rep3 | distinct2 |")
    P("| --- | --- | --- | --- | --- | --- | --- |")
    for R in arch:
        if R["weights"] == "trained":
            q = R["Q_t1"]
            P(f"| {R['config']} {R['name']} | {R['n_params']} | {R['train']['loss_last50_mean']} | {q['ref_match']:.2f} | {q['ref_f1']:.3f} | {q['rep3']:.3f} | {q['distinct2']:.3f} |")

pts = next((r["receipt"] for r in recs if r["head"] == "density:points"), None)
frs = next((r["receipt"] for r in recs if r["head"] == "density:fronts"), None)
ln = next((r["receipt"] for r in recs if r["head"] == "density:lean"), None)
if pts and frs:
    P("\n## Density as the computed consequence of the weight change (tasks/DENSITY_CONTRACT.json)\n")
    P(f"{pts['decisions']} step decisions (243 weight changes x {pts['decisions'] // 243} steps): sparse rule vs dense recompute disagreed on {pts['sparse_vs_dense_disagreements']}.")
    P(f"Fronts recomputed after each of {frs['appends']} appends; every new front was within the previous front plus the new point: {frs['all_monotone']}; Lean exit 0 every time: {frs['all_exit0']}.\n")
    P("Final front (minimize work, ban hits, topic misses on held-out words, end misses, loops, drift):\n")
    P("| point (weights in logits) | density | work/step | ban hits | topic misses | end misses | loops | drift | Chebyshev weights aimed here pick | Lean agree, sparseEq |")
    P("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for fp in frs["final_front"]:
        r = pts["points"][fp["point"]]
        lv = ln.get(fp["point"], {}) if ln else {}
        lean = f"{lv.get('agree')}, {lv.get('sparseEq')} of {lv.get('steps')}" if lv else "n/a"
        d = ", ".join(f"{a}{v}" for a, v in r["delta"].items() if v) or "none"
        P(f"| {d} | {r['density_mean']:.5f} | {r['work_mean']} | {r['ban_hits']} | {r['topic_misses']} | {r['end_misses']} | {r['loops']} | {r['drift']} | {', '.join(fp['chebmin'])} | {lean} |")

lr = next((r["receipt"] for r in recs if r["head"] == "loop:result"), None)
if lr:
    rnd = [r["receipt"] for r in recs if r["head"].startswith("loop:round:")]
    rc = next(r["receipt"] for r in recs if r["head"] == "loop:control:random")
    gc = next(r["receipt"] for r in recs if r["head"] == "loop:control:GAMED")
    lv = next(r["receipt"] for r in recs if r["head"] == "loop:lean")
    P("\n## Self-improving loop (tasks/LOOP_CONTRACT.json)\n")
    P("| round | evaluated | admitted | new on front | front size | Lean front = archive | frozen axes |")
    P("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rnd:
        P(f"| {r['round']} | {r['evaluated']} | {r['admitted']} | {r['new_front']} | {r['front']} | {r['lean_equals_archive']} | {', '.join(r['frozen']) or 'none'} |")
    P(f"\nOn TEST (strict, weak coverage): final covers initial {lr['test_final_covers_initial']}, initial covers final {lr['test_initial_covers_final']}; final points not dominated by the initial front {lr['test_final_not_dominated_by_initial']}. Sparse vs dense disagreements: {lr['sparse_vs_dense_disagreements']}.")
    P(f"Random search, same {rc['evaluations']} evaluations: front {rc['dev_front']}; loop covers random {rc['test_loop_covers_random']}, random covers loop {rc['test_random_covers_loop']}.")
    P(f"GAMED (selected on TEST): front {gc['test_front']}; GAMED covers loop {gc['test_gamed_covers_loop']}, loop covers GAMED {gc['test_loop_covers_gamed']}.")
    P(f"Lean replay of the {len(lv['points'])} final front points on 1,440 steps: all exact = {lv['all_exact']}.")

open(os.path.join(ROOT, "out", "REPORT-multi.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
