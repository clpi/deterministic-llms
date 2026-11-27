"""Value amendment 5 (tasks/VALUE_AMEND_5.json, record 4994): every completion ends at
its first end-of-document token. Rescores the stored token paths of every value run.

  python value_eos.py OUTDIR DIR...

Writes the corrected rows to OUTDIR/<k>/ (one folder per input DIR, same file names)
and OUTDIR/EOS.json: per row and method, the do-nothing control (the old rule on the
re-decoded tokens must reproduce every stored answer or program), accuracy before and
after, paths with an end-of-document token, paths changed (wrong->right, right->wrong),
generated tokens before and after the cut, and branch points past the parent's end of
document (CONTAMINATED rows, regenerated under the amendment). No model is run.
"""
import copy, glob, json, os, sys
import value as V

BRANCH = ("margin-branch-8", "random-branch-8", "margin-branch-8-filter", "random-branch-8-filter")
_T = {}


def tokenizer():
    if "t" not in _T:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
        eos = {t.convert_tokens_to_ids(s) for s in ("<|endoftext|>", "<|im_end|>")}
        eos.discard(None)
        eos.discard(t.unk_token_id)
        _T["t"] = (lambda ids: t.decode(list(ids), skip_special_tokens=False), eos)
    return _T["t"]


def first_eos(tokens, eos):
    return next((i for i, x in enumerate(tokens) if x in eos), None)


def starts(m, row):
    """Position where each path's own generation starts (branch paths: after the prefix
    and the second-ranked token)."""
    n = len(row["paths"])
    if m not in BRANCH:
        return [0] * n
    pts = row.get("branch_points") or []
    return [0] + [pts[j - 1][1] + 1 if j - 1 < len(pts) else 0 for j in range(1, n)]


def gen_after(paths, st, eos):
    tot = 0
    for p, s in zip(paths, st):
        e = first_eos(p, eos)
        end = len(p) if e is None else e + 1
        tot += max(0, end - s)
    return tot


def past_eos(m, row, eos):
    if m not in BRANCH:
        return 0
    n = 0
    for i, t, _ in row.get("branch_points") or []:
        e = first_eos(row["paths"][i], eos)
        n += int(e is not None and t > e)
    return n


def fix_math(r, dec, eos, st):
    for m, M in r["methods"].items():
        s = st.setdefault(m, {"paths": 0, "eos_paths": 0, "changed": 0, "wrong_to_right": 0, "right_to_wrong": 0,
                              "control_mismatch": 0, "correct_before": 0, "correct_after": 0,
                              "gen_before": 0, "gen_after": 0, "branch_past_eos": 0, "rows_with_past_eos": 0})
        new = []
        for k, p in enumerate(M["paths"]):
            old_stored = M["answers"][k]
            old = V.extract(dec(p))[0]
            if not ((old is None and old_stored is None) or V.same(old, old_stored)):
                s["control_mismatch"] += 1
            e = first_eos(p, eos)
            a = old_stored if e is None else V.extract(dec(p[:e]))[0]
            s["paths"] += 1
            s["eos_paths"] += int(e is not None)
            ch = not ((a is None and old_stored is None) or V.same(a, old_stored))
            s["changed"] += int(ch)
            if ch:
                s["wrong_to_right"] += int(not V.same(old_stored, r["gold"]) and V.same(a, r["gold"]))
                s["right_to_wrong"] += int(V.same(old_stored, r["gold"]) and not V.same(a, r["gold"]))
            new.append(a)
        pe = past_eos(m, M, eos)
        s["branch_past_eos"] += pe
        s["rows_with_past_eos"] += int(pe > 0)
        s["correct_before"] += int(M["correct"])
        s["gen_before"] += M["gen_tokens"]
        M["answers_uncorrected"] = M["answers"]
        M["answers"] = new
        M["pred"] = V.vote(new) if len(new) > 1 else new[0]
        M["correct"] = V.same(M["pred"], r["gold"])
        M["oracle"] = any(V.same(a, r["gold"]) for a in new)
        M["gen_tokens_uncorrected"] = M["gen_tokens"]
        M["gen_tokens"] = gen_after(M["paths"], starts(m, M), eos)
        M["branch_past_eos"] = pe
        s["correct_after"] += int(M["correct"])
        s["gen_after"] += M["gen_tokens"]


def fix_mbpp(r, dec, eos, st, tests):
    import value_code as C
    t = tests[r["index"]]
    vis, allt, imp = t["test_list"][:1], t["test_list"], t["test_imports"]
    for m, M in r["methods"].items():
        s = st.setdefault(m, {"paths": 0, "eos_paths": 0, "changed": 0, "wrong_to_right": 0, "right_to_wrong": 0,
                              "control_mismatch": 0, "correct_before": 0, "correct_after": 0,
                              "gen_before": 0, "gen_after": 0, "branch_past_eos": 0, "rows_with_past_eos": 0})
        progs, full = list(M["programs"]), list(M["full"])
        visible = list(M["visible"]) if M.get("visible") is not None else None
        for k, p in enumerate(M["paths"]):
            if C.program(dec(p)) != M["programs"][k]:
                s["control_mismatch"] += 1
            e = first_eos(p, eos)
            s["paths"] += 1
            s["eos_paths"] += int(e is not None)
            if e is None:
                continue
            q = C.program(dec(p[:e]))
            if q == progs[k]:
                continue
            s["changed"] += 1
            progs[k] = q
            f = C.runs(q, imp, allt)
            s["wrong_to_right"] += int(f and not full[k])
            s["right_to_wrong"] += int(full[k] and not f)
            full[k] = f
            if visible is not None:
                visible[k] = C.runs(q, imp, vis)
        sel = next((k for k, v in enumerate(visible) if v), 0) if visible else 0
        pe = past_eos(m, M, eos)
        s["branch_past_eos"] += pe
        s["rows_with_past_eos"] += int(pe > 0)
        s["correct_before"] += int(M["pass"])
        s["gen_before"] += M["gen_tokens"]
        changed_sel = progs[sel] != M["programs"][M["selected"]] or sel != M["selected"]
        M["uncorrected"] = {k: M[k] for k in ("selected", "pass", "visible_pass_selected", "visible", "full", "programs", "gen_tokens")}
        M["programs"], M["full"], M["visible"], M["selected"] = progs, full, visible, sel
        M["pass"] = full[sel]
        if changed_sel:
            M["visible_pass_selected"] = visible[sel] if visible is not None else C.runs(progs[sel], imp, vis)
        M["oracle"] = any(full)
        M["gen_tokens"] = gen_after(M["paths"], starts(m, M), eos)
        M["branch_past_eos"] = pe
        s["correct_after"] += int(M["pass"])
        s["gen_after"] += M["gen_tokens"]


def main(outdir, dirs):
    os.makedirs(outdir, exist_ok=True)
    tests, stats = None, {}
    for n, d in enumerate(dirs):
        for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
            rel = os.path.relpath(p, d)
            dst = os.path.join(outdir, f"r{n + 1}", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w") as out:
                for l in open(p):
                    if not l.strip():
                        continue
                    r = json.loads(l)
                    task, repo = r.get("task", "gsm8k"), r["inputs"]["repo"]
                    key = f"{task}:{repo}" + (":again" if "again" in os.path.abspath(p) else "")
                    st = stats.setdefault(key, {"rows": 0, "methods": {}})
                    st["rows"] += 1
                    if "qwen" in repo.lower():
                        dec, eos = tokenizer()
                        if task == "mbpp":
                            if tests is None:
                                import value_code as C
                                tests = {q["index"]: q for q in C.mbpp_load(os.environ.get("VALUE_MBPP_DATA"))}
                            fix_mbpp(r, dec, eos, st["methods"], tests)
                        else:
                            fix_math(r, dec, eos, st["methods"])
                        r["eos_corrected"] = "VALUE_AMEND_5"
                    else:
                        st["skipped"] = "not a Qwen tokenizer"
                    out.write(json.dumps(r) + "\n")
    for k, st in stats.items():
        n = st["rows"]
        for m, s in st["methods"].items():
            s["accuracy_before"] = round(s["correct_before"] / n, 4)
            s["accuracy_after"] = round(s["correct_after"] / n, 4)
        st["control_holds"] = all(s["control_mismatch"] == 0 for s in st["methods"].values())
        st["contaminated"] = sorted(m for m, s in st["methods"].items() if s["branch_past_eos"] > 0)
    json.dump(stats, open(os.path.join(outdir, "EOS.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
