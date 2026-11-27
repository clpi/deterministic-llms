"""Value amendment 3 (tasks/VALUE_AMEND_3.json, record 4992): the portfolio decoder.

  python value_portfolio.py OUTFILE DIR...      DIRs hold the value, value2 and value3 result files

Reuses every path the earlier runs generated; no model is run. Theorem P1 is
lean/Prlm/Portfolio.lean (portfolio_dominates_all).
"""
import ast, glob, json, operator, os, re, sys
import value as V

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def arith(expr):
    """Evaluate + - * / and parentheses over numbers only (no eval of model text)."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        raise ValueError("not arithmetic")
    return ev(ast.parse(expr, mode="eval"))

ORDER = ["prlm-0", "greedy", "sample", "SAMPLES", "BRANCH", "RANDOM"]
EQ = re.compile(r"((?:[\d.,$]+|[-+*/x×()]|\s)+?)=\s*\$?(-?[\d,]*\.?\d+)")
TOK = {}


def tokenizer(repo):
    if "qwen" not in repo.lower():
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import common
        return common.decode
    if "q" not in TOK:
        from transformers import AutoTokenizer
        TOK["q"] = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    t = TOK["q"]
    return lambda ids: t.decode(list(ids), skip_special_tokens=False)


def equations_ok(text):
    """Every '<arithmetic> = <number>' in the completion evaluates to the number."""
    t = V.truncate(text)
    for m in EQ.finditer(t):
        lhs, rhs = m.group(1).strip(), m.group(2)
        if not re.search(r"\d\s*[-+*/x×]\s*\(?\s*\$?\d", lhs):
            continue                      # not an arithmetic expression
        expr = lhs.replace(",", "").replace("$", "").replace("×", "*").replace("x", "*")
        expr = re.sub(r"^[^\d(]*", "", expr)
        try:
            val = arith(expr)
            want = float(rhs.replace(",", ""))
        except Exception:
            continue
        if abs(val - want) > 1e-6 * max(1.0, abs(want)):
            return False
    return True


def names(task):
    if task == "mbpp":
        return {"SAMPLES": "bo8-filter", "BRANCH": "margin-branch-8-filter", "RANDOM": "random-branch-8-filter"}
    return {"SAMPLES": "sc-8", "BRANCH": "margin-branch-8", "RANDOM": "random-branch-8"}


def candidates(r, task, which):
    """List of (method, path index) in the declared order."""
    n = names(task)
    if which == "portfolio-8":
        return [(n["BRANCH"], k) for k in range(4)] + [(n["SAMPLES"], k) for k in range(4)]
    out = []
    for o in ORDER:
        m = n.get(o, o)
        out += [(m, k) for k in range(len(r["methods"][m]["paths"]))]
    return out


def ratchet_steps(task):
    n = names(task)
    add = [[("prlm-0", 0)], [(n["BRANCH"], k) for k in range(1, 8)], [(n["SAMPLES"], k) for k in range(8)],
           [("greedy", 0), ("sample", 0)], [(n["RANDOM"], k) for k in range(1, 8)]]
    steps, cur = [], []
    for a in add:
        cur = cur + a
        steps.append(list(cur))
    return steps


def ratchet(R, idx, task, tests, dec, base):
    is_code = task == "mbpp"
    n = names(task)
    steps = ratchet_steps(task)
    out, vis_by_step, P_by_step = [], [], []
    for s_i, cands in enumerate(steps):
        P, V_ = [], []
        for i in idx:
            r = R[i]
            if is_code:
                vis = [mbpp_visible(r, tests, m, k) for m, k in cands]
                sel = next((j for j, v in enumerate(vis) if v), None)
                m, k = cands[sel] if sel is not None else ("prlm-0", 0)
                P.append(r["methods"][m]["full"][k]); V_.append(sel is not None)
            else:
                ans = [r["methods"][m]["answers"][k] for m, k in cands]
                ok = [a is not None and equations_ok(dec(r["methods"][m]["paths"][k])) for (m, k), a in zip(cands, ans)]
                pool = [a for a, o in zip(ans, ok) if o]
                pred = V.vote(pool) if pool else r["methods"]["prlm-0"]["answers"][0]
                P.append(V.same(pred, r["gold"])); V_.append(bool(pool))
        P_by_step.append(P); vis_by_step.append(V_)
        meths = {}
        for m, k in cands:
            meths.setdefault(m, 0)
        tok = sum(sum(R[i]["methods"][m]["gen_tokens"] for m in meths) for i in idx) / len(idx)
        row = {"step": f"R{s_i}", "paths": len(cands), "new_paths": len(cands) - (len(steps[s_i - 1]) if s_i else 0),
               "accuracy": round(sum(P) / len(idx), 4), "checker_pass": round(sum(V_) / len(idx), 4),
               "mean_gen_tokens_upper": round(tok, 1),
               "vs": {m: V.verdict(P, base[m]) for m in base}}
        for m in base:
            row["vs"][m]["bar_3pt"] = row["vs"][m]["verdict"] == "WIN" and row["vs"][m]["diff"] >= 0.03
        out.append(row)
    viol = sum(1 for j in range(len(idx)) for a in range(len(steps)) for b in range(a + 1, len(steps))
               if vis_by_step[a][j] and not vis_by_step[b][j]) if is_code else None
    nonmono = sum(1 for j in range(len(idx)) if any(P_by_step[a][j] and not P_by_step[a + 1][j] for a in range(len(steps) - 1)))
    return {"steps": out, "Q3_checker_violations": viol, "accuracy_non_monotone_problems": nonmono}


_VIS = {}


def mbpp_visible(r, tests, m, k):
    v = r["methods"][m].get("visible")
    if v is not None:
        return v[k]
    key = (r["inputs"]["repo"], r["index"], m, k)   # each program is executed once
    if key not in _VIS:
        import value_code as C
        t = tests[r["index"]]
        _VIS[key] = C.runs(r["methods"][m]["programs"][k], t["test_imports"], t["test_list"][:1])
    return _VIS[key]


def select(r, task, which, tests, dec):
    cands = candidates(r, task, which)
    if task == "mbpp":
        vis = [mbpp_visible(r, tests, m, k) for m, k in cands]
        sel = next((i for i, v in enumerate(vis) if v), None)
        m, k = cands[sel] if sel is not None else ("prlm-0", 0)
        return {"pass": r["methods"][m]["full"][k], "visible": sel is not None,
                "oracle": any(r["methods"][m2]["full"][k2] for m2, k2 in cands)}, vis
    ans = [r["methods"][m]["answers"][k] for m, k in cands]
    ok = [a is not None and equations_ok(dec(r["methods"][m]["paths"][k])) for (m, k), a in zip(cands, ans)]
    pool = [a for a, o in zip(ans, ok) if o]
    pred = V.vote(pool) if pool else V.vote(ans)
    return {"pass": V.same(pred, r["gold"]), "checker_passed": sum(ok),
            "oracle": any(V.same(a, r["gold"]) for a in ans)}, None


def main(outfile, dirs):
    rows = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
            if "again" in os.path.abspath(p):
                continue
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    rows.setdefault((r.get("task", "gsm8k"), r["inputs"]["repo"]), {})[r["index"]] = r
    tests = None
    out = {}
    for (task, repo), R in sorted(rows.items()):
        idx = sorted(R)
        is_code = task == "mbpp"
        if is_code and tests is None:
            import value_code as C
            tests = {p["index"]: p for p in C.mbpp_load(os.environ.get("VALUE_MBPP_DATA"))}
        dec = tokenizer(repo)
        key = "pass" if is_code else "correct"
        n = names(task)
        base = {m: [R[i]["methods"][m][key] for i in idx] for m in ("greedy", n["SAMPLES"], n["BRANCH"])}
        res = {"task": task, "problems": len(idx)}
        for which in ("portfolio-27", "portfolio-8"):
            sel = [select(R[i], "mbpp" if is_code else task, which, tests, dec) for i in idx]
            P = [s[0]["pass"] for s in sel]
            row = {"accuracy": round(sum(P) / len(idx), 4),
                   "oracle": round(sum(s[0]["oracle"] for s in sel) / len(idx), 4),
                   "vs": {m: V.verdict(P, base[m]) for m in base}}
            for m in base:
                row["vs"][m]["bar_3pt"] = (row["vs"][m]["verdict"] == "WIN" and row["vs"][m]["diff"] >= 0.03)
            if is_code and which == "portfolio-27":
                # P1 check: a baseline's selection passes VISIBLE but the portfolio's does not
                viol = 0
                for i, s in zip(idx, sel):
                    base_vis = any(mbpp_visible(R[i], tests, m, R[i]["methods"][m]["selected"]) for m in
                                   ("greedy", "prlm-0", "sample", n["SAMPLES"], n["BRANCH"], n["RANDOM"]))
                    viol += int(base_vis and not s[0]["visible"])
                row["P1_violations"] = viol
            meths = {m for m, _ in candidates(R[idx[0]], "mbpp" if is_code else task, which)}
            row["paths"] = len(candidates(R[idx[0]], "mbpp" if is_code else task, which))
            row["mean_gen_tokens_upper"] = round(sum(sum(R[i]["methods"][m]["gen_tokens"] for m in meths) for i in idx) / len(idx), 1)
            res[which] = row
        res["baselines_accuracy"] = {m: round(sum(v) / len(idx), 4) for m, v in base.items()}
        res["ratchet"] = ratchet(R, idx, "mbpp" if is_code else task, tests, dec, base)
        out[f"{task}:{repo}"] = res
    json.dump(out, open(outfile, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
