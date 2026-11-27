"""Control for the regeneration of VALUE_AMEND_5: every non-branch path of a regenerated
row must equal the stored path cut just after its first end-of-document token (same
prefix, same logits, the only change is the stop). Branch methods are counted the same
way for information; they may differ wherever the old run branched past the end of
document or the answer end moved.

  python value_eos_compare.py OUTFILE NEWDIR OLDDIR
"""
import glob, json, os, sys

EOS = (151643, 151645)
PLAIN = ("greedy", "prlm-0", "sample", "sc-8", "bo8-filter")


def load(d):
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
        if "again" in os.path.abspath(p):
            continue
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                rows[(r.get("task", "gsm8k"), r["inputs"]["repo"], r["index"])] = r
    return rows


def cut(p):
    i = next((k for k, x in enumerate(p) if x in EOS), None)
    return p if i is None else p[:i + 1]


def main(outfile, new, old):
    N, O = load(new), load(old)
    out = {}
    for key, r in sorted(N.items()):
        k = f"{key[0]}:{key[1]}"
        s = out.setdefault(k, {"rows": 0, "rows_with_old": 0, "methods": {}})
        s["rows"] += 1
        o = O.get(key)
        if o is None:
            continue
        s["rows_with_old"] += 1
        for m, M in r["methods"].items():
            if m not in o["methods"]:
                continue
            t = s["methods"].setdefault(m, {"paths": 0, "equal_to_cut": 0})
            for a, b in zip(M["paths"], o["methods"][m]["paths"]):
                t["paths"] += 1
                t["equal_to_cut"] += int(a == cut(b))
    for k, s in out.items():
        s["control_holds"] = all(t["equal_to_cut"] == t["paths"] for m, t in s["methods"].items() if m in PLAIN) \
            if s["rows_with_old"] else None
    json.dump(out, open(outfile, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
