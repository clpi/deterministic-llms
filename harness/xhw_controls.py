"""Controls and verdict rule of the cross-hardware audit (tasks/XHW_CONTRACT.json,
XHW_AMEND_1.json, XHW_AMEND_2.json), computed from the raw run files.

  python3 xhw_controls.py OUTFILE DIR...

do_nothing: each machine's fresh-process repeat of its reference config against
the first run of that config on the same machine: every per-step float hash and
every free generation must be equal. verdict_rule (b): per model and machine,
prlm-0's identical share on X4 against every other method's. Rows whose run
ended in an error, or whose job never wrote a file, are listed as UNMEASURED.
"""
import glob, json, os, sys

METHODS = ["prlm-0", "greedy-f32", "samp-seeded-t0.7", "samp-seeded-t1.0"]
MACHINES = ["ubuntu-24.04", "ubuntu-24.04-arm", "macos-latest", "windows-latest", "windows-11-arm"]
EXPECTED = {"GPT2": MACHINES + ["R"], "Q35": MACHINES, "Q3": MACHINES, "LFM2": MACHINES,
            "G4H": MACHINES, "MOE": MACHINES, "INK": ["ubuntu-24.04", "ubuntu-24.04-arm"]}
REF_CONFIG = {"GPT2": "t4-all"}
REF_MACHINE = {"GPT2": "R"}


def load(dirs):
    runs = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            try:
                r = json.load(open(p))
            except Exception:
                continue
            if "config" not in r or "machine" not in r:
                continue
            r["_model"] = r.get("key", "GPT2")
            r["_repeat"] = "again" in os.path.abspath(p)
            runs.append(r)
    return runs


def main(outfile, dirs):
    runs = load(dirs)
    out = {}
    for model, machines in EXPECTED.items():
        R = [r for r in runs if r["_model"] == model]
        rc = REF_CONFIG.get(model, "t4")
        rm = REF_MACHINE.get(model, "ubuntu-24.04")
        ref = next((r for r in R if r["machine"]["machine"] == rm and r["config"] == rc
                    and not r["_repeat"] and "teacher" in r), None)
        per = {}
        for m in machines:
            mine = [r for r in R if r["machine"]["machine"] == m]
            errs = [r["error"][:300] for r in mine if "error" in r]
            first = next((r for r in mine if r["config"] == rc and not r["_repeat"] and "teacher" in r), None)
            again = next((r for r in mine if r["config"] == rc and r["_repeat"] and "teacher" in r), None)
            row = {"configs_done": sorted({(r["config"], r["_repeat"]) for r in mine if "teacher" in r})}
            if errs:
                row["errors"] = errs
            if not first:
                row["status"] = "UNMEASURED"
                per[m] = row
                continue
            if again:
                row["do_nothing_float_steps_equal"] = sum(a["f"] == b["f"] for a, b in zip(first["teacher"], again["teacher"]))
                row["do_nothing_free_equal"] = {k: sum(x == y for x, y in zip(first["free"][k], again["free"][k])) for k in METHODS}
                row["do_nothing_holds"] = (row["do_nothing_float_steps_equal"] == len(first["teacher"])
                                           and all(v == len(first["free"][k]) for k, v in row["do_nothing_free_equal"].items()))
            else:
                row["do_nothing_holds"] = None   # repeat not run (job limit) -> UNMEASURED
            if ref and m != rm:
                fi = {k: sum(x == y for x, y in zip(first["free"][k], ref["free"][k])) for k in METHODS}
                row["free_identical_ref_config"] = fi
                row["prlm0_beaten_by"] = [k for k in METHODS[1:] if fi[k] > fi["prlm-0"]]
                # every teacher-forced step whose certified pick differs from the reference:
                # step, reference pick and margin, this machine's pick and margin, and score diff
                row["pick_flips"] = [[k, b["pick"], b["margin"], a["pick"], a["margin"], a.get("maxdiff")]
                                     for k, (a, b) in enumerate(zip(first["teacher"], ref["teacher"])) if a["pick"] != b["pick"]][:40]
            per[m] = row
        measured = [m for m in machines if per[m].get("status") != "UNMEASURED"]
        out[model] = {"reference": f"{rm} {rc}", "reference_found": ref is not None,
                      "unmeasured_machines": [m for m in machines if m not in measured],
                      "do_nothing_all_hold": all(per[m].get("do_nothing_holds") for m in measured
                                                 if per[m].get("do_nothing_holds") is not None),
                      "do_nothing_unmeasured": [m for m in measured if per[m].get("do_nothing_holds") is None],
                      "prlm0_losses": {m: per[m]["prlm0_beaten_by"] for m in measured if per[m].get("prlm0_beaten_by")},
                      "machines": per}
    json.dump(out, open(outfile, "w"), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("unmeasured_machines", "do_nothing_all_hold", "do_nothing_unmeasured", "prlm0_losses")}
                      for k, v in out.items()}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
