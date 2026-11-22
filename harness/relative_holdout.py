"""Retrospective threshold selection on prompts 0–19, evaluation on 20–39."""
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())["observations"]
for name, prompts in data.items():
    assert [p["prompt_index"] for p in prompts] == list(range(40))
    train, hold = ([r for p in g for r in p["rows"]] for g in (prompts[:20], prompts[20:]))
    lower = min(r["minimum_drift"] for r in train)
    upper = max(r["maximum_drift"] for r in train)
    eligible = lambda r: upper-lower <= r["minimum_allowance"]
    outside = lambda r: r["minimum_drift"] < lower or r["maximum_drift"] > upper
    eligible_rows = [r for r in hold if eligible(r)]
    print(name, json.dumps({
        "calibration_interval": [lower,upper],
        "heldout_rows": len(hold),
        "heldout_rows_outside_observed_calibration_interval": sum(map(outside,hold)),
        "heldout_rows_eligible_under_assumed_interval": len(eligible_rows),
        "eligible_rows_outside_interval": sum(map(outside,eligible_rows)),
        "eligible_rows_with_wrong_pick": sum(not r["relative_row"] for r in eligible_rows),
        "eligible_replies_under_assumed_interval": sum(all(eligible(r) for r in p["rows"]) for p in prompts[20:]),
    },sort_keys=True))
