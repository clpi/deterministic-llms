"""Interim tally from finished value shards' job logs (VALUE_AMEND_2 'speed'). Not a verdict."""
import ast, json, re, subprocess, sys
REPO = "clpi/deterministic-llms"
def gh(path):
    return subprocess.run(["gh", "api", path], capture_output=True, text=True).stdout
tally = {}
for run in sys.argv[1:]:
    jobs = json.loads(gh(f"repos/{REPO}/actions/runs/{run}/jobs?per_page=100"))["jobs"]
    jobs += json.loads(gh(f"repos/{REPO}/actions/runs/{run}/jobs?per_page=100&page=2")).get("jobs", [])
    for j in jobs:
        if j["status"] != "completed" or not j["name"].startswith(("run", "mbpp", "gsm")):
            continue
        key = j["name"].split(",")[0].split("(")[-1] if j["name"].startswith("run") else j["name"].split(" ")[0]
        for line in gh(f"repos/{REPO}/actions/jobs/{j['id']}/logs").splitlines():
            m = re.match(r"^\S+ (\d+) (\{.*\})$", line.strip())
            if m:
                d = ast.literal_eval(m.group(2))
                tally.setdefault(key, {})[int(m.group(1))] = d
out = {}
for k, R in tally.items():
    ms = list(next(iter(R.values())))
    out[k] = {"problems": len(R), "correct": {m: sum(r[m] for r in R.values()) for m in ms}}
print(json.dumps(out, indent=1))
