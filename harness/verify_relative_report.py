"""Recount a report, without pretending the summary proves its model inputs."""
import json
import sys
from pathlib import Path


def verify(path, manifest=None):
    report = json.loads(Path(path).read_text())
    if manifest is not None:
        assert report["run"]["assets"] == json.loads(Path(manifest).read_text())
    assert report["run"]["skipped_forwards"] == 0
    for name, data in report["observations"].items():
        assert [p["prompt_index"] for p in data] == list(range(40))
        assert all(len(p["rows"]) == len(p["reference_tokens"]) == 24 for p in data)
        rows = [r for p in data for r in p["rows"]]
        lo, hi = min(r["minimum_drift"] for r in rows), max(r["maximum_drift"] for r in rows)
        recounted = report["metrics"][name]
        assert (lo,hi) == (recounted["corpus_minimum_drift"],recounted["corpus_maximum_drift"])
        for row in rows:
            low, high, m = row["minimum_drift"], row["maximum_drift"], row["minimum_allowance"]
            assert lo <= low <= high <= hi and m >= 0
            assert row["relative_row"] == (row["actual"] == row["reference"])
            assert row["symmetric_row"] == (2*max(abs(low),abs(high)) <= m)
            assert row["interval_row"] == (high-low <= m)
            assert row["symmetric_corpus"] == (2*max(abs(lo),abs(hi)) <= m)
            assert row["interval_corpus"] == (hi-lo <= m)
            assert not row["interval_row"] or row["relative_row"]
        for p in data:
            assert p["teacher_forced_equal"] == all(row["reference"] == row["actual"] for row in p["rows"])
            if name != "int8_batch_vs_int8_step":
                assert p["reference_tokens"] == [r["reference"] for r in p["rows"]]
            if p["free_tokens"] is not None:
                assert (p["reference_tokens"]==p["free_tokens"]) == p["free_equal"] == p["teacher_forced_equal"]
        assert recounted["teacher_forced_equal"] == sum(p["teacher_forced_equal"] for p in data)
        for kind in ("symmetric_row","interval_row","relative_row","symmetric_corpus","interval_corpus"):
            hits = sum(r[kind] for r in rows)
            full = sum(all(row[kind] for row in p["rows"]) for p in data)
            assert {"steps":hits,"complete_replies":full} == recounted["coverage"][kind]
        print(name, json.dumps(recounted, sort_keys=True))


if __name__ == "__main__":
    verify(*sys.argv[1:])
