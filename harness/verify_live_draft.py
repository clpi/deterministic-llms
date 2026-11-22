"""Check recorded evidence and price target savings against live drafting."""
import json
import statistics
import sys
from pathlib import Path

def main(filename):
    d = json.loads(Path(filename).read_text())
    assert d["meta"]["horizon"]==24 and d["tamper_refused"]
    rows = d["rows"]
    assert [r["prompt"] for r in rows] == list(range(40))
    for row in rows:
        a,b = row["serial"],row["verified"]
        n = len(row["prefix"])
        assert n and a["calls"] == 24 and a["positions"] == 24*n+276
        assert b["draft_calls"] == 24 and b["draft_positions"] == n+23 and b["ns"] >= b["draft_ns"] > 0
        assert b["calls"] == 1 + 24*b["fallback"]
        assert b["positions"] == n+23 + b["fallback"]*a["positions"]
        for t in (a,b):
            assert len(t["trace"])==len(t["out"])==24
            assert t["out"] == [x[0] for x in t["trace"]]
            assert all(x[2]>=0 for x in t["trace"])
        assert row["reply_equal"] == (a["out"] == b["out"])
        assert row["certificate_equal"] == (a["trace"] == b["trace"])
    a,c = [[r[m] for r in rows] for m in ("serial","verified")]
    def ticks(x):
        return sum(r["ns"] for r in x)/1e9
    def trace(x):
        return {p:sum(r.get(p,0) for r in x) for p in ("calls","positions","draft_calls","draft_positions")}
    summary=dict(reply_equal=sum(r["reply_equal"] for r in rows),
       certificate_equal=sum(r["certificate_equal"] for r in rows),
       fallback=sum(r["fallback"] for r in c), target_serial=trace(a), target_verified=trace(c),
       serial_wall_s=ticks(a), verified_wall_s=ticks(c),
       serial_median_ms=statistics.median(r["ns"] for r in a)/1e6,
       verified_median_ms=statistics.median(r["ns"] for r in c)/1e6,
       paired_verified_faster=sum(x["ns"]>y["ns"] for x,y in zip(a,c)),
       draft_ns_total=sum(r["draft_ns"] for r in c), controls=1)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main(sys.argv[1])
