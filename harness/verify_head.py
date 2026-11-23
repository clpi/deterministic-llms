"""Recount observer-specific qualification and measured fronts."""
import itertools
import json
import statistics
import sys
from pathlib import Path


def summarize(p):
    m = p["scope"]
    r = p["rows"]
    assert len(r) == 40 and sorted(x["prompt"] for x in r) == list(range(40))
    assert m["steps"] == 24 and p["tamper_refused"]
    modes = list(r[0]["arms"])
    assert len(modes)==5
    for a in modes:
        assert sorted(x["order"].index(a) for x in r)==sorted(list(range(5))*8)
    totals={}
    for row in r:
        n=len(row["prefix"])
        x=row["arms"]
        assert set(x)==set(modes) and len(set(row["order"]))==5
        assert x["serial_full"]["ort_head_rows"]==x["serial_full"]["ort_positions"]==24*n+276
        assert x["serial_last"]["ort_positions"]==24*n+276 and x["serial_last"]["ort_head_rows"]==24
        assert x["kv_last"]["kv_positions"]==n+23 and x["kv_last"]["kv_head_rows"]==24
        assert x["verify_full"]["kv_head_rows"]==n+23 and x["verify_last"]["kv_head_rows"]==24
        for k,y in x.items():
            assert len(y["trace"])==24 and y["out"]==[z[0] for z in y["trace"]]
            if k.startswith("verify"):
                assert y["ort_positions"]==n+23+y["fallback"]*(24*n+276)
                assert y["ort_calls"]==1+y["fallback"]*24
    for name in sorted(modes):
        a=[x["arms"][name] for x in r]
        oracle=[x["arms"]["serial_full"] for x in r]
        v={k:sum(x[k] for x in a) for k in ("ns","ort_calls","kv_calls","ort_positions","kv_positions",
                                                   "ort_head_rows","kv_head_rows","fallback")}
        v.update(reply_equal=sum(x["out"]==y["out"] for x,y in zip(a,oracle)),
                 decision_receipt_equal=sum(x["trace"]==y["trace"] for x,y in zip(a,oracle)),
                 exact_steps=sum(sum(u==v for u,v in zip(x["trace"],y["trace"])) for x,y in zip(a,oracle)),
                 median_ns=statistics.median(x["ns"] for x in a))
        totals[name]=v
    def objectives(x):
        return x["ns"],x["ort_positions"]+x["kv_positions"],x["ort_head_rows"]+x["kv_head_rows"],x["ort_calls"]+x["kv_calls"]
    def dom(a,b):
        a,b=map(objectives,(a,b))
        return a!=b and all(x<=y for x,y in zip(a,b))
    front={}
    for obs in ("reply_equal","decision_receipt_equal"):
        q={k:v for k,v in totals.items() if v[obs]==40}
        front[obs]=[k for k,v in q.items() if not any(dom(w,v) for w in q.values())]
    return {"arms":totals,"finite_measured_front":front}

if __name__=="__main__":
    print(json.dumps(summarize(json.loads(Path(sys.argv[1]).read_text())),indent=2))
