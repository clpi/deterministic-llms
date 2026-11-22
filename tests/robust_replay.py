"""Independent exact GPT-2 replay auditor: strict wire format and sharp L∞ radii."""
import argparse, hashlib, json, struct
from pathlib import Path

PIN = "97f7d942c46bb93fa34965e2112eb7a9076ea5cfc571810a8371c13e4e4ea0c2"

def parse(data):
    end = len(data)
    o = 0
    while o < end:
        if end-o < 8:
            raise ValueError("trailing partial header")
        origin = o
        n,k = struct.unpack_from("<II", data, o)
        start = o+8
        nf_at = start+4*n+8*k
        if n == 0 or nf_at+4 > end:
            raise ValueError("invalid dimensions")
        nf, = struct.unpack_from("<I", data, nf_at)
        picks_at = nf_at+4+12*nf
        if picks_at+8 > end:
            raise ValueError("truncated body")
        a = struct.unpack_from(f"<{n+2*k}i", data, start)
        base, theta, delta = a[:n], a[n:n+k], a[n+k:]
        entries = list(struct.iter_unpack("<iii", data[nf_at+4:picks_at]))
        if any(not (0 <= tok < n and 0 <= axis < k) for tok,axis,val in entries):
            raise ValueError("feature index out of range")
        expected = struct.unpack_from("<ii",data,picks_at)
        if any(not 0 <= p < n for p in expected):
            raise ValueError("bad expected pick")
        o = picks_at+8
        yield origin, o, base, theta, delta, entries, expected
    if not o:
        raise ValueError("empty replay")

def pick(scores):
    # Deliberately use an independent left-to-right selection, not sorting.
    winner = 0
    for i in range(1,len(scores)):
        if scores[i] > scores[winner]:
            winner = i
    return winner

def robust(scores, w):
    assert w == pick(scores)
    if len(scores)==1:
        return {"winner":w,"radius":None,"witness":None}
    radius,j = min(((scores[w]-v-(i<w))//2,i)
                   for i,v in enumerate(scores) if i!=w)
    assert radius>=0
    # This adversarial mutation exercises the *first* radius which can fail.
    mutations = list(scores)
    mutations[w] -= radius+1
    mutations[j] += radius+1
    assert pick(mutations) != w
    runner = max(v for i,v in enumerate(scores) if i!=w)
    return {"winner":w,"winner_score":scores[w],"plain_margin":scores[w]-runner,
            "radius":radius,"failed_at":radius+1,"witness":j,
            "witness_relates":("lower_token_id" if j<w else "higher_token_id"),
            "winner_at_failed_radius":pick(mutations)}

def verify(data, count=None):
    findings=[]
    for o,end,base,theta,delta,ent,expect in parse(data):
        ordinary=list(base)
        retuned=list(base)
        for tok,axis,val in ent:
            ordinary[tok]+=theta[axis]*val
            retuned[tok]+=(theta[axis]+delta[axis])*val
        p0,p1=pick(ordinary),pick(retuned)
        if (p0,p1)!=expect:
            raise ValueError("decisions disagree")
        findings.append({"offset":o,"end":end,"n_vocab":len(base),"n_axes":len(theta),
                         "feature_rows":len(ent),"base":robust(ordinary,p0),
                         "retuned":robust(retuned,p1)})
    if count is not None and len(findings)!=count:
        raise ValueError("record count differs")
    return findings

def controls(data, rows):
    s0=rows[0]; s1=rows[1]
    assert s0["feature_rows"]>0
    n,k=struct.unpack_from("<II",data,0)
    first_feature=8+4*n+8*k+4
    def set_i32(at,val):
        return data[:at]+struct.pack("<i",val)+data[at+4:]
    tails={f"trailing_{i}":data+b"\xff"*i for i in range(1,8)}
    cases={"empty":b"",
           "partial_header":data[:5],
           "last_byte_missing":data[:-1],
           "whole_last_record_missing":data[:s1["end"]],
           "missing_expected_picks":data[:s0["end"]-8],
           "negative_feature_token":set_i32(first_feature,-1),
           "too_large_feature_token":set_i32(first_feature,n),
           "negative_feature_axis":set_i32(first_feature+4,-1),
           "too_large_feature_axis":set_i32(first_feature+4,k),
           "wrong_pick":set_i32(s0["end"]-8,(s0["base"]["winner"]+1)%n),
           "zero_vocabulary":set_i32(0,0),
           "out_of_range_expected_token":set_i32(s0["end"]-8,n),
           **tails}
    out={}
    for name,value in cases.items():
        try:
            verify(value,count=3)
        except ValueError as e:
            out[name]=str(e)
        else:
            raise AssertionError(f"accepted {name}")
    changed=data[:s0["end"]] + data[s0["end"]:]
    assert changed==data
    # Structurally valid logs are not authenticated by their own picks.
    assert len(verify(data[:s1["end"]]))==2
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("fixture",type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    data=args.fixture.read_bytes()
    sha=hashlib.sha256(data).hexdigest()
    if sha!=PIN:
        raise SystemExit("fixture hash mismatch")
    rows=verify(data,count=3)
    result={"fixture_sha256":sha,"records":rows,"malformed_controls":controls(data,rows)}
    result["global_base_radius"]=min(v["base"]["radius"] for v in rows)
    result["global_retuned_radius"]=min(v["retuned"]["radius"] for v in rows)
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    main()
