"""Test the actual old/new Lean executable behavior, not just Python parsing."""
import sys
import argparse,hashlib,json,struct,subprocess,tempfile
from pathlib import Path
from robust_replay import PIN,verify

p=argparse.ArgumentParser()
p.add_argument("checker",type=Path)
p.add_argument("fixture",type=Path)
p.add_argument("--baseline",type=Path)
p.add_argument("--output",type=Path)
a=p.parse_args()
data=a.fixture.read_bytes()
assert hashlib.sha256(data).hexdigest()==PIN
rows=verify(data,count=3)
vocab,axes=struct.unpack_from("<II",data,0)
feature=8+4*vocab+8*axes+4
def edit(p,v):return data[:p]+struct.pack("<i",v)+data[p+4:]
cases={"recorded_pass":(data,True),
       "empty":(b"",False),
       "missing_whole_record":(data[:rows[1]["end"]],False),
       "partial_header":(data[:5],False),
       "missing_last_byte":(data[:-1],False),
       "missing_picks":(data[:rows[0]["end"]-8],False),
       "negative_token":(edit(feature,-1),False),
       "wide_token":(edit(feature,vocab),False),
       "negative_axis":(edit(feature+4,-1),False),
       "wide_axis":(edit(feature+4,axes),False),
       "bad_pick":(edit(rows[0]["end"]-8,(rows[0]["base"]["winner"]+1)%vocab),False),
       "zero_vocabulary":(edit(0,0),False),
       "out_of_range_pick":(edit(rows[0]["end"]-8,vocab),False),
       **{f"trailing_{i}":(data+b"\xff"*i,False) for i in range(1,8)}}
def check(exe,case,expected=True):
    return subprocess.run([str(exe.resolve()),"decide",str(case)]+(["--expect","3"] if expected else []),
                          capture_output=True,timeout=10)
result=[]
with tempfile.TemporaryDirectory() as d:
    q=Path(d)/"input.bin"
    for label,(case,valid) in cases.items():
        q.write_bytes(case)
        r=check(a.checker,q)
        assert (r.returncode==0)==valid,(label,r.returncode,r.stderr)
        row={"case":label,"candidate_exit":r.returncode}
        if a.baseline and (valid or label in {"empty","missing_whole_record","partial_header","bad_pick"} or label.startswith("trailing_")):
            x=check(a.baseline,q,False)
            row["baseline_exit"]=x.returncode
        result.append(row)
radii = subprocess.run([str(a.checker.resolve()), "decide", str(a.fixture.resolve()), "--robust"],
                       capture_output=True, check=True, text=True, timeout=10)
found = [line for line in radii.stdout.splitlines() if line.startswith("radius ")]
expected = [f"radius {i+1} {row['base']['radius']} {row['retuned']['radius']}" for i,row in enumerate(rows)]
assert found == expected, (found, expected)
if a.output:
    a.output.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "harness" / "relative_controls.py"),
                str(a.fixture.resolve())], check=True, timeout=30, stdout=sys.stderr)
