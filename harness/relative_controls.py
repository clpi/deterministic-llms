"""Exhaustive integer tie and artificial drift controls; not new model runs."""
import hashlib
import itertools
import sys
from pathlib import Path

from relative_replay import evaluate


def main(fixture):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tests"))
    from robust_replay import PIN, parse
    data = Path(fixture).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PIN
    def ordered(scores):
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[0]
    values = list(itertools.product(range(-2,3),repeat=3))
    counts = [0,0,0]
    changed = 0
    for base in values:
        for d in values:
            variant = [x+y for x,y in zip(base,d)]
            r = evaluate(base,variant)
            winner, other = ordered(base),ordered(variant)
            flags = [r[k] for k in ("symmetric_row","interval_row","relative_row")]
            assert (winner,other) == (r["reference"],r["actual"])
            assert flags[2] == (winner==other)
            counts = [i+j for i,j in zip(counts,flags)]
            changed += winner!=other
    assert counts == [925,3445,10210] and changed == 5415
    print(f"exact three-token control: {len(values)**2} pairs; symmetric/interval/relative {counts}; winner changes {changed}")
    allowances = []
    for _,_,base,theta,edit,features,picks in parse(data):
        for weights,expected in zip([theta,[a+b for a,b in zip(theta,edit)]],picks):
            scores = list(base)
            for t,a,k in features:
                scores[t] += weights[a]*k
            winner = ordered(scores)
            assert winner==expected
            allowance,corner = min((scores[winner]-s-(i<winner),i)
                                   for i,s in enumerate(scores) if i!=winner)
            shifted = [s+1000 for s in scores]
            x = evaluate(scores,shifted)
            assert x["interval_row"] and not x["symmetric_row"]
            shifted[corner] += allowance
            assert evaluate(scores,shifted)["relative_row"]
            shifted[corner] += 1
            assert not evaluate(scores,shifted)["relative_row"]
            allowances.append(allowance)
    assert len(allowances)==6
    print(f"6 sealed GPT-2 score rows: simulated shift, exact boundary, and minimal adverse corner pass; allowances {allowances}")


if __name__ == "__main__":
    main(sys.argv[1])
