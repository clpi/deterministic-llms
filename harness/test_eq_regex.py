"""The linear EQ regex of value_portfolio.py finds exactly what the old one found.

  python test_eq_regex.py
Random strings over the characters either regex can see, plus the long digit run that hung
the old form (timed for the new one only).
"""
import random, re, time
import value_portfolio as P

OLD = re.compile(P.EQ_OLD)
ALPHA = "0123456789.,$ -+*/x×()=a\n"
rng = random.Random(0)
for _ in range(200000):
    s = "".join(rng.choice(ALPHA) for _ in range(rng.randint(0, 14)))
    a = [(m.span(), m.groups()) for m in OLD.finditer(s)]
    b = [(m.span(), m.groups()) for m in P.EQ.finditer(s)]
    assert a == b, (s, a, b)
t = time.time()
list(P.EQ.finditer("1" * 4000 + "a 2 + 3 = 5"))
assert time.time() - t < 2, "new form is slow on a long digit run"
print("EQ: 200000 random strings agree; 4000-digit run in", round(time.time() - t, 3), "s")
