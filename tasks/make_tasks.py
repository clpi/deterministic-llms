"""Builds tasks/tasks.json deterministically. Written and hashed into the log
BEFORE any axis code exists (contract rule: never add a task in the same
change as the feature that solves it).

T1 continuation: prose from Python standard-library module docstrings
   (this container's Python). Modules sorted by name; even positions are
   TRAIN (for the n-gram automaton and the distilled rule table), odd
   positions are TEST. Prompt = first 32 tokens, reference = next 24.
T2 closed cloze: fixed list with one expected next word each.
T3 control: per TEST prompt, a declared target (ban list, topic lexicon,
   or sentence end within N tokens). Lexicons have a TRAIN half (allowed
   to build axes) and a TEST half (only used to score).
"""
import importlib, json, os, pkgutil, re, sys, sysconfig, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
from common import encode, decode

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.json")

SKIP = {"antigravity", "this", "idlelib", "turtledemo", "tkinter", "test", "lib2to3",
        "ensurepip", "venv", "pydoc_data", "__main__", "__phello__", "_pyrepl"}


def prose(doc):
    doc = re.sub(r"\s+", " ", doc).strip()
    return doc


def collect():
    names = sorted(m.name for m in pkgutil.iter_modules(None)
                   if not m.name.startswith("_") and m.name not in SKIP
                   and m.module_finder.path.startswith(sysconfig.get_paths()["stdlib"]))
    out = []
    for n in names:
        try:
            mod = importlib.import_module(n)
        except Exception:
            continue
        d = getattr(mod, "__doc__", None)
        if not d:
            continue
        p = prose(d)
        ids = encode(p)
        if len(ids) >= 56 and sum(c.isalpha() for c in p[:300]) > 200:
            out.append((n, ids))
    return out


def main():
    docs = collect()
    train = [d for i, d in enumerate(docs) if i % 2 == 0]
    test = [d for i, d in enumerate(docs) if i % 2 == 1][:40]
    t1 = [{"id": f"t1/{n}", "module": n, "prompt": ids[:32], "reference": ids[32:56]}
          for n, ids in test]
    corpus = [{"module": n, "ids": ids} for n, ids in train]

    cloze = [
        ("The capital of France is", "Paris"), ("The capital of Germany is", "Berlin"),
        ("The capital of Italy is", "Rome"), ("The capital of Japan is", "Tokyo"),
        ("The capital of Spain is", "Madrid"), ("The capital of Russia is", "Moscow"),
        ("The capital of England is", "London"), ("The capital of China is", "Beijing"),
        ("The capital of Egypt is", "Cairo"), ("The capital of Canada is", "Ottawa"),
        ("1, 2, 3, 4, 5,", "6"), ("2, 4, 6, 8,", "10"), ("10, 20, 30, 40,", "50"),
        ("Monday, Tuesday, Wednesday,", "Thursday"), ("January, February, March,", "April"),
        ("A, B, C, D, E,", "F"), ("one, two, three, four,", "five"),
        ("The opposite of hot is", "cold"), ("The opposite of up is", "down"),
        ("The color of the sky is", "blue"), ("Water freezes at zero degrees", "Celsius"),
        ("The first president of the United States was George", "Washington"),
        ("Romeo and Juliet was written by William", "Shakespeare"),
        ("The largest planet in the solar system is", "Jupiter"),
        ("The chemical symbol for gold is", "Au"), ("Barack Obama was born in", "Hawaii"),
        ("The Eiffel Tower is located in", "Paris"), ("The sun rises in the", "east"),
        ("A week has seven", "days"), ("Cats and", "dogs"),
        ("import numpy as", "np"), ("def __init__(self", ","),
        ("for i in range(", "10"), ("Thank you very", "much"),
        ("Once upon a", "time"), ("Happy birthday to", "you"),
        ("The quick brown fox jumps over the lazy", "dog"),
        ("Rock and", "roll"), ("Salt and", "pepper"), ("Ladies and", "gentlemen"),
    ]
    t2 = [{"id": f"t2/{i}", "prompt": encode(p), "answer": a, "text": p} for i, (p, a) in enumerate(cloze)]

    lex = {
        "music": (["music", "song", "songs", "guitar", "piano", "band"],
                  ["melody", "singer", "album", "concert", "drum", "orchestra"]),
        "food": (["food", "eat", "dinner", "bread", "cheese", "cook"],
                 ["pizza", "soup", "meal", "kitchen", "fruit", "breakfast"]),
        "weather": (["weather", "rain", "snow", "storm", "wind", "sunny"],
                    ["cloud", "clouds", "temperature", "forecast", "thunder", "cold"]),
        "sports": (["football", "soccer", "team", "game", "player", "score"],
                   ["basketball", "coach", "match", "league", "tennis", "goal"]),
    }
    ban = ["the", "a", "an", "is", "of", "and", "to", "in"]
    t3 = []
    topics = sorted(lex)
    for i, item in enumerate(t1[:20]):
        t3.append({"id": f"t3/ban/{i}", "kind": "ban", "prompt": item["prompt"], "words": ban, "steps": 24})
        tp = topics[i % 4]
        t3.append({"id": f"t3/topic/{i}", "kind": "topic", "prompt": item["prompt"], "topic": tp,
                   "train_words": lex[tp][0], "test_words": lex[tp][1], "steps": 24})
        t3.append({"id": f"t3/end/{i}", "kind": "end", "prompt": item["prompt"], "within": 12, "steps": 24})

    doc = {"python": sys.version.split()[0], "t1": t1, "t2": t2, "t3": t3,
           "lexicons": lex, "corpus": corpus,
           "counts": {"t1": len(t1), "t2": len(t2), "t3": len(t3), "corpus_docs": len(corpus),
                      "corpus_tokens": sum(len(c["ids"]) for c in corpus)}}
    with open(OUT, "w") as f:
        json.dump(doc, f, sort_keys=True)
    print(json.dumps(doc["counts"]))


if __name__ == "__main__":
    main()
