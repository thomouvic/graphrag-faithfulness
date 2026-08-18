"""
Chain-structure analyzer for SPARQL-CoT outputs (mechanism probe).

Parses a model's Step-1 / Step-2 / Step-3 output into measurable structural
features, to test *why* in-domain few-shot closes the 8B-vs-70B gap rather
than just *whether* it does. Features:

  - n_triple_patterns : compositional depth of the Step-1 SPARQL. Compared
                        against the question's gold hop count and against the
                        70B's distribution.
  - trace_chars/words : length of the Step-2 trace (the "sweet spot" probe).
  - final_answer      : the Step-3 answer string.
  - literal_grounding : fraction of quoted literals in the SPARQL that appear
                        in the retrieved context (a cheap faithfulness proxy;
                        the richer binding-level check reuses
                        probe_sparql_verified once full chains exist).
  - parse_ok          : whether the Step-1 SPARQL block was found at all.

Dependency-free (stdlib only) so the self-test runs anywhere, including this
repo checkout where experiments/ and probe_results/ are absent.

Run the self-test:  python -u probe/probe_chain_structure.py --selftest
"""
import argparse
import json
import re
from pathlib import Path


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def extract_sparql_block(raw: str):
    """Return the text inside the first SELECT ... { ... } block, or None."""
    m = re.search(r"SELECT\b.*?\{(.*?)\}", raw, re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else None


def count_triple_patterns(sparql_body: str) -> int:
    """Count triple patterns: clauses terminated by ' .' with >=3 tokens.

    Robust to one-per-line or multiple-per-line formatting. Ignores empty
    lines and stray punctuation.
    """
    if not sparql_body:
        return 0
    # Split on '.' that terminate a triple (not inside quotes is good enough
    # for this prompt format, which uses no '.' inside literals).
    clauses = [c.strip() for c in sparql_body.split(".")]
    n = 0
    for c in clauses:
        if not c:
            continue
        toks = c.split()
        if len(toks) >= 3:  # subject predicate object(+)
            n += 1
    return n


def extract_literals(sparql_body: str):
    """Quoted literals used as objects/subjects in the SPARQL."""
    if not sparql_body:
        return []
    return re.findall(r'"([^"]+)"', sparql_body)


def extract_trace(raw: str) -> str:
    """Text of the Step-2 trace, between 'Step 2' and 'Step 3'/'FINAL ANSWER'."""
    m = re.search(r"Step\s*2\b(.*?)(?:Step\s*3\b|FINAL\s*ANSWER)",
                  raw, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_final_answer(raw: str):
    matches = re.findall(r"(?i)FINAL\s*ANSWER\s*:\s*(.+)", raw)
    if matches:
        return matches[-1].strip().split("\n")[0].strip()
    return None


def literal_grounding(literals, context: str):
    """Fraction of SPARQL literals that appear (normalized substring) in ctx."""
    if not literals:
        return None  # nothing to ground
    ctx = _normalize(context)
    hit = sum(1 for lit in literals if _normalize(lit) in ctx)
    return hit / len(literals)


def analyze_chain(raw: str, context: str = "", gold_hops: int = None) -> dict:
    body = extract_sparql_block(raw)
    literals = extract_literals(body) if body else []
    trace = extract_trace(raw)
    n_tp = count_triple_patterns(body)
    return {
        "parse_ok": body is not None,
        "n_triple_patterns": n_tp,
        "gold_hops": gold_hops,
        "depth_match": (None if gold_hops is None else int(n_tp == gold_hops)),
        "trace_chars": len(trace),
        "trace_words": len(trace.split()),
        "final_answer": extract_final_answer(raw),
        "n_literals": len(literals),
        "literal_grounding": literal_grounding(literals, context),
    }


# --------------------------------------------------------------------------
# Self-test fixtures: representative *model outputs* (the `raw` we'd log).
# Includes well-formed chains and the messy deviations the 8B actually emits.
# --------------------------------------------------------------------------
_WELLFORMED = """Step 1: Write a simple SPARQL query.
SELECT ?capital WHERE {
  "Toyota Motor Corporation" headquarteredIn ?city .
  ?city locatedIn ?country .
  ?country hasCapital ?capital .
}
Step 2: Trace.
Toyota headquarteredIn Toyota City, which is locatedIn Japan.
Country = Japan. Japan hasCapital Tokyo.
Step 3: FINAL ANSWER: Tokyo"""

_INLINE_TRIPLES = """Step 1: SELECT ?x WHERE { "Boston" foundedIn ?yb . "Philadelphia" foundedIn ?yp . }
Step 2: Boston 1630, Philadelphia 1682. Boston earlier.
Step 3: FINAL ANSWER: Boston"""

_MESSY_NO_STEPS = """The answer is Tokyo because Toyota is in Japan and Japan's capital is Tokyo.
FINAL ANSWER: Tokyo"""

_TRUNCATED_500 = _WELLFORMED[:500]  # simulate the raw[:500] we currently save

_CONTEXT_TOYOTA = ("Toyota Motor Corporation is headquartered in Toyota City "
                   "in Japan. The capital of Japan is Tokyo.")


def _selftest():
    cases = [
        ("wellformed (3-hop)", _WELLFORMED, _CONTEXT_TOYOTA, 3,
         dict(parse_ok=True, n_triple_patterns=3, final_answer="Tokyo")),
        ("inline triples (2-hop)", _INLINE_TRIPLES, "", 2,
         dict(parse_ok=True, n_triple_patterns=2, final_answer="Boston")),
        ("messy / no SPARQL", _MESSY_NO_STEPS, "", 1,
         dict(parse_ok=False, n_triple_patterns=0, final_answer="Tokyo")),
        ("truncated to 500ch", _TRUNCATED_500, _CONTEXT_TOYOTA, 3,
         dict(parse_ok=True, n_triple_patterns=3)),
    ]
    ok = True
    print(f"raw[:500] cutoff vs full wellformed chain: "
          f"{len(_WELLFORMED)} chars total -> "
          f"{'CLIPS' if len(_WELLFORMED) > 500 else 'FITS within'} 500\n")
    for name, raw, ctx, hops, expect in cases:
        got = analyze_chain(raw, ctx, gold_hops=hops)
        fails = [f"{k}: got {got.get(k)!r} != {v!r}"
                 for k, v in expect.items() if got.get(k) != v]
        status = "PASS" if not fails else "FAIL"
        if fails:
            ok = False
        print(f"[{status}] {name}")
        print(f"        tp={got['n_triple_patterns']} "
              f"depth_match={got['depth_match']} "
              f"trace_words={got['trace_words']} "
              f"grounding={got['literal_grounding']} "
              f"ans={got['final_answer']!r}")
        for f in fails:
            print(f"        ! {f}")
    print("\n" + ("ALL PASS" if ok else "SOME FAILURES"))
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--jsonl", type=Path,
                   help="A *_fewshot.jsonl file with row['trace']['raw'] chains.")
    p.add_argument("--context-json", type=Path,
                   help="Optional {id: context} map for grounding.")
    args = p.parse_args()

    if args.selftest or not args.jsonl:
        _selftest()
        return

    ctx_map = {}
    if args.context_json and args.context_json.exists():
        ctx_map = json.loads(args.context_json.read_text(encoding="utf-8"))

    rows = [json.loads(l) for l in
            args.jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    feats = []
    for r in rows:
        raw = (r.get("trace") or {}).get("raw", "")
        ctx = ctx_map.get(str(r.get("id")), "")
        f = analyze_chain(raw, ctx)
        f["verdict"] = r.get("verdict")
        feats.append(f)

    n = len(feats)
    parsed = [f for f in feats if f["parse_ok"]]
    print(f"rows={n}  parse_ok={len(parsed)} ({len(parsed)/max(n,1):.0%})")
    if parsed:
        tps = [f["n_triple_patterns"] for f in parsed]
        print(f"triple-pattern count: mean={sum(tps)/len(tps):.2f} "
              f"min={min(tps)} max={max(tps)}")
        grounded = [f["literal_grounding"] for f in parsed
                    if f["literal_grounding"] is not None]
        if grounded:
            print(f"literal grounding: mean={sum(grounded)/len(grounded):.2f}")


if __name__ == "__main__":
    main()
