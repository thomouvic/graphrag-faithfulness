"""
Adjudicate: is the 8B-vs-70B gap a retrieval-recall wall or a use problem?

Joins the upstream supporting-fact coverage CSVs (does the retrieved context
actually contain the gold entities / answer?) with the 8B and 70B SPARQL-CoT
verdicts, per question. No API calls.

Key strata:
  - answer-present (substring_covered=True): the gold answer string is sitting
    in the context. Accuracy here is a clean 'can the model use what it has'
    measure. A large 8B<70B gap here => needle-finding / noise deficit, and the
    answer is reachable (evidence-selection or symbolic execution should help).
  - answer-absent: recall-limited; no 8B-side method can fix these.
"""
import csv
import glob
import os
from pathlib import Path

csv.field_size_limit(10_000_000)
UP = Path(os.environ["PAPER1_REPO"])  # Paper 1 (arXiv 2603.14045) base repo
BENCHES = ["hotpotqa", "musique", "2wikimultihopqa"]


def load_verdicts(bench, sub):
    f = sorted(glob.glob(str(UP / "experiments" / bench / "large_scale" /
                              "results_sparql_groq" / f"*{sub}*.csv")))
    if not f:
        return {}
    out = {}
    for r in csv.DictReader(open(f[-1], encoding="utf-8")):
        out[str(r["id"])] = r["eval_verdict"].strip().lower() == "correct"
    return out


def load_coverage(bench):
    f = UP / "revision" / "coverage" / f"{bench}_coverage.csv"
    out = {}
    for r in csv.DictReader(open(f, encoding="utf-8")):
        out[str(r["id"])] = {
            "all_present": r["all_entities_present"].strip().lower() == "true",
            "substr": r["substring_covered"].strip().lower() == "true",
            "ctx": int(r["context_chars"]) if r["context_chars"] else 0,
        }
    return out


def acc(ids, v):
    xs = [v[i] for i in ids if i in v]
    return (sum(xs) / len(xs), len(xs)) if xs else (float("nan"), 0)


def main():
    for bench in BENCHES:
        cov = load_coverage(bench)
        v8 = load_verdicts(bench, "llama-3.1-8b")
        v70 = load_verdicts(bench, "llama-3.3-70b")
        ids = [i for i in cov if i in v8 and i in v70]

        present = [i for i in ids if cov[i]["substr"]]
        absent = [i for i in ids if not cov[i]["substr"]]
        ent_present = [i for i in ids if cov[i]["all_present"]]

        print(f"\n=== {bench}  (joined n={len(ids)}) ===")
        print(f"  answer-in-context rate (substr): {len(present)/len(ids):.0%}   "
              f"all-entities-present rate: {len(ent_present)/len(ids):.0%}")
        for label, sub in [("ALL", ids),
                           ("answer PRESENT", present),
                           ("answer ABSENT", absent),
                           ("all-entities present", ent_present)]:
            a8, n = acc(sub, v8)
            a70, _ = acc(sub, v70)
            gap = a70 - a8
            print(f"  {label:24} n={n:4}  8B={a8:.2f}  70B={a70:.2f}  gap={gap:+.2f}")

        # Noise test: within answer-PRESENT, does 8B acc fall as context grows?
        if present:
            present_sorted = sorted(present, key=lambda i: cov[i]["ctx"])
            half = len(present_sorted) // 2
            lo, hi = present_sorted[:half], present_sorted[half:]
            a8lo, _ = acc(lo, v8); a8hi, _ = acc(hi, v8)
            a70lo, _ = acc(lo, v70); a70hi, _ = acc(hi, v70)
            mlo = sum(cov[i]["ctx"] for i in lo)/len(lo)
            mhi = sum(cov[i]["ctx"] for i in hi)/len(hi)
            print(f"  [noise|answer-present]  short-ctx(~{mlo:.0f}ch): 8B={a8lo:.2f} 70B={a70lo:.2f}"
                  f"   long-ctx(~{mhi:.0f}ch): 8B={a8hi:.2f} 70B={a70hi:.2f}")


if __name__ == "__main__":
    main()
