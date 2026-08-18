"""
Pilot: structural comparison of 8B vs 70B baseline SPARQL-CoT chains.

Uses the *existing* no-few-shot SPARQL-CoT detail CSVs from the upstream
Paper 1 base repo. No API calls, no re-run. Tests the mechanism hypothesis on
the baseline gap the few-shot work aims to close:

  Does the 70B emit structurally different reasoning chains than the 8B
  (triple-pattern count, trace length, parse rate), and does chain structure
  track correctness within each model?

If yes, the few-shot re-run is worth it. If 8B and 70B chains look the same,
the structural-mechanism angle is in doubt.
"""
import csv
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from probe_chain_structure import analyze_chain

csv.field_size_limit(10_000_000)

UPSTREAM = Path(os.environ["PAPER1_REPO"]) / "experiments"  # Paper 1 (arXiv 2603.14045) base repo
BENCHES = ["hotpotqa", "musique", "2wikimultihopqa"]


def latest(bench, model_substr):
    d = UPSTREAM / bench / "large_scale" / "results_sparql_groq"
    hits = sorted(glob.glob(str(d / f"*{model_substr}*.csv")))
    return hits[-1] if hits else None


def gold_hops_from_id(qid: str):
    # MuSiQue ids look like '2hop__...', '3hop__...', '4hop__...'
    for h in (2, 3, 4):
        if qid.startswith(f"{h}hop"):
            return h
    return None


def summarize(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    feats = []
    for r in rows:
        gh = gold_hops_from_id(str(r.get("id", "")))
        f = analyze_chain(r.get("sparql_cot", ""), gold_hops=gh)
        f["correct"] = (r.get("eval_verdict", "").strip().lower() == "correct")
        feats.append(f)
    n = len(feats)
    parsed = [f for f in feats if f["parse_ok"]]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    correct = [f for f in feats if f["correct"]]
    wrong = [f for f in feats if not f["correct"]]
    return {
        "n": n,
        "acc": mean([f["correct"] for f in feats]),
        "parse_ok": mean([f["parse_ok"] for f in feats]),
        "tp_mean": mean([f["n_triple_patterns"] for f in parsed]),
        "trace_words": mean([f["trace_words"] for f in parsed]),
        "depth_match": mean([f["depth_match"] for f in parsed]),
        "tp_correct": mean([f["n_triple_patterns"] for f in correct if f["parse_ok"]]),
        "tp_wrong": mean([f["n_triple_patterns"] for f in wrong if f["parse_ok"]]),
        "parse_correct": mean([f["parse_ok"] for f in correct]),
        "parse_wrong": mean([f["parse_ok"] for f in wrong]),
    }


def main():
    hdr = ("bench/model", "n", "acc", "parseOK", "tp_mean", "trace_w",
           "depthM", "tp_corr", "tp_wrong", "parseOK_corr", "parseOK_wrong")
    print(f"{hdr[0]:<22}" + "".join(f"{h:>10}" for h in hdr[1:]))
    print("-" * 132)
    for bench in BENCHES:
        for tag, sub in [("8b", "llama-3.1-8b"), ("70b", "llama-3.3-70b")]:
            p = latest(bench, sub)
            if not p:
                print(f"{bench}/{tag:<18} (no file)")
                continue
            s = summarize(p)
            print(f"{bench+'/'+tag:<22}"
                  f"{s['n']:>10}{s['acc']:>10.2f}{s['parse_ok']:>10.2f}"
                  f"{s['tp_mean']:>10.2f}{s['trace_words']:>10.1f}"
                  f"{s['depth_match'] if s['depth_match']==s['depth_match'] else float('nan'):>10.2f}"
                  f"{s['tp_correct']:>10.2f}{s['tp_wrong']:>10.2f}"
                  f"{s['parse_correct']:>10.2f}{s['parse_wrong']:>10.2f}")
        print()


if __name__ == "__main__":
    main()
