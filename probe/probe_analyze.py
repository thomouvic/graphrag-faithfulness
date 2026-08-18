"""
Analyze probe results: compute the (covered × correct) transition matrix
per cell, and check the decision rule from the probe report.

Inputs:
    --probe-dir   directory with {base}_{method}.jsonl from probe_run.py
    --subset-json output of probe_compute_subset.py (per-question covered flags)

Output:
    Decision report JSON + printed transition matrices.

Decision rule:
    STRONG SIGNAL  - KET vs LightRAG IRCoT gain differs by >5pp AND
                     gain decompositions differ qualitatively
    MIXED SIGNAL   - both bases gain similarly (<3pp gap)
    FAILURE        - IRCoT barely moves either (<2pp on both)
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            out[str(row["id"])] = row
    return out


def transitions(base: str, sparql_rows: dict, ircot_rows: dict,
                covered_flag_map: dict) -> dict:
    """For each question, classify the (covered, sparql_verdict, ircot_verdict)
    state and compute the transition matrix.

    States:
      cs: covered, sparql correct
      cw: covered, sparql wrong (= covered-but-wrong, target population)
      nc: not covered
    """
    common = set(sparql_rows) & set(ircot_rows) & set(covered_flag_map)
    n = len(common)

    counts = {
        "cs->c": 0, "cs->w": 0,
        "cw->c": 0, "cw->w": 0,
        "nc->c": 0, "nc->w": 0,
    }
    detail = []
    for qid in sorted(common):
        cov = covered_flag_map[qid]
        sv = sparql_rows[qid]["verdict"] == "correct"
        iv = ircot_rows[qid]["verdict"] == "correct"
        if cov and sv:
            state = "cs"
        elif cov and not sv:
            state = "cw"
        else:
            state = "nc"
        target = "c" if iv else "w"
        counts[f"{state}->{target}"] += 1
        detail.append({"qid": qid, "covered": cov, "sparql": sv,
                       "ircot": iv, "state": state, "target": target})

    cs_total = counts["cs->c"] + counts["cs->w"]
    cw_total = counts["cw->c"] + counts["cw->w"]
    nc_total = counts["nc->c"] + counts["nc->w"]

    # Total-correct counts for apples-to-apples accuracy comparison.
    # Earlier versions of this analyzer treated `sparql_correct = cs_total`
    # which under-counted SPARQL: NC cases where SPARQL was right via
    # parametric knowledge weren't credited. Use the full verdict set.
    sparql_correct = sum(1 for qid in common
                         if sparql_rows[qid]["verdict"] == "correct")
    ircot_correct = counts["cs->c"] + counts["cw->c"] + counts["nc->c"]
    delta_correct = ircot_correct - sparql_correct
    cbw_recovery = counts["cw->c"]
    coverage_recovery = counts["nc->c"]
    regression = counts["cs->w"]

    return {
        "base": base, "n": n,
        "cs_total": cs_total, "cw_total": cw_total, "nc_total": nc_total,
        "counts": counts,
        "sparql_acc": sparql_correct / n if n else 0,
        "ircot_acc": ircot_correct / n if n else 0,
        "delta_acc": (ircot_correct - sparql_correct) / n if n else 0,
        "cbw_recovery": cbw_recovery,
        "cbw_recovery_rate": cbw_recovery / cw_total if cw_total else 0,
        "coverage_recovery": coverage_recovery,
        "coverage_recovery_rate": coverage_recovery / nc_total if nc_total else 0,
        "regression": regression,
        "detail": detail,
    }


def decision(ket_T: dict, lr_T: dict) -> dict:
    """Apply the decision rule from the probe report."""
    ket_delta = ket_T["delta_acc"]
    lr_delta = lr_T["delta_acc"]
    delta_gap = abs(ket_delta - lr_delta)

    # Mechanism profile: ratio of cbw_recovery to coverage_recovery
    def profile(T):
        cbw = T["cbw_recovery"]
        cov = T["coverage_recovery"]
        total = cbw + cov
        if total == 0:
            return "no_recovery"
        cbw_share = cbw / total
        if cbw_share > 0.66:
            return "reasoning_dominant"
        elif cbw_share < 0.33:
            return "coverage_dominant"
        else:
            return "mixed"

    ket_profile = profile(ket_T)
    lr_profile = profile(lr_T)
    profiles_differ = ket_profile != lr_profile

    # Decision logic — there are TWO ways to support the headline claim:
    #   (i)  Different MAGNITUDES of gain (delta_gap > 5pp)
    #   (ii) Different MECHANISMS of gain (profiles differ AND each base has a
    #        non-trivial recovery — i.e. it's not "no_recovery" on either side)
    # Either alone is enough. Both together is the cleanest signal.
    nontrivial_recovery = (
        ket_T["cbw_recovery"] + ket_T["coverage_recovery"] >= 3
        and lr_T["cbw_recovery"] + lr_T["coverage_recovery"] >= 3
    )

    if abs(ket_delta) < 0.02 and abs(lr_delta) < 0.02:
        verdict = "FAILURE"
        action = ("IRCoT barely moves either base. Pivot to memorization audit "
                  "(closed-book vs Graph-RAG) — one-day experiment, separate paper.")
    elif delta_gap > 0.05 and profiles_differ and nontrivial_recovery:
        verdict = "STRONG"
        action = ("Headline claim alive on BOTH magnitude AND mechanism axes. "
                  "Commit to full plan: scale to 500 questions across three "
                  "benchmarks; add cost-conditioned analysis; tighten the "
                  "mechanistic explanation.")
    elif profiles_differ and nontrivial_recovery:
        verdict = "STRONG_MECHANISM"
        action = ("Magnitudes are similar but mechanisms differ qualitatively "
                  f"(KET={ket_profile}, LightRAG={lr_profile}). This IS the "
                  "headline claim — the asymmetry is mechanistic, not magnitudinal. "
                  "Commit to full plan; reframe headline as: 'same gain, "
                  "different reason — retrieval structure governs which "
                  "failure mode iteration repairs.'")
    elif delta_gap > 0.05 and not profiles_differ:
        verdict = "MAGNITUDE_ONLY"
        action = ("Magnitudes differ but both bases recover via the same "
                  "mechanism. Defensible but weaker paper: 'iteration helps "
                  "more on base X, but for the same reason.' Consider "
                  "framing as a Pareto/efficiency story instead of asymmetry.")
    elif delta_gap < 0.03 and not profiles_differ:
        verdict = "MIXED"
        action = ("Both bases respond similarly to IRCoT in both magnitude "
                  "and mechanism. Headline asymmetry claim is dead. Reframe "
                  "as 'iterative reasoning saturates uniformly on retrieval-"
                  "side SOTA' OR pivot to memorization audit.")
    else:
        verdict = "BORDERLINE"
        action = ("Signal is ambiguous. Recommended: rerun on 100 MuSiQue "
                  "questions to triangulate, OR upgrade to live IRCoT "
                  "(5-10 days more engineering) for higher-resolution signal.")

    return {
        "verdict": verdict, "action": action,
        "ket_delta": ket_delta, "lightrag_delta": lr_delta,
        "delta_gap": delta_gap,
        "ket_profile": ket_profile, "lightrag_profile": lr_profile,
        "profiles_differ": profiles_differ,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-dir", required=True, type=Path)
    p.add_argument("--subset-json", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    subset = json.loads(args.subset_json.read_text(encoding="utf-8"))
    per_q = subset["per_question"]

    ket_cov = {qid: r["ket_covered"] for qid, r in per_q.items()}
    lr_cov = {qid: r["lightrag_covered"] for qid, r in per_q.items()}

    ket_sparql = load_jsonl(args.probe_dir / "ket_sparql.jsonl")
    ket_ircot = load_jsonl(args.probe_dir / "ket_ircot.jsonl")
    lr_sparql = load_jsonl(args.probe_dir / "lightrag_sparql.jsonl")
    lr_ircot = load_jsonl(args.probe_dir / "lightrag_ircot.jsonl")

    print("=== Loaded ===")
    print(f"  ket sparql: {len(ket_sparql)}  ircot: {len(ket_ircot)}")
    print(f"  lr  sparql: {len(lr_sparql)}  ircot: {len(lr_ircot)}")

    ket_T = transitions("ket", ket_sparql, ket_ircot, ket_cov)
    lr_T = transitions("lightrag", lr_sparql, lr_ircot, lr_cov)

    def fmt(T):
        c = T["counts"]
        return (
            f"  base={T['base']}  n={T['n']}\n"
            f"    sparql_acc={T['sparql_acc']:.3f}  ircot_acc={T['ircot_acc']:.3f}  "
            f"delta={T['delta_acc']:+.3f}\n"
            f"    states  cs={T['cs_total']}  cw={T['cw_total']}  nc={T['nc_total']}\n"
            f"    transitions:\n"
            f"      covered+sparql_correct  -> ircot correct: {c['cs->c']:3d}   wrong: {c['cs->w']:3d}\n"
            f"      covered+sparql_wrong    -> ircot correct: {c['cw->c']:3d}   wrong: {c['cw->w']:3d}\n"
            f"      not_covered             -> ircot correct: {c['nc->c']:3d}   wrong: {c['nc->w']:3d}\n"
            f"    cbw_recovery={T['cbw_recovery']}/{T['cw_total']} "
            f"({T['cbw_recovery_rate']:.2%})  "
            f"coverage_recovery={T['coverage_recovery']}/{T['nc_total']} "
            f"({T['coverage_recovery_rate']:.2%})  "
            f"regression={T['regression']}/{T['cs_total']}"
        )

    print("\n=== KET-RAG ===")
    print(fmt(ket_T))
    print("\n=== LightRAG ===")
    print(fmt(lr_T))

    d = decision(ket_T, lr_T)
    print("\n=== DECISION ===")
    print(f"  Verdict: {d['verdict']}")
    print(f"  KET delta:       {d['ket_delta']:+.3f}  profile: {d['ket_profile']}")
    print(f"  LightRAG delta:  {d['lightrag_delta']:+.3f}  profile: {d['lightrag_profile']}")
    print(f"  Delta gap:       {d['delta_gap']:.3f}")
    print(f"\n  Action: {d['action']}")

    args.out.write_text(json.dumps({
        "decision": d,
        "ket_transitions": ket_T,
        "lightrag_transitions": lr_T,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
