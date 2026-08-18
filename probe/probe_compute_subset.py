"""
Derive the per-question coverage and covered-but-wrong subsets.

Joins:
  - Coverage: does the gold answer string appear in the original (uncompressed)
    retrieved context for question Q under base B?
  - Verdict: did the baseline (or any specific method) answer Q correctly?

Outputs a JSON: {qid: {ket_covered, ket_baseline_correct,
                       lightrag_covered, lightrag_baseline_correct, ...}}

Usage:
    python probe_compute_subset.py \
        --qa-pairs experiments/hotpotqa/large_scale/qa-pairs/qa-pairs.json \
        --ket-context experiments/hotpotqa/large_scale/output/large_scale-keyword-0.5.json \
        --lightrag-context experiments/hotpotqa/large_scale_lightrag/lightrag_contexts.json \
        --ket-baseline experiments/hotpotqa/large_scale/checkpoints/baseline.jsonl \
        --lightrag-baseline experiments/hotpotqa/large_scale_lightrag/results/Baseline_8B.jsonl \
        --limit 100 \
        --out probe_subset.json
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _gold_strings(q: dict) -> list:
    """All acceptable gold answer surface forms."""
    out = []
    a = q.get("answer")
    if isinstance(a, str):
        out.append(a)
    elif isinstance(a, list):
        out.extend(a)
    aa = q.get("answers")
    if isinstance(aa, list):
        out.extend([x for x in aa if isinstance(x, str)])
    return [s for s in out if s and s.strip()]


def covered(gold_list: list, context: str) -> bool:
    """Substring-presence check (normalized)."""
    if not context:
        return False
    ctx_n = _norm(context)
    for g in gold_list:
        gn = _norm(g)
        if gn and gn in ctx_n:
            return True
    return False


def load_ket_contexts(ket_context_path: Path) -> dict:
    """KET-RAG output JSON is a list of records [{id, context, ...}, ...]."""
    data = json.loads(ket_context_path.read_text(encoding="utf-8"))
    out = {}
    for rec in data:
        qid = str(rec.get("id") or rec.get("query_id") or rec.get("qid"))
        ctx = rec.get("context", "") or rec.get("retrieved_context", "")
        out[qid] = ctx
    return out


def load_lightrag_contexts(lr_path: Path) -> dict:
    """LightRAG cache is {qid: context_string}."""
    return json.loads(lr_path.read_text(encoding="utf-8"))


def load_baseline_verdicts(jsonl_path: Path) -> dict:
    """Baseline checkpoint JSONL — returns {qid: 'correct'/'wrong'}."""
    out = {}
    if not jsonl_path or not Path(jsonl_path).exists():
        return out
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        qid = str(row.get("id"))
        verdict = row.get("eval_verdict", "")
        out[qid] = "correct" if verdict == "correct" else "wrong"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa-pairs", required=True, type=Path)
    p.add_argument("--ket-context", required=True, type=Path)
    p.add_argument("--lightrag-context", required=True, type=Path)
    p.add_argument("--ket-baseline", required=True, type=Path,
                   help="JSONL checkpoint of KET-RAG baseline runs")
    p.add_argument("--lightrag-baseline", required=True, type=Path,
                   help="JSONL of LightRAG baseline runs")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    qa_list = json.loads(args.qa_pairs.read_text(encoding="utf-8"))
    qa_subset = qa_list[:args.limit]
    print(f"Loaded {len(qa_subset)} QA pairs (first {args.limit} of {len(qa_list)})")

    ket_ctx = load_ket_contexts(args.ket_context)
    lr_ctx = load_lightrag_contexts(args.lightrag_context)
    print(f"  KET contexts: {len(ket_ctx)}  LightRAG contexts: {len(lr_ctx)}")

    ket_base = load_baseline_verdicts(args.ket_baseline)
    lr_base = load_baseline_verdicts(args.lightrag_baseline)
    print(f"  KET baseline: {len(ket_base)}  LightRAG baseline: {len(lr_base)}")

    rows = {}
    for q in qa_subset:
        qid = str(q["id"])
        golds = _gold_strings(q)
        rows[qid] = {
            "question": q.get("question", ""),
            "gold": golds,
            "ket_covered": covered(golds, ket_ctx.get(qid, "")),
            "ket_baseline": ket_base.get(qid, "missing"),
            "lightrag_covered": covered(golds, lr_ctx.get(qid, "")),
            "lightrag_baseline": lr_base.get(qid, "missing"),
        }

    # Summarize
    n = len(rows)
    ket_cov = sum(1 for r in rows.values() if r["ket_covered"])
    lr_cov = sum(1 for r in rows.values() if r["lightrag_covered"])
    ket_cbw = sum(1 for r in rows.values()
                  if r["ket_covered"] and r["ket_baseline"] == "wrong")
    lr_cbw = sum(1 for r in rows.values()
                 if r["lightrag_covered"] and r["lightrag_baseline"] == "wrong")

    summary = {
        "n_questions": n,
        "ket_coverage": ket_cov / n if n else 0,
        "ket_covered_but_wrong": ket_cbw,
        "ket_covered_but_wrong_pct": ket_cbw / ket_cov if ket_cov else 0,
        "lightrag_coverage": lr_cov / n if n else 0,
        "lightrag_covered_but_wrong": lr_cbw,
        "lightrag_covered_but_wrong_pct": lr_cbw / lr_cov if lr_cov else 0,
    }

    print(f"\n=== Summary ===")
    print(f"  N: {n}")
    print(f"  KET coverage: {summary['ket_coverage']:.2%}  CBW: {ket_cbw} "
          f"({summary['ket_covered_but_wrong_pct']:.2%} of covered)")
    print(f"  LR  coverage: {summary['lightrag_coverage']:.2%}  CBW: {lr_cbw} "
          f"({summary['lightrag_covered_but_wrong_pct']:.2%} of covered)")

    args.out.write_text(json.dumps(
        {"summary": summary, "per_question": rows},
        indent=2, ensure_ascii=False
    ), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
