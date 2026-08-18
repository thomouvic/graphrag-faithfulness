"""
SC + GW: graph-walk-compress the context first, then run SPARQL-CoT + SC
over the compressed context. The natural product of Paper 1's GW + this
project's SC.

Same interface as probe_run_sc.py — adds a single pre-compression step.

Usage:
    python -u probe_run_sc_gw.py \
        --qa-pairs experiments/hotpotqa/large_scale/qa-pairs/qa-pairs.json \
        --ket-context experiments/hotpotqa/large_scale/output/large_scale-keyword-0.5.json \
        --lightrag-context experiments/hotpotqa/large_scale_lightrag/lightrag_contexts.json \
        --limit 100 --k 3 --temperature 0.7 \
        --out-dir probe_results/
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from qa_pipeline import eval_once, is_abstain
from probe_sparql_sc import answer_with_sparql_cot_sc
from probe_compute_subset import (
    load_ket_contexts, load_lightrag_contexts, _gold_strings,
)
from probe_gw import build_gw_lookup

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
QA_MODEL = "meta-llama/llama-3.1-8b-instruct"
EVAL_MODEL = "meta-llama/llama-3.1-8b-instruct"


def _eval(client, q, gold, pred):
    try:
        v = eval_once(client, EVAL_MODEL, q, gold, pred)
        return v.get("verdict", "unknown"), v.get("reason", "")
    except Exception as e:
        return "unknown", repr(e)[:200]


def _run_cell(client, qa_subset, gw_lookup, base, k, temperature, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    rows = []
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                done.add(str(row["id"]))
        print(f"  [resume] {base}+sparql_sc_gw: {len(done)} already done")

    t0 = time.time()
    n = len(qa_subset)
    for i, q in enumerate(qa_subset):
        qid = str(q["id"])
        if qid in done:
            continue
        question = q["question"]
        gold_list = _gold_strings(q)
        ctx = gw_lookup.get(qid, "")
        if not ctx.strip():
            pred = "I don't know"
            trace = {"k": k, "samples": [], "temperature": temperature}
        else:
            try:
                pred, trace = answer_with_sparql_cot_sc(
                    client, QA_MODEL, question, ctx,
                    k=k, temperature=temperature, max_tokens=512,
                )
            except Exception as e:
                pred = f"ERROR: {e}"
                trace = {"k": k, "samples": [], "temperature": temperature,
                         "error": repr(e)[:200]}
        verdict, reason = _eval(client, question, gold_list, pred)
        norms = [s["normalized"] for s in trace["samples"]]
        row = {"id": qid, "question": question, "gold": gold_list,
               "pred": pred, "verdict": verdict, "reason": reason,
               "sc_k": k, "sc_temperature": temperature,
               "sc_unique_answers": len(set(norms)),
               "sc_norms": norms, "ctx_chars": len(ctx),
               "base": base, "method": "sparql_sc_gw"}
        rows.append(row)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done.add(qid)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            correct = sum(1 for r in rows if r["verdict"] == "correct")
            print(f"  [{base}+sparql_sc_gw] {i+1}/{n} acc={correct/len(rows):.3f} "
                  f"elapsed={time.time()-t0:.0f}s")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa-pairs", required=True, type=Path)
    p.add_argument("--ket-context", required=True, type=Path)
    p.add_argument("--lightrag-context", required=True, type=Path)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-hops", type=int, default=3)
    p.add_argument("--budget-tokens", type=int, default=4000)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    load_dotenv(".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set"); sys.exit(1)

    qa_list = json.loads(args.qa_pairs.read_text(encoding="utf-8"))
    qa_subset = qa_list[:args.limit]
    print(f"Loaded {len(qa_subset)} questions")

    ket_ctx = load_ket_contexts(args.ket_context)
    lr_ctx = load_lightrag_contexts(args.lightrag_context)

    # Pre-compress with GW
    print("Compressing KET contexts via GW...")
    ket_gw, ket_stats = build_gw_lookup(qa_subset, ket_ctx, "ket",
                                        max_hops=args.max_hops,
                                        budget_tokens=args.budget_tokens)
    print(f"  compressed={ket_stats['n_compressed']} fallback={ket_stats['n_fallback']} "
          f"avg_words {ket_stats['avg_orig_words']:.0f} -> {ket_stats['avg_new_words']:.0f}")

    print("Compressing LightRAG contexts via GW...")
    lr_gw, lr_stats = build_gw_lookup(qa_subset, lr_ctx, "lightrag",
                                      max_hops=args.max_hops,
                                      budget_tokens=args.budget_tokens)
    print(f"  compressed={lr_stats['n_compressed']} fallback={lr_stats['n_fallback']} "
          f"avg_words {lr_stats['avg_orig_words']:.0f} -> {lr_stats['avg_new_words']:.0f}")

    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url=OPENROUTER_BASE_URL)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells = [("ket", ket_gw), ("lightrag", lr_gw)]
    summary = {}
    for base, gw in cells:
        out_path = args.out_dir / f"{base}_sparql_sc_gw.jsonl"
        print(f"\n=== Cell: {base} + sparql_sc_gw (k={args.k}, T={args.temperature}) ===")
        rows = _run_cell(client, qa_subset, gw, base, args.k, args.temperature, out_path)
        correct = sum(1 for r in rows if r["verdict"] == "correct")
        summary[f"{base}_sparql_sc_gw"] = {
            "n": len(rows),
            "acc": correct / len(rows) if rows else 0,
        }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:30s}  acc={v['acc']:.3f}  n={v['n']}")
    (args.out_dir / "summary_sc_gw.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
