"""
Probe runner: 4-cell experiment to falsify or confirm the headline claim.

Cells:
    KET-RAG  + SPARQL-CoT (control, already-implemented method)
    KET-RAG  + IRCoT-frozen (new)
    LightRAG + SPARQL-CoT (control)
    LightRAG + IRCoT-frozen (new)

On 100 HotpotQA questions (first 100 of large_scale split).

Usage (from repo root, after main paper's setup.py has been run):
    python probe_run.py \
        --qa-pairs experiments/hotpotqa/large_scale/qa-pairs/qa-pairs.json \
        --ket-context experiments/hotpotqa/large_scale/output/large_scale-keyword-0.5.json \
        --lightrag-context experiments/hotpotqa/large_scale_lightrag/lightrag_contexts.json \
        --limit 100 \
        --out-dir probe_results
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from qa_pipeline import (
    answer_with_sparql_cot,
    answer_with_context,
    eval_once,
    is_abstain,
)
from probe_ircot_frozen import answer_with_ircot_frozen
from probe_compute_subset import (
    load_ket_contexts, load_lightrag_contexts, _gold_strings,
)

# OpenRouter via OpenAI-compatible API. The repo's qa_pipeline.call_groq_chat
# is client-agnostic: it only invokes client.chat.completions.create(...),
# which both Groq SDK and OpenAI SDK implement identically.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
QA_MODEL = "meta-llama/llama-3.1-8b-instruct"
EVAL_MODEL = "meta-llama/llama-3.1-8b-instruct"


def _eval_row(client, question, gold_list, pred):
    try:
        v = eval_once(client, EVAL_MODEL, question, gold_list, pred)
        return v.get("verdict", "unknown"), v.get("reason", "")
    except Exception as e:
        return "unknown", repr(e)[:200]


def _run_cell(client, embedder, qa_subset, ctx_lookup, base_name, method_name,
              out_path: Path):
    """Run one cell. Resumable via JSONL checkpoint at out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    rows = []
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                done.add(str(row["id"]))
        print(f"  [resume] {base_name}+{method_name}: {len(done)} already done")

    t0 = time.time()
    n = len(qa_subset)
    for i, q in enumerate(qa_subset):
        qid = str(q["id"])
        if qid in done:
            continue
        question = q["question"]
        gold_list = _gold_strings(q)
        ctx = ctx_lookup.get(qid, "")
        if not ctx.strip():
            row = {"id": qid, "question": question, "gold": gold_list,
                   "pred": "I don't know", "verdict": "wrong",
                   "reason": "no_context", "n_iters": 0,
                   "early_stop": "no_context",
                   "base": base_name, "method": method_name}
        else:
            try:
                if method_name == "sparql":
                    pred, raw = answer_with_sparql_cot(
                        client, QA_MODEL, question, ctx,
                        max_tokens=512, temperature=0.0,
                    )
                    n_iters = 1
                    early = "sparql_singleshot"
                elif method_name == "ircot":
                    base_kw = "ket" if base_name == "ket" else "lightrag"
                    pred, trace = answer_with_ircot_frozen(
                        client, QA_MODEL, question, ctx, embedder,
                        max_iters=5, top_k_init=8, top_k_step=3,
                        max_tokens=256, temperature=0.0, base=base_kw,
                    )
                    n_iters = trace["n_iters"]
                    early = trace["early_stop"]
                else:
                    raise ValueError(method_name)
                verdict, reason = _eval_row(client, question, gold_list, pred)
            except Exception as e:
                pred = f"ERROR: {e}"
                verdict, reason = "unknown", repr(e)[:200]
                n_iters = 0
                early = "error"
            row = {"id": qid, "question": question, "gold": gold_list,
                   "pred": pred, "verdict": verdict, "reason": reason,
                   "n_iters": n_iters, "early_stop": early,
                   "base": base_name, "method": method_name}
        rows.append(row)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done.add(qid)
        elapsed = time.time() - t0
        if (i + 1) % 10 == 0 or (i + 1) == n:
            correct = sum(1 for r in rows if r["verdict"] == "correct")
            print(f"  [{base_name}+{method_name}] {i+1}/{n} "
                  f"acc={correct/len(rows):.3f} elapsed={elapsed:.0f}s")
    return rows


def _summarize(rows):
    n = len(rows)
    correct = sum(1 for r in rows if r["verdict"] == "correct")
    abstain = sum(1 for r in rows if is_abstain(r.get("pred", "")))
    return {"n": n, "acc": correct / n if n else 0, "abstain_rate": abstain / n if n else 0,
            "avg_iters": sum(r.get("n_iters", 0) for r in rows) / n if n else 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa-pairs", required=True, type=Path)
    p.add_argument("--ket-context", required=True, type=Path)
    p.add_argument("--lightrag-context", required=True, type=Path)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set"); sys.exit(1)

    qa_list = json.loads(args.qa_pairs.read_text(encoding="utf-8"))
    qa_subset = qa_list[:args.limit]
    print(f"Loaded {len(qa_subset)} questions")

    ket_ctx = load_ket_contexts(args.ket_context)
    lr_ctx = load_lightrag_contexts(args.lightrag_context)

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells = [
        ("ket",      "sparql", ket_ctx),
        ("ket",      "ircot",  ket_ctx),
        ("lightrag", "sparql", lr_ctx),
        ("lightrag", "ircot",  lr_ctx),
    ]
    cell_rows = {}
    for base, method, ctx in cells:
        out_path = args.out_dir / f"{base}_{method}.jsonl"
        print(f"\n=== Cell: {base} + {method} ===")
        rows = _run_cell(client, embedder, qa_subset, ctx, base, method, out_path)
        cell_rows[(base, method)] = rows

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary = {}
    for (base, method), rows in cell_rows.items():
        s = _summarize(rows)
        summary[f"{base}_{method}"] = s
        print(f"  {base:9s} + {method:7s}  "
              f"acc={s['acc']:.3f}  abstain={s['abstain_rate']:.3f}  "
              f"avg_iters={s['avg_iters']:.1f}  n={s['n']}")

    # Cell deltas
    print("\nDeltas (IRCoT vs SPARQL, paired on same question IDs):")
    for base in ["ket", "lightrag"]:
        s_rows = {r["id"]: r for r in cell_rows[(base, "sparql")]}
        i_rows = {r["id"]: r for r in cell_rows[(base, "ircot")]}
        common = set(s_rows) & set(i_rows)
        if not common:
            continue
        s_acc = sum(1 for q in common if s_rows[q]["verdict"] == "correct") / len(common)
        i_acc = sum(1 for q in common if i_rows[q]["verdict"] == "correct") / len(common)
        print(f"  {base:9s}  sparql={s_acc:.3f}  ircot={i_acc:.3f}  delta={i_acc - s_acc:+.3f}  n={len(common)}")

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
