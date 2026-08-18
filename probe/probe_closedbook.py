"""
Closed-book audit: ask Llama-3.1-8B the same 100 HotpotQA questions with
NO retrieved context. Use the same eval pipeline as the rest of the probe.

Purpose
-------
Tests how much of Graph-RAG's reported accuracy is parametric memorization
vs retrieval-grounded reasoning. The "Graph-RAG advantage" per question is
defined as (graph_rag_correct AND closed_book_wrong). The "parametric leak"
per question is (graph_rag_correct AND closed_book_correct).

Comparable to MRKE/CofCA but applied to the Graph-RAG-family eval rather
than to the LLM in isolation.

Output: probe_results/closedbook_8b.jsonl with the same row schema as the
SPARQL/IRCoT cells, so probe_analyze can join on qid.

Usage:
    python probe_closedbook.py \
        --qa-pairs experiments/hotpotqa/large_scale/qa-pairs/qa-pairs.json \
        --limit 100 \
        --out probe_results/closedbook_8b.jsonl
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
from probe_compute_subset import _gold_strings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
QA_MODEL = "meta-llama/llama-3.1-8b-instruct"
EVAL_MODEL = "meta-llama/llama-3.1-8b-instruct"


def answer_closed_book(client, question: str) -> str:
    """Ask the model to answer with no context, in 1-5 words."""
    resp = client.chat.completions.create(
        model=QA_MODEL,
        messages=[
            {"role": "system",
             "content": "Answer in 1-5 words. If you do not know, reply: I don't know"},
            {"role": "user", "content": f"QUESTION: {question}\nANSWER:"},
        ],
        max_tokens=20,
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    # Strip echoed prefixes
    for prefix in ("ANSWER:", "Answer:", "answer:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.split("\n")[0].strip().strip('"\'')
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa-pairs", required=True, type=Path)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    load_dotenv(".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set"); sys.exit(1)

    qa_list = json.loads(args.qa_pairs.read_text(encoding="utf-8"))
    qa_subset = qa_list[:args.limit]
    print(f"Loaded {len(qa_subset)} questions")

    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url=OPENROUTER_BASE_URL)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(str(json.loads(line)["id"]))
        print(f"  [resume] {len(done)} already done")

    rows = []
    t0 = time.time()
    for i, q in enumerate(qa_subset):
        qid = str(q["id"])
        if qid in done:
            continue
        question = q["question"]
        gold_list = _gold_strings(q)
        try:
            pred = answer_closed_book(client, question)
        except Exception as e:
            pred = f"ERROR: {e}"
        try:
            v = eval_once(client, EVAL_MODEL, question, gold_list, pred)
            verdict = v.get("verdict", "unknown")
            reason = v.get("reason", "")
        except Exception as e:
            verdict, reason = "unknown", repr(e)[:200]
        row = {"id": qid, "question": question, "gold": gold_list,
               "pred": pred, "verdict": verdict, "reason": reason,
               "method": "closedbook"}
        rows.append(row)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done.add(qid)
        if (i + 1) % 10 == 0:
            correct = sum(1 for r in rows if r["verdict"] == "correct")
            print(f"  [closedbook] {i+1}/{len(qa_subset)} "
                  f"acc={correct/len(rows):.3f} elapsed={time.time()-t0:.0f}s")

    correct = sum(1 for r in rows if r["verdict"] == "correct")
    abstain = sum(1 for r in rows if is_abstain(r["pred"]))
    print(f"\nDONE. n={len(rows)} acc={correct/len(rows):.3f} abstain={abstain/len(rows):.3f}")


if __name__ == "__main__":
    main()
