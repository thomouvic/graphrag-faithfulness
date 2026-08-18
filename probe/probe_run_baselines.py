"""
1×-cost-tier baselines for the example-source headline table (plan.md §3).

Runs two single-call baselines on the same KET-RAG and LightRAG cells / paired
questions used by the few-shot variants, so the headline claim — "FS-best beats
Self-Ask and generic CoT at 1× cost" — can be stated with paired-bootstrap CIs:

  * generic_cot : Paper 1's plain chain-of-thought prompt (qa_pipeline).
  * self_ask    : Self-Ask-style prompting (Press et al. 2023) over the FIXED
                  retrieved context — single call, no re-retrieval (frozen,
                  matching this repo's IRCoT-frozen comparison methodology).

Both are 1 LLM call/question. Output JSONL mirrors probe_run_fewshot's rows so
probe_gradient_analyze / probe_analyze can read them.

Usage:
    python3 -u probe_run_baselines.py \
        --qa-pairs experiments/musique/large_scale/qa-pairs/qa-pairs.json \
        --ket-context experiments/musique/large_scale/output/large_scale-keyword-0.5.json \
        --lightrag-context experiments/musique/large_scale_lightrag/lightrag_contexts.json \
        --limit 500 --methods generic_cot self_ask --out-dir probe_results/
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from qa_pipeline import call_groq_chat, eval_once
from probe_compute_subset import (
    load_ket_contexts, load_lightrag_contexts, _gold_strings,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
QA_MODEL = "meta-llama/llama-3.1-8b-instruct"
EVAL_MODEL = "meta-llama/llama-3.1-8b-instruct"


def _parse_final(raw: str) -> str:
    m = re.findall(r"(?i)(?:final answer|so the final answer is)\s*:?\s*(.+)", raw or "")
    if m:
        cand = m[-1].strip().split("\n")[0].strip(" .")
        if cand and not cand.startswith("<"):
            return cand
    return raw


def answer_generic_cot(client, model, question, context, temperature=0.0):
    """Try qa_pipeline's generic CoT; fall back to an inline equivalent prompt."""
    try:
        from qa_pipeline import answer_with_generic_cot  # Paper 1's prompt
        return answer_with_generic_cot(client, model, question, context,
                                       temperature=temperature)
    except Exception:
        prompt = (
            "Answer the multi-hop question using ONLY the provided context.\n"
            "Think step by step, then give the answer.\n"
            "End with a line: FINAL ANSWER: <answer>\n"
            "If the answer is not in the context, write FINAL ANSWER: I don't know\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        )
        raw = call_groq_chat(client, model, [{"role": "user", "content": prompt}],
                             max_tokens=512, temperature=temperature)
        return _parse_final(raw), raw


_SELF_ASK_INSTR = (
    "Answer the question using ONLY the provided context, in the Self-Ask format.\n"
    "Decide whether follow-up questions are needed. For each follow-up, write\n"
    "'Follow up:' then 'Intermediate answer:' grounded in the context. Then write\n"
    "'So the final answer is: <answer>'. Use only the context; if the answer is\n"
    "not present, the final answer is: I don't know\n\n"
    "Question: When was the director of the film Pulp Fiction born?\n"
    "Are follow up questions needed here: Yes.\n"
    "Follow up: Who directed Pulp Fiction?\n"
    "Intermediate answer: Quentin Tarantino.\n"
    "Follow up: When was Quentin Tarantino born?\n"
    "Intermediate answer: March 27, 1963.\n"
    "So the final answer is: March 27, 1963\n\n"
)


def answer_self_ask(client, model, question, context, temperature=0.0):
    """Single-call Self-Ask over the fixed context (frozen — no re-retrieval)."""
    prompt = (
        f"{_SELF_ASK_INSTR}"
        f"CONTEXT:\n{context}\n\n"
        f"Question: {question}\n"
        "Are follow up questions needed here:"
    )
    raw = call_groq_chat(client, model, [{"role": "user", "content": prompt}],
                         max_tokens=512, temperature=temperature)
    return _parse_final(raw), raw


_METHODS = {"generic_cot": answer_generic_cot, "self_ask": answer_self_ask}


def _eval(client, q, gold, pred):
    try:
        v = eval_once(client, EVAL_MODEL, q, gold, pred)
        return v.get("verdict", "unknown"), v.get("reason", "")
    except Exception as e:
        return "unknown", repr(e)[:200]


def _run_cell(client, qa_subset, ctx, base, method, out_path):
    fn = _METHODS[method]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done, rows = set(), []
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); rows.append(r); done.add(str(r["id"]))
        print(f"  [resume] {base}+{method}: {len(done)} done")
    t0, n = time.time(), len(qa_subset)
    for i, q in enumerate(qa_subset):
        qid = str(q["id"])
        if qid in done:
            continue
        question, gold_list = q["question"], _gold_strings(q)
        c = ctx.get(qid, "")
        try:
            if not c.strip():
                pred, raw = "I don't know", ""
            else:
                pred, raw = fn(client, QA_MODEL, question, c)
        except Exception as e:
            pred, raw = f"ERROR: {e}", repr(e)[:200]
        verdict, reason = _eval(client, question, gold_list, pred)
        row = {"id": qid, "question": question, "gold": gold_list,
               "pred": pred, "verdict": verdict, "reason": reason,
               "base": base, "method": method, "trace": {"raw": (raw or "")[:500]}}
        rows.append(row); done.add(qid)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (i + 1) % 10 == 0 or (i + 1) == n:
            correct = sum(1 for r in rows if r["verdict"] == "correct")
            print(f"  [{base}+{method}] {i+1}/{n} acc={correct/len(rows):.3f} "
                  f"elapsed={time.time()-t0:.0f}s")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa-pairs", required=True, type=Path)
    p.add_argument("--ket-context", required=True, type=Path)
    p.add_argument("--lightrag-context", required=True, type=Path)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--methods", nargs="+", default=["generic_cot", "self_ask"],
                   choices=list(_METHODS.keys()))
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    load_dotenv(".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr); sys.exit(1)

    qa_subset = json.loads(args.qa_pairs.read_text(encoding="utf-8"))[:args.limit]
    print(f"Loaded {len(qa_subset)} questions")
    ket_ctx = load_ket_contexts(args.ket_context)
    lr_ctx = load_lightrag_contexts(args.lightrag_context)
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url=OPENROUTER_BASE_URL)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for base, ctx in [("ket", ket_ctx), ("lightrag", lr_ctx)]:
        for method in args.methods:
            out_path = args.out_dir / f"{base}_{method}.jsonl"
            print(f"\n=== Cell: {base} + {method} ===")
            rows = _run_cell(client, qa_subset, ctx, base, method, out_path)
            correct = sum(1 for r in rows if r["verdict"] == "correct")
            summary[f"{base}_{method}"] = {"n": len(rows),
                                           "acc": correct / max(len(rows), 1)}

    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    for k, v in summary.items():
        print(f"  {k:28s}  acc={v['acc']:.3f}  n={v['n']}")
    (args.out_dir / "summary_baselines.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
