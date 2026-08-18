"""
SPARQL-CoT with Self-Consistency (SC).

Mechanism: hold input fixed (same retrieved context, same SPARQL-CoT prompt
as Paper 1) and sample k independent generations at temperature > 0. Normalize
each prediction, majority-vote across samples. Returns the majority answer
plus the per-sample trace.

Target failure mode: covered-but-wrong cases where the answer is in context
but single-shot SPARQL-CoT produces noise. Self-consistency suppresses
sampling-noise errors; the correct answer (when retrievable) tends to be
more consistent across temperature-perturbed samples than any single wrong
candidate.

Contrast with IRCoT-frozen (which retrieves new chunks per iteration and
introduces new sources of confusion): SC introduces no new content, only
noise cancellation.

Usage from probe_run.py:
    from probe_sparql_sc import answer_with_sparql_cot_sc, run_sparql_sc
"""
import json
import re
import time
from collections import Counter

from qa_pipeline import (
    call_groq_chat, answer_with_sparql_cot, normalize_answer,
    is_abstain, get_gold,
)
from qa_pipeline import _load_checkpoint, _append_checkpoint
import pandas as pd


def answer_with_sparql_cot_sc(client, model, question, context,
                              k: int = 3,
                              temperature: float = 0.7,
                              max_tokens: int = 512):
    """Run SPARQL-CoT k times, majority-vote on normalized answers.

    Returns (final_answer, trace_dict).
    """
    samples = []
    for i in range(k):
        try:
            ans, raw = answer_with_sparql_cot(client, model, question, context,
                                              temperature=temperature,
                                              max_tokens=max_tokens)
        except Exception as e:
            ans, raw = f"ERROR: {e}", str(e)
        samples.append({"sample_idx": i, "answer": ans, "raw": raw,
                        "normalized": normalize_answer(ans)})

    return _vote(samples), {
        "k": k,
        "temperature": temperature,
        "samples": samples,
    }


def _vote(samples: list) -> str:
    """Majority vote on normalized answers, ignoring abstentions and errors.

    Tie-breaker: pick the answer from the first sample that produced it.
    """
    valid = []
    for s in samples:
        ans = s["answer"]
        if ans.startswith("ERROR:"):
            continue
        if is_abstain(ans):
            continue
        valid.append(s)
    if not valid:
        # All samples abstained or errored — fall back to first non-error
        for s in samples:
            if not s["answer"].startswith("ERROR:"):
                return s["answer"]
        return "I don't know"

    # Group by normalized answer
    norms = [s["normalized"] for s in valid]
    counts = Counter(norms)
    # Find max-vote group(s)
    top_count = counts.most_common(1)[0][1]
    top_norms = [n for n, c in counts.items() if c == top_count]
    # Tie-breaker: pick first sample in valid that belongs to one of the top
    for s in valid:
        if s["normalized"] in top_norms:
            return s["answer"]
    return valid[0]["answer"]


def run_sparql_sc(client, model, qa_list, context_lookup,
                  k: int = 3,
                  temperature: float = 0.7,
                  limit: int = None,
                  checkpoint_path=None):
    """Run SPARQL-CoT + SC on a list of QA pairs.

    Matches the signature of qa_pipeline.run_sparql so it slots into the
    same orchestration code.
    """
    rows, done_ids = _load_checkpoint(checkpoint_path)
    total = min(len(qa_list), limit) if limit else len(qa_list)
    t0 = time.time()
    for i, q in enumerate(qa_list):
        if limit is not None and i >= limit:
            break
        qid = str(q["id"])
        if qid in done_ids:
            continue
        question = q["question"]
        gold = get_gold(q)
        context = context_lookup.get(qid, "")
        if not context.strip():
            pred = "I don't know"
            samples = []
        else:
            pred, trace = answer_with_sparql_cot_sc(
                client, model, question, context, k=k, temperature=temperature
            )
            samples = trace["samples"]
        # Diversity stat: number of unique normalized answers across samples
        norms = [s["normalized"] for s in samples] if samples else []
        unique_norms = len(set(norms))
        row = {
            "qa_model": model, "id": qid, "question": question, "gold": gold,
            "context_chars": len(context),
            "sc_k": k, "sc_temperature": temperature,
            "sc_unique_answers": unique_norms,
            "final_pred": pred, "final_abstain": is_abstain(pred),
            "sc_samples": json.dumps(
                [{"i": s["sample_idx"], "n": s["normalized"]} for s in samples],
                ensure_ascii=False
            ),
        }
        rows.append(row)
        _append_checkpoint(checkpoint_path, row)
        elapsed = time.time() - t0
        q_safe = question[:50].encode("ascii", "replace").decode()
        if (i + 1) % 10 == 0 or (i + 1) == total:
            correct_so_far = "?"
            print(f"  [sparql_sc k={k}] {i+1}/{total}  elapsed={elapsed:.1f}s  q={q_safe}")
    return pd.DataFrame(rows)
