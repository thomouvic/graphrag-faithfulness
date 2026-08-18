"""
Programmatic SPARQL-trace verifier (idea #2 from ideas.md).

Self-consistency over SPARQL-CoT votes by surface-level answer agreement;
it has no signal for whether each sample's answer is *supported* by the
retrieved context. SPARQL-CoT's structured trace is uniquely amenable to
programmatic verification — we parse the trace, check the final answer
against the retrieved context, and drop samples whose answers are
unsupported from the vote.

Why SPARQL-CoT specifically: generic CoT outputs are free-form prose
that admits no clean check; SPARQL-CoT outputs a structured query plus a
final answer in known positions, both of which can be verified
programmatically.

Verifier (simple version): does the final answer phrase appear as a
substring in the retrieved context (after the same normalization the LLM
judge uses)? A sample that fails this check is almost certainly
hallucinated; we drop it from the SC vote.

Public API:
    answer_with_sparql_cot_sc_verified(client, model, question, context,
                                       k=3, temperature=0.7, max_tokens=512)
                                                          -> (final_answer, trace)
"""
import re
from collections import Counter

from qa_pipeline import call_groq_chat, normalize_answer, is_abstain
from probe_fewshot_sparql import answer_with_few_shot_sparql_cot


def verify_answer_grounded(answer: str, context: str) -> bool:
    """Return True if the (normalized) answer phrase appears as a substring
    in the (normalized) context. Skips empty answers, errors, and abstentions.
    """
    if not answer or answer.startswith("ERROR:"):
        return False
    if is_abstain(answer):
        return False
    a_norm = normalize_answer(answer).lower().strip()
    if not a_norm or a_norm == "i don't know":
        return False
    c_norm = normalize_answer(context).lower()
    # Substring match handles entity-name answers; for multi-token answers
    # we require the full phrase to appear (paraphrase-tolerant variants
    # are an obvious extension; held back for the first pass).
    return a_norm in c_norm


def answer_with_sparql_cot_sc_verified(client, model, question, context,
                                       k: int = 3,
                                       temperature: float = 0.7,
                                       max_tokens: int = 512):
    """Few-shot SPARQL-CoT + self-consistency + verifier-filtered voting.

    Same cost as SC (k LLM calls). After sampling, drop samples whose
    final answer is not grounded in the context, then majority-vote on the
    survivors. Falls back to ungrounded samples only if all k are
    ungrounded (preserves SC's recall guarantee).
    """
    samples = []
    for i in range(k):
        try:
            ans, raw = answer_with_few_shot_sparql_cot(
                client, model, question, context,
                temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as e:
            ans, raw = f"ERROR: {e}", str(e)
        samples.append({
            "sample_idx": i,
            "answer": ans,
            "raw": raw,
            "normalized": normalize_answer(ans),
            "grounded": verify_answer_grounded(ans, context),
        })

    grounded = [s for s in samples if s["grounded"]]
    voting_pool = grounded if grounded else [
        s for s in samples
        if not s["answer"].startswith("ERROR:") and not is_abstain(s["answer"])
    ]

    if not voting_pool:
        # Everything errored or abstained — return whatever we have
        for s in samples:
            if not s["answer"].startswith("ERROR:"):
                return s["answer"], {"k": k, "temperature": temperature,
                                     "samples": samples,
                                     "n_grounded": len(grounded),
                                     "voting_pool_size": 0,
                                     "fallback": True}
        return "I don't know", {"k": k, "temperature": temperature,
                                "samples": samples,
                                "n_grounded": 0,
                                "voting_pool_size": 0,
                                "fallback": True}

    counts = Counter(s["normalized"] for s in voting_pool)
    top_count = counts.most_common(1)[0][1]
    top_norms = {n for n, c in counts.items() if c == top_count}
    for s in voting_pool:
        if s["normalized"] in top_norms:
            return s["answer"], {
                "k": k, "temperature": temperature, "samples": samples,
                "n_grounded": len(grounded),
                "voting_pool_size": len(voting_pool),
                "fallback": not bool(grounded),
            }

    # Should be unreachable, but defensive
    return voting_pool[0]["answer"], {
        "k": k, "temperature": temperature, "samples": samples,
        "n_grounded": len(grounded),
        "voting_pool_size": len(voting_pool),
        "fallback": not bool(grounded),
    }
