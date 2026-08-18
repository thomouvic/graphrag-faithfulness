"""
In-domain few-shot SPARQL-CoT answering.

Identical prompt scaffold and answer-parsing to probe_fewshot_sparql, except
the worked-example block is supplied per question by a selector from
probe_indomain_examples (generic / domain / train / dynamic / structure).
Holding everything else fixed, the *only* variable across the improvement
gradient is which examples appear in the prompt — so any accuracy delta is
attributable to example selection, not to prompt or decoding changes.

Cost is unchanged: 1 LLM call/question (single-shot) or k calls (SC). Selecting
the examples is LLM-free.
"""
import re
from collections import Counter

from qa_pipeline import call_groq_chat, normalize_answer, is_abstain


def _build_prompt(example_block: str, question: str, context: str) -> str:
    return (
        "You are answering a multi-hop question using ONLY the provided context.\n"
        "First, study these examples of how to answer correctly.\n\n"
        f"{example_block}"
        "Now answer the following question in the SAME format "
        "(Step 1 SPARQL → Step 2 trace → Step 3 FINAL ANSWER).\n"
        "If the answer is not in the context, write FINAL ANSWER: I don't know\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )


def _parse_final(raw: str) -> str:
    answer = raw
    matches = re.findall(r'(?i)FINAL\s*ANSWER\s*:\s*(.+)', raw)
    if matches:
        candidate = matches[-1].strip().split("\n")[0].strip()
        if candidate and not candidate.startswith("<"):
            answer = candidate
    return answer


def answer_with_indomain_fewshot(client, model, question, context,
                                 example_block: str,
                                 max_tokens: int = 512,
                                 temperature: float = 0.0):
    """Single-shot in-domain few-shot SPARQL-CoT."""
    prompt = _build_prompt(example_block, question, context)
    raw = call_groq_chat(client, model,
                         [{"role": "user", "content": prompt}],
                         max_tokens=max_tokens, temperature=temperature)
    return _parse_final(raw), raw


def answer_with_indomain_fewshot_sc(client, model, question, context,
                                    example_block: str,
                                    k: int = 3,
                                    temperature: float = 0.7,
                                    max_tokens: int = 512):
    """In-domain few-shot + self-consistency (same vote logic as the others)."""
    samples = []
    for i in range(k):
        try:
            ans, raw = answer_with_indomain_fewshot(
                client, model, question, context, example_block,
                temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as e:
            ans, raw = f"ERROR: {e}", str(e)
        samples.append({"sample_idx": i, "answer": ans, "raw": raw,
                        "normalized": normalize_answer(ans)})

    trace = {"k": k, "temperature": temperature, "samples": samples}
    valid = [s for s in samples
             if not s["answer"].startswith("ERROR:")
             and not is_abstain(s["answer"])]
    if not valid:
        for s in samples:
            if not s["answer"].startswith("ERROR:"):
                return s["answer"], trace
        return "I don't know", trace

    counts = Counter(s["normalized"] for s in valid)
    top_count = counts.most_common(1)[0][1]
    top_norms = [n for n, c in counts.items() if c == top_count]
    for s in valid:
        if s["normalized"] in top_norms:
            return s["answer"], trace
    return valid[0]["answer"], trace
