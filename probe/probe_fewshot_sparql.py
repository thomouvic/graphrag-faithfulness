"""
Few-shot SPARQL-CoT.

Augments Paper 1's SPARQL-CoT prompt with 3 worked examples covering the
three multi-hop question types (single-bridge, comparison, multi-hop-bridge).
The examples are hand-crafted from generic world knowledge — NONE come from
the HotpotQA / MuSiQue / 2WikiMHQA evaluation samples — so there is no
contamination of the held-out test sets.

Cost vs single-shot SPARQL: 1× LLM call per question, with a longer prompt
(~600 extra tokens of context per call). Llama-3.1-8B has 128k context so
this is well within bounds.

Compatible with self-consistency: pass `n_samples` and `temperature` and
the function will sample multiple completions like answer_with_sparql_cot_sc.
"""
import re
from qa_pipeline import call_groq_chat, normalize_answer, is_abstain


# Hand-crafted worked examples. Style mirrors Paper 1's SPARQL-CoT format.
# Source domain: generic world knowledge; none of these are in the HotpotQA,
# MuSiQue, or 2WikiMHQA evaluation pools used in this project.
_FEW_SHOT_EXAMPLES = """Example 1 (single bridge):
CONTEXT:
Forrest Gump (1994) is an American comedy-drama film directed by Robert
Zemeckis and starring Tom Hanks. The film won the Academy Award for Best
Picture at the 67th Academy Awards in 1995.
QUESTION:
Which film, directed by Robert Zemeckis, won the Academy Award for Best
Picture at the 67th Academy Awards?
Step 1: Write a simple SPARQL query (max 4 triple patterns, plain English
predicates, NO URIs, NO FILTER, NO subqueries).
SELECT ?film WHERE {
  ?film directedBy "Robert Zemeckis" .
  ?film wonAward "Academy Award for Best Picture" .
  ?film atCeremony "67th Academy Awards" .
}
Step 2: Trace each variable through the context.
From context: Forrest Gump directedBy Robert Zemeckis, wonAward Best
Picture, atCeremony 67th Academy Awards. So ?film = Forrest Gump.
Step 3: FINAL ANSWER: Forrest Gump

Example 2 (comparison):
CONTEXT:
Boston is a city in the U.S. state of Massachusetts; it was founded in
1630 by Puritan settlers. Philadelphia is the largest city in the
Commonwealth of Pennsylvania, founded in 1682 by William Penn.
QUESTION:
Which city was founded earlier, Boston or Philadelphia?
Step 1: Write a simple SPARQL query.
SELECT ?answer WHERE {
  "Boston" foundedIn ?yearB .
  "Philadelphia" foundedIn ?yearP .
  ?answer earlierOf ?yearB ?yearP .
}
Step 2: Trace.
Boston foundedIn 1630. Philadelphia foundedIn 1682. 1630 < 1682, so
Boston was founded earlier.
Step 3: FINAL ANSWER: Boston

Example 3 (multi-hop bridge):
CONTEXT:
Toyota Motor Corporation is headquartered in Toyota City, located in
Aichi Prefecture, Japan. The capital of Japan is Tokyo, located on the
island of Honshu.
QUESTION:
What is the capital of the country where Toyota Motor Corporation is
headquartered?
Step 1: Write a simple SPARQL query.
SELECT ?capital WHERE {
  "Toyota Motor Corporation" headquarteredIn ?city .
  ?city locatedIn ?country .
  ?country hasCapital ?capital .
}
Step 2: Trace.
Toyota headquarteredIn Toyota City, which is locatedIn Aichi, Japan.
Country = Japan. Japan hasCapital Tokyo.
Step 3: FINAL ANSWER: Tokyo

"""


def answer_with_few_shot_sparql_cot(client, model, question, context,
                                    max_tokens: int = 512,
                                    temperature: float = 0.0):
    """Single-shot SPARQL-CoT with 3 worked examples in the prompt."""
    prompt = (
        "You are answering a multi-hop question using ONLY the provided context.\n"
        "First, study these examples of how to answer correctly.\n\n"
        f"{_FEW_SHOT_EXAMPLES}"
        "Now answer the following question in the SAME format "
        "(Step 1 SPARQL → Step 2 trace → Step 3 FINAL ANSWER).\n"
        "If the answer is not in the context, write FINAL ANSWER: I don't know\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )
    raw = call_groq_chat(client, model,
                         [{"role": "user", "content": prompt}],
                         max_tokens=max_tokens, temperature=temperature)
    answer = raw
    matches = re.findall(r'(?i)FINAL\s*ANSWER\s*:\s*(.+)', raw)
    if matches:
        candidate = matches[-1].strip().split("\n")[0].strip()
        if candidate and not candidate.startswith("<"):
            answer = candidate
    return answer, raw


def answer_with_few_shot_sparql_cot_sc(client, model, question, context,
                                       k: int = 3,
                                       temperature: float = 0.7,
                                       max_tokens: int = 512):
    """Few-shot SPARQL-CoT + self-consistency (composition).

    Same vote-tally as probe_sparql_sc.answer_with_sparql_cot_sc, but each
    sample uses the few-shot variant of the prompt.
    """
    from collections import Counter
    samples = []
    for i in range(k):
        try:
            ans, raw = answer_with_few_shot_sparql_cot(
                client, model, question, context,
                temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as e:
            ans, raw = f"ERROR: {e}", str(e)
        samples.append({"sample_idx": i, "answer": ans, "raw": raw,
                        "normalized": normalize_answer(ans)})

    # Same vote logic as SC: drop errors/abstentions, majority on normalized,
    # tie-break by first sample in top group.
    valid = [s for s in samples
             if not s["answer"].startswith("ERROR:")
             and not is_abstain(s["answer"])]
    if not valid:
        for s in samples:
            if not s["answer"].startswith("ERROR:"):
                return s["answer"], {"k": k, "temperature": temperature, "samples": samples}
        return "I don't know", {"k": k, "temperature": temperature, "samples": samples}

    counts = Counter(s["normalized"] for s in valid)
    top_count = counts.most_common(1)[0][1]
    top_norms = [n for n, c in counts.items() if c == top_count]
    for s in valid:
        if s["normalized"] in top_norms:
            return s["answer"], {"k": k, "temperature": temperature, "samples": samples}
    return valid[0]["answer"], {"k": k, "temperature": temperature, "samples": samples}
