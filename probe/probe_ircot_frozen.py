"""
IRCoT-frozen: Interleaved Retrieval Chain-of-Thought operating on a frozen
candidate pool of chunks already present in the dumped Graph-RAG context.

Design rationale
----------------
Faithful IRCoT (Trivedi et al., ACL'23) issues live retrieval calls during
reasoning. Reimplementing live retrieval against KET-RAG and LightRAG indices
takes 1-2 weeks of plumbing in this harness. The frozen-pool approximation
holds the underlying retrieval fixed (we use whatever chunks the base system
already retrieved into the dumped context) and only varies the reasoning loop.

This is the right experimental contrast for the headline claim:
    "Reasoning-side methods produce different gain patterns on retrieval-side
     SOTA bases due to retrieval-pipeline structure."

If the asymmetry shows up here, it cannot be attributed to differences in
live-retrieval interfaces — only to how iteration interacts with what each
base already surfaced.

Interface
---------
answer_with_ircot_frozen(client, model, question, context, embedder,
                         max_iters=5, top_k_init=8, top_k_step=3,
                         max_tokens=512, temperature=0.0)
    -> (answer, trace_dict)

trace_dict contains: n_iters, included_chunk_ids, sentences, final_context_chars
"""
import json
import re
import numpy as np

from qa_pipeline import (
    call_groq_chat,
    _parse_ket_context,
    _estimate_words,
)


# ── Pool extraction ───────────────────────────────────────────────

def extract_pool_from_ket(context: str) -> list:
    """Extract candidate chunks from a KET-RAG context dump.

    Returns: list of (chunk_id, text) tuples.
    """
    parsed = _parse_ket_context(context)
    pool = list(parsed["chunks"])
    # Treat sources (community reports) as additional candidates
    for sid, text in parsed.get("sources", []):
        pool.append((f"source_{sid}", text))
    return pool


def parse_lightrag_context_inline(context: str) -> dict:
    """Inline copy of lightrag_musique.parse_lightrag_context to avoid
    importing that module (which transitively imports groq)."""
    result = {"entities": {}, "relationships": [], "sources": [], "chunks": [], "adj": {}}
    if not context:
        return result
    blocks = re.findall(r'```json\s*\n(.*?)```', context, re.DOTALL)
    sections = re.split(r'```json\s*\n.*?```', context, flags=re.DOTALL)
    for i, block in enumerate(blocks):
        header = sections[i] if i < len(sections) else ""
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if "Entity" in header:
            for line in lines:
                try:
                    obj = json.loads(line)
                    name = (obj.get("entity") or obj.get("entity_name") or "").upper()
                    desc = obj.get("description", "")
                    if name:
                        result["entities"][name] = desc
                except json.JSONDecodeError:
                    continue
        elif "Relationship" in header:
            for line in lines:
                try:
                    obj = json.loads(line)
                    src = (obj.get("entity1") or obj.get("src_id") or "").upper()
                    tgt = (obj.get("entity2") or obj.get("tgt_id") or "").upper()
                    desc = obj.get("description", "")
                    weight = float(obj.get("weight", 1.0))
                    if src and tgt:
                        result["relationships"].append((src, tgt, desc, weight))
                        result["adj"].setdefault(src, set()).add(tgt)
                        result["adj"].setdefault(tgt, set()).add(src)
                except (json.JSONDecodeError, ValueError):
                    continue
        elif "Chunk" in header or "Document" in header:
            for j, line in enumerate(lines):
                try:
                    obj = json.loads(line)
                    cid = obj.get("reference_id", f"chunk_{j}")
                    content = obj.get("content", "")
                    if content:
                        result["chunks"].append((cid, content))
                except json.JSONDecodeError:
                    continue
    return result


def extract_pool_from_lightrag(context: str) -> list:
    """Extract candidate chunks from a LightRAG context dump."""
    parsed = parse_lightrag_context_inline(context)
    return list(parsed["chunks"])


# ── Entity-relationship summary (used as scaffold context) ────────

def _format_er_summary(parsed: dict, max_chars: int = 1500) -> str:
    """Build a short entity-relationship summary string from parsed context.

    Used as the always-included scaffold so the model has structural context
    even when no chunks have been retrieved yet.
    """
    parts = []
    n_ents = 0
    for name, desc in list(parsed.get("entities", {}).items())[:30]:
        line = f"{name}: {desc}" if desc else name
        parts.append(line)
        n_ents += 1
    parts.append("---")
    n_rels = 0
    for src, tgt, desc, _w in parsed.get("relationships", [])[:30]:
        parts.append(f"{src} -> {tgt}: {desc}")
        n_rels += 1
    out = "\n".join(parts)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n[…ER summary truncated]"
    return out


# ── IRCoT prompt ──────────────────────────────────────────────────

_IRCOT_SYSTEM = (
    "You are answering a multi-hop question step by step. "
    "At each step, write ONE next reasoning sentence based on the context so "
    "far. Do not skip ahead. Each sentence should establish a single fact or "
    "a single inference from the context. When you have enough information, "
    "write FINAL ANSWER: <answer in a few words>. If the answer is not in "
    "the context, write FINAL ANSWER: I don't know."
)


_IRCOT_TEMPLATE = (
    "QUESTION:\n{question}\n\n"
    "CONTEXT (entities, relationships, and retrieved chunks so far):\n"
    "{context}\n\n"
    "REASONING SO FAR:\n{trace}\n\n"
    "Write the NEXT one sentence of reasoning. If you can answer now, write "
    "FINAL ANSWER: <answer>."
)


# ── Embedding helpers ──────────────────────────────────────────────

def _embed_one(embedder, text: str) -> np.ndarray:
    v = embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return v[0]


def _embed_pool(embedder, pool: list) -> np.ndarray:
    """Returns (N, D) normalized embeddings for the pool."""
    if not pool:
        return np.zeros((0, embedder.get_sentence_embedding_dimension()))
    texts = [t for _, t in pool]
    return embedder.encode(texts, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)


def _topk(query_vec: np.ndarray, pool_mat: np.ndarray, k: int,
          excluded: set, pool_ids: list) -> list:
    """Return top-k pool indices (by cosine sim, since vectors are normalized)
    excluding any already in `excluded`."""
    if pool_mat.shape[0] == 0:
        return []
    sims = pool_mat @ query_vec
    order = np.argsort(-sims)
    out = []
    for idx in order:
        cid = pool_ids[idx]
        if cid in excluded:
            continue
        out.append(int(idx))
        if len(out) >= k:
            break
    return out


# ── IRCoT-frozen main loop ────────────────────────────────────────

def answer_with_ircot_frozen(client, model, question, context, embedder,
                             max_iters: int = 5,
                             top_k_init: int = 8,
                             top_k_step: int = 3,
                             max_tokens: int = 256,
                             temperature: float = 0.0,
                             base: str = "ket"):
    """Run IRCoT with a frozen candidate pool extracted from the dumped context.

    Args:
        base: "ket" or "lightrag" — selects the appropriate parser.

    Returns:
        (final_answer, trace_dict)
    """
    # 1. Build the frozen pool + ER scaffold
    if base == "ket":
        parsed = _parse_ket_context(context)
        pool = list(parsed["chunks"])
        for sid, text in parsed.get("sources", []):
            pool.append((f"source_{sid}", text))
    elif base == "lightrag":
        parsed = parse_lightrag_context_inline(context)
        pool = list(parsed["chunks"])
    else:
        raise ValueError(f"Unknown base: {base}")

    er_summary = _format_er_summary(parsed)

    if not pool:
        # Degenerate: no chunks to retrieve from. Single-shot fallback.
        prompt = _IRCOT_TEMPLATE.format(
            question=question,
            context=er_summary or "(no context available)",
            trace="(no reasoning yet)",
        )
        raw = call_groq_chat(client, model,
                             [{"role": "system", "content": _IRCOT_SYSTEM},
                              {"role": "user", "content": prompt}],
                             max_tokens=max_tokens, temperature=temperature)
        answer = _extract_final_answer(raw)
        return answer, {
            "n_iters": 0,
            "included_chunk_ids": [],
            "sentences": [raw],
            "final_context_chars": len(er_summary),
            "early_stop": "no_pool",
        }

    # 2. Embed pool + question
    pool_ids = [cid for cid, _ in pool]
    pool_mat = _embed_pool(embedder, pool)
    q_vec = _embed_one(embedder, question)

    # 3. Initial retrieval (top_k_init by question similarity)
    init_idxs = _topk(q_vec, pool_mat, top_k_init, set(), pool_ids)
    included = set(pool_ids[i] for i in init_idxs)

    def _build_context():
        chunk_blocks = []
        for cid, text in pool:
            if cid in included:
                chunk_blocks.append(f"[{cid}] {text}")
        chunks_str = "\n\n".join(chunk_blocks)
        return f"{er_summary}\n\n--- Retrieved chunks ---\n{chunks_str}"

    # 4. Iterate
    sentences = []
    early_stop = None
    for it in range(max_iters):
        ctx = _build_context()
        trace = "\n".join(f"- {s}" for s in sentences) if sentences else "(start)"
        prompt = _IRCOT_TEMPLATE.format(
            question=question, context=ctx, trace=trace
        )
        raw = call_groq_chat(client, model,
                             [{"role": "system", "content": _IRCOT_SYSTEM},
                              {"role": "user", "content": prompt}],
                             max_tokens=max_tokens, temperature=temperature)
        new_sent = raw.strip().split("\n")[0].strip()
        sentences.append(raw.strip())

        # Check for final answer
        if re.search(r'(?i)FINAL\s*ANSWER\s*:', raw):
            early_stop = "final_answer"
            break

        # Retrieve more based on the new sentence
        s_vec = _embed_one(embedder, new_sent)
        new_idxs = _topk(s_vec, pool_mat, top_k_step, included, pool_ids)
        if not new_idxs:
            # Pool exhausted — one more shot to extract answer
            sentences.append("(pool exhausted; answer if possible)")
            ctx = _build_context()
            trace = "\n".join(f"- {s}" for s in sentences)
            prompt = _IRCOT_TEMPLATE.format(
                question=question, context=ctx, trace=trace
            )
            raw = call_groq_chat(client, model,
                                 [{"role": "system", "content": _IRCOT_SYSTEM},
                                  {"role": "user", "content": prompt}],
                                 max_tokens=max_tokens, temperature=temperature)
            sentences.append(raw.strip())
            early_stop = "pool_exhausted"
            break
        for idx in new_idxs:
            included.add(pool_ids[idx])
    else:
        early_stop = "max_iters"

    # 5. Dedicated short-answer extraction step.
    # Even when the trace contains a "FINAL ANSWER:" line, the iteration prompt
    # tends to produce verbose sentences instead of a clean short answer phrase.
    # We do one more focused call that takes the FULL trace and extracts the
    # answer in 1-5 words, matching how the LLM judge scores predictions.
    answer = _short_extract(client, model, question, sentences,
                            temperature=temperature)

    return answer, {
        "n_iters": len(sentences),
        "included_chunk_ids": sorted(included),
        "sentences": sentences,
        "final_context_chars": sum(len(t) for cid, t in pool if cid in included),
        "early_stop": early_stop,
    }


_EXTRACT_SYSTEM = (
    "You extract short final answers from a reasoning trace. "
    "Output ONLY the answer, in 1-5 words, with no preamble. "
    "If the trace does not contain enough information, output exactly: "
    "I don't know"
)


def _short_extract(client, model, question, sentences, temperature: float = 0.0) -> str:
    """Take the IRCoT reasoning trace and extract a short answer phrase.

    Mirrors how SPARQL-CoT produces a clean noun-phrase answer; without this
    step IRCoT outputs are verbose sentences that the LLM judge reliably
    rejects even when the right entity is present.
    """
    trace_text = "\n".join(f"- {s}" for s in sentences) if sentences else "(empty)"
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"REASONING TRACE:\n{trace_text}\n\n"
        "Output the final answer to the question in 1-5 words. "
        "If the trace doesn't contain the answer, output: I don't know\n"
        "ANSWER:"
    )
    raw = call_groq_chat(client, model,
                         [{"role": "system", "content": _EXTRACT_SYSTEM},
                          {"role": "user", "content": prompt}],
                         max_tokens=20, temperature=temperature)
    # Strip a leading "ANSWER:" or "Final answer:" if model echoed it.
    answer = re.sub(r"^\s*(?:final\s*)?answer\s*:\s*", "", raw, flags=re.IGNORECASE).strip()
    # Take only the first line.
    answer = answer.split("\n")[0].strip()
    # Strip leading/trailing quotes.
    answer = answer.strip('"\'')
    return answer or "I don't know"


def _extract_final_answer(text: str) -> str:
    """Same regex pattern as answer_with_sparql_cot."""
    matches = re.findall(r'(?i)FINAL\s*ANSWER\s*:\s*(.+)', text or "")
    if matches:
        candidate = matches[-1].strip().split("\n")[0].strip()
        if candidate and not candidate.startswith("<"):
            return candidate
    return text.strip().split("\n")[0].strip() if text else ""
