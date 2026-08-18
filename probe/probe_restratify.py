"""
BM25 re-stratification of LightRAG context.

The mechanism claim from §5 of the paper is that LightRAG's uniformly-relevant
hybrid retrieval breaks structure-exploiting augmentations (GW especially).
This module artificially injects a relevance gradient into LightRAG's flat
chunk pool by BM25-scoring chunks against the question and keeping only the
top-k. Downstream methods (FS, SC, GW) then operate on the re-stratified
context.

Test: if FS+SC+GW on re-stratified LightRAG context beats FS+SC alone, the
mechanism is engineerable, not just observable.

Public API:
    restratify_lightrag_context(question, context, top_k=10) -> str
"""
import json
import re
from rank_bm25 import BM25Okapi


# Matches the "Document Chunks (...): \n```json\n...\n```" block in a
# LightRAG context dump.
_CHUNKS_BLOCK_RE = re.compile(
    r"(?P<header>(?:Document Chunks|Knowledge Graph Data \(Chunk\))[^\n]*\n+)"
    r"```json\n"
    r"(?P<body>.*?)\n"
    r"```",
    re.DOTALL,
)


def _tokenize(text: str) -> list:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def restratify_lightrag_context(question: str, context: str,
                                top_k: int = 10) -> str:
    """Re-rank LightRAG chunks by BM25 against the question and keep top_k.

    Returns the modified context string with the chunk block trimmed to
    the top-k chunks (in BM25-ranked order, highest first). Entity and
    relationship sections are preserved verbatim.

    If the context has fewer than `top_k` chunks, returns it unchanged.
    If no chunks block is found, returns the context unchanged.
    """
    m = _CHUNKS_BLOCK_RE.search(context)
    if not m:
        return context

    body = m.group("body")
    chunk_lines = [l for l in body.strip().split("\n") if l.strip()]

    if len(chunk_lines) <= top_k:
        # Nothing to filter — fewer chunks than requested
        return context

    # Parse each chunk JSON to extract content
    chunks = []
    for line in chunk_lines:
        try:
            obj = json.loads(line)
            content = obj.get("content", "")
            chunks.append((line, content))
        except json.JSONDecodeError:
            # Keep malformed lines at the front in original order;
            # they bias toward "keep" since BM25 score 0 < something
            chunks.append((line, ""))

    # BM25-score each chunk's content against the question
    tokenized_chunks = [_tokenize(content) for _, content in chunks]
    tokenized_question = _tokenize(question)

    if not tokenized_question or not any(tokenized_chunks):
        return context

    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenized_question)

    # Pick top-k indices by score (descending)
    ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])[:top_k]

    kept_lines = [chunks[i][0] for i in ranked]
    new_body = "\n".join(kept_lines)

    new_block = m.group("header") + "```json\n" + new_body + "\n```"
    new_context = context[:m.start()] + new_block + context[m.end():]
    return new_context


def build_restratified_lookup(qa_list, base_lookup, top_k: int = 10):
    """Apply restratification to every LightRAG context in a lookup.

    Returns: (new_lookup, stats)
    """
    out = {}
    n_modified = n_unchanged = 0
    orig_total = new_total = 0
    for q in qa_list:
        qid = str(q["id"])
        ctx = base_lookup.get(qid, "")
        if not ctx.strip():
            out[qid] = ""
            continue
        new_ctx = restratify_lightrag_context(q["question"], ctx, top_k=top_k)
        out[qid] = new_ctx
        orig_total += len(ctx)
        new_total += len(new_ctx)
        if new_ctx != ctx:
            n_modified += 1
        else:
            n_unchanged += 1
    return out, {
        "n_modified": n_modified,
        "n_unchanged": n_unchanged,
        "avg_orig_chars": orig_total / max(len(qa_list), 1),
        "avg_new_chars": new_total / max(len(qa_list), 1),
        "top_k": top_k,
    }
