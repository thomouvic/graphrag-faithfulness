"""
Graph-walk compression utilities — extracted/copied from the parent harness
so they can run without depending on lightrag_musique (which imports `groq`
and `lightrag` at top level).

For KET-RAG: reuses qa_pipeline.compress_context_graph_walk directly.
For LightRAG: inlines compress_lightrag_context from lightrag_musique.py.
"""
import re
from collections import deque

from qa_pipeline import (
    _bfs_entities, _expand_via_text, _match_question_entities, _estimate_words,
    compress_context_graph_walk as _compress_ket,
)
from probe_ircot_frozen import parse_lightrag_context_inline


def compress_ket(question: str, context: str,
                 max_hops: int = 3, budget_tokens: int = 4000):
    """KET-RAG graph-walk compression. Pass-through to qa_pipeline."""
    return _compress_ket(question, context, max_hops=max_hops,
                         budget_tokens=budget_tokens)


def compress_lightrag(question: str, context: str,
                      max_hops: int = 3, budget_tokens: int = 4000):
    """Inline copy of lightrag_musique.compress_lightrag_context.

    Avoids importing lightrag_musique (which imports groq + lightrag at
    top level). Behaviour identical to the parent.
    """
    parsed = parse_lightrag_context_inline(context)
    orig_words = _estimate_words(context)

    if not parsed["entities"]:
        return context, {"mode": "no_entities", "orig_words": orig_words,
                         "new_words": orig_words, "seeds": 0, "chain": 0}

    seeds = _match_question_entities(question, list(parsed["entities"].keys()))
    if not seeds:
        return context, {"mode": "no_seeds", "orig_words": orig_words,
                         "new_words": orig_words, "seeds": 0, "chain": 0}

    chain = _bfs_entities(parsed["adj"], seeds, max_hops)
    text_expanded = _expand_via_text(chain, set(parsed["entities"].keys()),
                                     parsed["chunks"])
    chain.update(text_expanded)
    chain_set = set(chain.keys())

    budget_words = int(budget_tokens / 1.3)
    max_hop_seen = max(chain.values()) if chain else 0

    hop_entities = {}
    for ent, hop in chain.items():
        hop_entities.setdefault(hop, []).append(ent)
    for h in hop_entities:
        hop_entities[h].sort()

    rel_by_src = {}
    for src, tgt, desc, _ in parsed["relationships"]:
        if src in chain_set and tgt in chain_set:
            rel_by_src.setdefault(src, []).append((tgt, desc))

    q_content = {w for w in re.findall(r"[a-z0-9]+", (question or "").lower())
                 if len(w) > 2}

    chunk_hop_assignment, tier2 = {}, []
    chunk_map = {cid: text for cid, text in parsed["chunks"]}
    for cid, text in parsed["chunks"]:
        text_low = text.lower()
        best_hop, chain_score = None, 0
        for e in chain_set:
            if e.lower() in text_low:
                chain_score += 1
                h = chain[e]
                if best_hop is None or h < best_hop:
                    best_hop = h
        if chain_score > 0:
            chunk_hop_assignment[cid] = (best_hop, chain_score)
        else:
            chunk_words = set(re.findall(r"[a-z0-9]+", text_low))
            kw_score = len(q_content & chunk_words)
            if kw_score > 0:
                tier2.append((kw_score, cid, text))
    tier2.sort(key=lambda x: -x[0])

    parts, used_words, used_cids = [], 0, set()
    chunk_parts_count = rel_lines_count = 0

    for hop in range(max_hop_seen + 1):
        ents_at_hop = hop_entities.get(hop, [])
        if not ents_at_hop:
            continue
        hop_lines = []
        hop_label = "from question" if hop == 0 else f"hop {hop}"
        hop_lines.append(f"=== Step {hop} ({hop_label}) ===")
        for ent in ents_at_hop:
            desc = parsed["entities"].get(ent, "")
            hop_lines.append(f"  {ent}: {desc}" if desc else f"  {ent}")
            for tgt, rdesc in rel_by_src.get(ent, []):
                hop_lines.append(f"    -> {tgt}: {rdesc}")
                rel_lines_count += 1
        hop_chunks = [(cid, chunk_hop_assignment[cid][1])
                      for cid in chunk_hop_assignment
                      if chunk_hop_assignment[cid][0] == hop and cid not in used_cids]
        hop_chunks.sort(key=lambda x: -x[1])
        hop_block = "\n".join(hop_lines)
        parts.append(hop_block)
        used_words += _estimate_words(hop_block)
        if used_words < budget_words:
            for cid, score in hop_chunks:
                text = chunk_map.get(cid, "")
                w = _estimate_words(text)
                if used_words + w > budget_words:
                    continue
                parts.append(f"  [{cid}] {text}")
                used_words += w
                used_cids.add(cid)
                chunk_parts_count += 1

    if tier2:
        t2_parts = []
        for score, cid, text in tier2:
            if cid in used_cids:
                continue
            w = _estimate_words(text)
            if used_words + w > budget_words:
                continue
            t2_parts.append(f"  [{cid}] {text}")
            used_words += w
            used_cids.add(cid)
            chunk_parts_count += 1
        if t2_parts:
            parts.append("=== Additional context ===")
            parts.extend(t2_parts)

    current = "\n\n".join(parts)
    return current, {
        "mode": "compressed", "orig_words": orig_words,
        "new_words": _estimate_words(current),
        "seeds": len(seeds), "chain": len(chain),
        "n_chunks_kept": chunk_parts_count, "n_rels_kept": rel_lines_count,
    }


def build_gw_lookup(qa_list, base_lookup, base: str,
                    max_hops: int = 3, budget_tokens: int = 4000):
    """Apply GW compression to every context in `base_lookup`.

    base: "ket" or "lightrag" — selects the appropriate compressor.
    Returns: (new_lookup, stats)
    """
    fn = compress_ket if base == "ket" else compress_lightrag
    out = {}
    n_compressed = n_fallback = 0
    orig_total = new_total = 0
    for q in qa_list:
        qid = str(q["id"])
        ctx = base_lookup.get(qid, "")
        if not ctx.strip():
            out[qid] = ""
            continue
        new, meta = fn(q["question"], ctx, max_hops=max_hops,
                       budget_tokens=budget_tokens)
        out[qid] = new
        orig_total += meta.get("orig_words", 0)
        new_total += meta.get("new_words", 0)
        if meta.get("mode") == "compressed":
            n_compressed += 1
        else:
            n_fallback += 1
    return out, {
        "n_compressed": n_compressed, "n_fallback": n_fallback,
        "avg_orig_words": orig_total / max(len(qa_list), 1),
        "avg_new_words": new_total / max(len(qa_list), 1),
    }
