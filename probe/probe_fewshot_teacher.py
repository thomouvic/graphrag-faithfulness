"""
Variant C: teacher-harvested few-shot example pool (plan.md §2.C, §4 Phase 2).

Pipeline:
  1. Run a stronger TEACHER model (default Llama-3.3-70B) with the variant-A
     few-shot SPARQL-CoT prompt on each benchmark's TRAIN split.
  2. LLM-judge each output against the gold answer; keep verdict=correct.
  3. Structural filter: clean Step-1 / Step-2 / Step-3 layout AND a SPARQL
     query with >= MIN_TRIPLES triple patterns AND a parseable FINAL ANSWER.
  4. Select up to --per-type examples of each question type (bridge/comparison),
     preferring the most triple-rich, to form a per-benchmark pool.
  5. Write a pool JSON consumable by probe_run_fewshot.py --example-source teacher.

The teacher's own Step-2 prose is kept VERBATIM (stored in each record's
`rendered` field), per plan.md §4 step 4 — we do not re-template it. Each
example's CONTEXT is the trimmed gold-supporting snippet (~300-400 tokens) by
default, or a provided retrieved-context file via --train-contexts.

Parsing/filter helpers (split_steps, count_triple_patterns, structural_ok,
extract_final) are import-light and unit-tested in tests/test_indomain.py.
Harness/API imports (qa_pipeline, openai) are lazy so this module imports
without the parent harness present.

Usage:
    python3 -u probe_fewshot_teacher.py \
        --benchmark musique \
        --train-split train_splits/musique_train.jsonl \
        --n-input 500 --per-type 3 \
        --teacher-model meta-llama/llama-3.3-70b-instruct \
        --out probe_results/teacher_pool_musique.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

MIN_TRIPLES = 3
TRIM_CHARS = 1600  # ~300-400 tokens


# --------------------------------------------------------------------------
# Import-light parsing / filtering helpers (no qa_pipeline / openai).
# --------------------------------------------------------------------------

def extract_final(raw: str) -> str:
    m = re.findall(r"(?i)FINAL\s*ANSWER\s*:\s*(.+)", raw or "")
    if not m:
        return ""
    cand = m[-1].strip().split("\n")[0].strip()
    return "" if cand.startswith("<") else cand


def split_steps(raw: str) -> dict:
    """Return {'step1','step2','step3'} text spans, or '' for missing ones."""
    raw = raw or ""
    out = {"step1": "", "step2": "", "step3": ""}
    marks = [("step1", r"(?i)step\s*1\s*:"),
             ("step2", r"(?i)step\s*2\s*:"),
             ("step3", r"(?i)step\s*3\s*:")]
    positions = []
    for key, pat in marks:
        m = re.search(pat, raw)
        positions.append((key, m.start() if m else None))
    for i, (key, pos) in enumerate(positions):
        if pos is None:
            continue
        end = None
        for _, npos in positions[i + 1:]:
            if npos is not None:
                end = npos
                break
        out[key] = raw[pos:end].strip() if end else raw[pos:].strip()
    return out


def count_triple_patterns(sparql_or_raw: str) -> int:
    """Count triple patterns inside the WHERE { ... } block.

    A triple pattern is a non-comment, non-brace line ending in '.' with at
    least three whitespace-separated tokens.
    """
    text = sparql_or_raw or ""
    m = re.search(r"WHERE\s*\{(.+?)\}", text, flags=re.S | re.I)
    body = m.group(1) if m else text
    # Triples end with '.', whether one-per-line or several on one line.
    n = 0
    for seg in body.split("."):
        seg = seg.strip()
        if not seg or seg.startswith("#"):
            continue
        if len(seg.split()) >= 3:
            n += 1
    return n


def min_triples_for(hops) -> int:
    """Hop-aware triple-count floor: a complete decomposition has one triple
    per hop, so a 2-hop question needs 2 (not 3). Cap at 3 (the quality bar
    for deep questions); floor at 2 so single-triple/trivial outputs are still
    rejected. hops=None falls back to the flat MIN_TRIPLES."""
    if hops is None:
        return MIN_TRIPLES
    return max(2, min(int(hops), MIN_TRIPLES))


def structural_ok(raw: str, hops=None) -> bool:
    """Clean 3-step structure + a SELECT/WHERE SPARQL with enough triples."""
    steps = split_steps(raw)
    if not (steps["step1"] and steps["step2"] and steps["step3"]):
        return False
    if not re.search(r"(?i)select\s.+where\s*\{", raw, flags=re.S):
        return False
    if count_triple_patterns(raw) < min_triples_for(hops):
        return False
    if not extract_final(raw):
        return False
    return True


def trim_context(text: str, max_chars: int = TRIM_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def build_rendered(raw: str, qtype: str, context: str) -> str:
    """Assemble a verbatim worked example from the teacher's raw output.

    Keeps the teacher's Step-1/2/3 prose unchanged; prepends the (trimmed)
    CONTEXT + QUESTION framing used by the few-shot prompt format.
    """
    steps = split_steps(raw)
    body = "\n".join(s for s in (steps["step1"], steps["step2"], steps["step3"]) if s)
    return f"CONTEXT:\n{context}\n{body}\n"


def balance_by_hop(pool: list, n: int) -> list:
    """Round-robin across hop counts so the first `n` records span depths.

    MuSiQue's train file is sorted by hop (all 2-hop, then 3-hop, then 4-hop),
    so a naive first-n truncation yields only 2-hop questions. This interleaves
    the hop buckets to give the teacher harvest balanced depth coverage."""
    buckets = {}
    for r in pool:
        buckets.setdefault(r.get("hops", 2), []).append(r)
    order = sorted(buckets)
    out, i = [], 0
    while len(out) < n and any(i < len(buckets[h]) for h in order):
        for h in order:
            if i < len(buckets[h]):
                out.append(buckets[h][i])
                if len(out) >= n:
                    break
        i += 1
    return out


def select_pool(candidates: list, per_type: int) -> list:
    """Pick up to per_type examples of each question type.

    Within a type, prefer HOP DIVERSITY (the richest example from each distinct
    hop depth first, so the pool spans 2/3/4-hop demonstrations), then fill the
    remaining slots with the next most triple-rich examples."""
    by_type = {}
    for c in sorted(candidates, key=lambda x: (-x["n_triples"], str(x.get("id", "")))):
        by_type.setdefault(c["type"], []).append(c)
    pool = []
    for _t, items in by_type.items():
        chosen, seen_hops = [], set()
        for c in items:                       # one richest per distinct hop
            h = c.get("hops")
            if h not in seen_hops:
                chosen.append(c); seen_hops.add(h)
            if len(chosen) >= per_type:
                break
        for c in items:                       # fill remaining slots
            if len(chosen) >= per_type:
                break
            if c not in chosen:
                chosen.append(c)
        pool.extend(chosen[:per_type])
    return pool


# --------------------------------------------------------------------------
# Harvest (needs qa_pipeline + openai + an API key).
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True,
                   choices=["musique", "hotpotqa", "2wikimultihopqa"])
    p.add_argument("--train-split", required=True, type=Path)
    p.add_argument("--n-input", type=int, default=500,
                   help="How many train questions to run the teacher on.")
    p.add_argument("--balance-hops", action="store_true",
                   help="Round-robin the train input across hop counts so the "
                        "harvested pool spans 2/3/4-hop (MuSiQue train is "
                        "hop-sorted, so a naive prefix is all 2-hop).")
    p.add_argument("--per-type", type=int, default=3,
                   help="Max examples to keep per question type.")
    p.add_argument("--teacher-model",
                   default="meta-llama/llama-3.3-70b-instruct")
    p.add_argument("--judge-model",
                   default="meta-llama/llama-3.1-8b-instruct")
    p.add_argument("--train-contexts", type=Path, default=None,
                   help="Optional {qid: context} JSON of retrieved contexts for "
                        "train questions. If omitted, the trimmed gold-supporting "
                        "snippet from the train record is used as the example "
                        "context (self-contained, no retrieval needed).")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    # lazy imports so the module is importable / testable without the harness
    import os
    from dotenv import load_dotenv
    from openai import OpenAI
    from qa_pipeline import eval_once
    from probe_fewshot_sparql import answer_with_few_shot_sparql_cot
    from probe_indomain_examples import load_train_pool, _gold_list
    from probe_compute_subset import load_lightrag_contexts

    load_dotenv(".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr); sys.exit(1)
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url="https://openrouter.ai/api/v1")

    load_limit = None if args.balance_hops else args.n_input
    pool_recs = load_train_pool(args.train_split, args.benchmark, limit=load_limit)
    if args.balance_hops:
        from collections import Counter
        pool_recs = balance_by_hop(pool_recs, args.n_input)
        print(f"Hop-balanced input: {dict(Counter(r['hops'] for r in pool_recs))}")
    else:
        pool_recs = pool_recs[:args.n_input]
    print(f"Loaded {len(pool_recs)} train records for {args.benchmark}")

    train_ctx = {}
    if args.train_contexts:
        train_ctx = load_lightrag_contexts(args.train_contexts)
        print(f"  using {len(train_ctx)} provided retrieved contexts")

    candidates = []
    n_correct = n_struct = 0
    for i, rec in enumerate(pool_recs):
        qid, question = rec["id"], rec["question"]
        context = trim_context(train_ctx.get(qid) or rec.get("snippet") or "")
        if not context:
            continue
        try:
            pred, raw = answer_with_few_shot_sparql_cot(
                client, args.teacher_model, question, context, temperature=0.0)
        except Exception as e:
            print(f"  [warn] teacher error on {qid}: {e}"); continue
        try:
            v = eval_once(client, args.judge_model, question,
                          _gold_list({"answer": rec["answer"]}), pred)
            verdict = v.get("verdict", "unknown")
        except Exception:
            verdict = "unknown"
        if verdict != "correct":
            continue
        n_correct += 1
        if not structural_ok(raw, hops=rec["hops"]):
            continue
        n_struct += 1
        candidates.append({
            "id": qid, "question": question, "answer": rec["answer"],
            "type": rec["type"], "hops": rec["hops"],
            "n_triples": count_triple_patterns(raw),
            "rendered": build_rendered(raw, rec["type"], context),
        })
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(pool_recs)}] correct={n_correct} "
                  f"clean={n_struct} candidates={len(candidates)}")

    pool = select_pool(candidates, args.per_type)
    out = {"benchmark": args.benchmark, "teacher_model": args.teacher_model,
           "n_input": len(pool_recs), "n_correct": n_correct,
           "n_clean": n_struct, "n_selected": len(pool), "examples": pool}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nHarvest: {n_correct} correct, {n_struct} clean, "
          f"{len(pool)} selected -> {args.out}")
    if n_correct and n_correct / max(len(pool_recs), 1) < 0.40:
        print("  [note] teacher correct-rate < 40% (plan risk #4): consider a "
              "larger --n-input to reach the candidate pool.")


if __name__ == "__main__":
    main()
