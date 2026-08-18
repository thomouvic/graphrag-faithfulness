"""
Run few-shot SPARQL-CoT (single-shot, T=0.0) on KET-RAG and LightRAG.

Adds two new cell types per benchmark:
    {base}_fewshot.jsonl         : few-shot SPARQL-CoT alone
    {base}_fewshot_sc.jsonl      : few-shot SPARQL-CoT + SC (composition)

Usage:
    python -u probe_run_fewshot.py \
        --qa-pairs experiments/hotpotqa/large_scale/qa-pairs/qa-pairs.json \
        --ket-context experiments/hotpotqa/large_scale/output/large_scale-keyword-0.5.json \
        --lightrag-context experiments/hotpotqa/large_scale_lightrag/lightrag_contexts.json \
        --limit 100 \
        --out-dir probe_results/
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
from probe_fewshot_sparql import (
    answer_with_few_shot_sparql_cot,
    answer_with_few_shot_sparql_cot_sc,
)
from probe_fewshot_indomain import (
    answer_with_indomain_fewshot,
    answer_with_indomain_fewshot_sc,
)
from probe_sparql_verified import answer_with_sparql_cot_sc_verified
from probe_compute_subset import (
    load_ket_contexts, load_lightrag_contexts, _gold_strings,
)
from probe_gw import build_gw_lookup
from probe_restratify import build_restratified_lookup
from probe_indomain_examples import (
    build_selector, load_train_pool, load_pool_json, audit_contamination,
)
from probe_domain_pool import domain_pool

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
QA_MODEL = "meta-llama/llama-3.1-8b-instruct"
EVAL_MODEL = "meta-llama/llama-3.1-8b-instruct"


def _eval(client, q, gold, pred):
    try:
        v = eval_once(client, EVAL_MODEL, q, gold, pred)
        return v.get("verdict", "unknown"), v.get("reason", "")
    except Exception as e:
        return "unknown", repr(e)[:200]


def _run_cell(client, qa_subset, ctx, base, method, out_path,
              selector=None, n_shots=3, **kw):
    """method ∈ {'fewshot', 'fewshot_sc'}.

    If `selector` is given, the worked-example block is chosen per question by
    that selector (in-domain gradient); otherwise the legacy fixed generic
    examples are used (byte-identical to the original baseline)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set(); rows = []
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); rows.append(r); done.add(str(r["id"]))
        print(f"  [resume] {base}+{method}: {len(done)} already done")
    t0 = time.time()
    n = len(qa_subset)
    for i, q in enumerate(qa_subset):
        qid = str(q["id"])
        if qid in done: continue
        question = q["question"]
        gold_list = _gold_strings(q)
        c = ctx.get(qid, "")
        ex_block = (selector.select(question, qid, base, n_shots)
                    if selector is not None else None)
        try:
            if not c.strip():
                pred = "I don't know"; trace = {"empty_context": True}
            elif method == "fewshot":
                if selector is not None:
                    pred, raw = answer_with_indomain_fewshot(
                        client, QA_MODEL, question, c, ex_block, temperature=0.0
                    )
                else:
                    pred, raw = answer_with_few_shot_sparql_cot(
                        client, QA_MODEL, question, c, temperature=0.0
                    )
                trace = {"raw": raw[:500]}
            elif method == "fewshot_sc":
                if selector is not None:
                    pred, trace = answer_with_indomain_fewshot_sc(
                        client, QA_MODEL, question, c, ex_block,
                        k=kw["k"], temperature=kw["temperature"],
                    )
                else:
                    pred, trace = answer_with_few_shot_sparql_cot_sc(
                        client, QA_MODEL, question, c,
                        k=kw["k"], temperature=kw["temperature"],
                    )
                trace = {"k": trace["k"], "temperature": trace["temperature"],
                         "samples": [{"i": s["sample_idx"], "n": s["normalized"]}
                                     for s in trace["samples"]]}
            elif method == "fewshot_sc_verified":
                pred, trace = answer_with_sparql_cot_sc_verified(
                    client, QA_MODEL, question, c,
                    k=kw["k"], temperature=kw["temperature"],
                )
                trace = {"k": trace["k"], "temperature": trace["temperature"],
                         "n_grounded": trace["n_grounded"],
                         "voting_pool_size": trace["voting_pool_size"],
                         "fallback": trace["fallback"],
                         "samples": [{"i": s["sample_idx"],
                                      "n": s["normalized"],
                                      "g": s["grounded"]}
                                     for s in trace["samples"]]}
            else:
                raise ValueError(method)
        except Exception as e:
            pred = f"ERROR: {e}"; trace = {"error": repr(e)[:200]}
        verdict, reason = _eval(client, question, gold_list, pred)
        row = {"id": qid, "question": question, "gold": gold_list,
               "pred": pred, "verdict": verdict, "reason": reason,
               "base": base, "method": method, "trace": trace}
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
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--k", type=int, default=3,
                   help="SC k for fewshot_sc cells")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="SC temperature for fewshot_sc cells")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--methods", nargs="+",
                   default=["fewshot", "fewshot_sc"],
                   choices=["fewshot", "fewshot_sc", "fewshot_sc_verified"])
    p.add_argument("--gw", action="store_true",
                   help="Pre-compress contexts with graph-walk before "
                        "running few-shot. Adds '_gw' suffix to output filenames.")
    p.add_argument("--max-hops", type=int, default=3)
    p.add_argument("--budget-tokens", type=int, default=4000)
    p.add_argument("--restratify-lr", type=int, default=None, metavar="K",
                   help="Apply BM25 re-stratification to LightRAG contexts "
                        "before any other processing: keep only the top-K "
                        "chunks by question-BM25 score. KET contexts pass "
                        "through unchanged (control). Adds '_restrat' suffix.")
    # --- few-shot example variants (plan.md A/B/C) ---
    p.add_argument("--variant", default=None, choices=["a", "b", "c"],
                   help="Paper variant (plan.md): a=generic, b=domain (hand-"
                        "written in-domain), c=teacher (harvested pool). Sets "
                        "--example-source accordingly; for c, pass --teacher-pool.")
    p.add_argument("--teacher-pool", type=Path, default=None,
                   help="Per-benchmark teacher-harvested pool JSON "
                        "(from probe_fewshot_teacher.py). Required for variant c.")
    p.add_argument("--example-source", default="generic",
                   choices=["generic", "domain", "teacher",
                            "train", "dynamic", "structure"],
                   help="Worked-example source. A/B/C ladder: generic=A, "
                        "domain=B, teacher=C. (train/dynamic/structure are "
                        "appendix/future ablations — see ideas.md.)")
    p.add_argument("--n-shots", type=int, default=3,
                   help="Number of worked examples in the prompt (R2 sweep).")
    p.add_argument("--train-split", type=Path, default=None,
                   help="Path to a normalisable train split (required for "
                        "--example-source train/dynamic/structure).")
    p.add_argument("--train-benchmark", default=None,
                   choices=["musique", "hotpotqa", "2wikimultihopqa"],
                   help="Benchmark format of --train-split.")
    p.add_argument("--train-pool-limit", type=int, default=2000,
                   help="Cap the train pool size (speed; default 2000).")
    p.add_argument("--sim-backend", default="lexical",
                   choices=["lexical", "st"],
                   help="Similarity backend for dynamic/structure selection.")
    p.add_argument("--match-base-format", action="store_true",
                   help="R6 only: render example CONTEXT blocks in the target "
                        "base's style (ket chunk-dump / lightrag relation-summary).")
    args = p.parse_args()

    load_dotenv(".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set"); sys.exit(1)

    qa_list = json.loads(args.qa_pairs.read_text(encoding="utf-8"))
    qa_subset = qa_list[:args.limit]
    print(f"Loaded {len(qa_subset)} questions")

    ket_ctx = load_ket_contexts(args.ket_context)
    lr_ctx = load_lightrag_contexts(args.lightrag_context)

    # Optional: BM25-restratify LR contexts BEFORE any other processing.
    # KET contexts pass through unchanged (β-stratified-by-design control).
    if args.restratify_lr is not None:
        print(f"BM25-restratifying LightRAG contexts (top_k={args.restratify_lr})...")
        lr_ctx, r_stats = build_restratified_lookup(qa_subset, lr_ctx,
                                                    top_k=args.restratify_lr)
        print(f"  modified={r_stats['n_modified']} unchanged={r_stats['n_unchanged']}  "
              f"avg_chars {r_stats['avg_orig_chars']:.0f} -> {r_stats['avg_new_chars']:.0f}")

    if args.gw:
        print("Compressing KET contexts via graph-walk...")
        ket_ctx, k_stats = build_gw_lookup(qa_subset, ket_ctx, "ket",
                                          max_hops=args.max_hops,
                                          budget_tokens=args.budget_tokens)
        print(f"  compressed={k_stats['n_compressed']} fallback={k_stats['n_fallback']} "
              f"avg_words {k_stats['avg_orig_words']:.0f} -> {k_stats['avg_new_words']:.0f}")
        print("Compressing LightRAG contexts via graph-walk...")
        lr_ctx, l_stats = build_gw_lookup(qa_subset, lr_ctx, "lightrag",
                                         max_hops=args.max_hops,
                                         budget_tokens=args.budget_tokens)
        print(f"  compressed={l_stats['n_compressed']} fallback={l_stats['n_fallback']} "
              f"avg_words {l_stats['avg_orig_words']:.0f} -> {l_stats['avg_new_words']:.0f}")

    # --variant {a,b,c} is sugar over --example-source (plan.md A/B/C).
    if args.variant:
        args.example_source = {"a": "generic", "b": "domain", "c": "teacher"}[args.variant]

    # --- build the few-shot example selector ---
    selector = None
    if args.example_source != "generic" or args.n_shots != 3:
        if args.example_source == "generic":
            selector = build_selector("generic")
        elif args.example_source == "domain":
            pool = domain_pool()
            print(f"Domain-flavoured pool (variant B): {len(pool)} examples")
            selector = build_selector("domain", pool=pool)
        elif args.example_source == "teacher":
            if not args.teacher_pool:
                print("ERROR: --teacher-pool required for variant c / "
                      "--example-source teacher", file=sys.stderr)
                sys.exit(1)
            pool = load_pool_json(args.teacher_pool)
            print(f"Teacher-harvested pool (variant C): {len(pool)} examples "
                  f"from {args.teacher_pool}")
            pool, audit = audit_contamination(pool, qa_subset)
            print(f"  contamination audit: {audit}")
            (args.out_dir).mkdir(parents=True, exist_ok=True)
            (args.out_dir / "contamination_audit.json").write_text(
                json.dumps(audit, indent=2), encoding="utf-8")
            selector = build_selector("teacher", pool=pool)
        else:  # train / dynamic / structure -> need a train split
            if not args.train_split or not args.train_benchmark:
                print("ERROR: --train-split and --train-benchmark are required "
                      f"for --example-source {args.example_source}", file=sys.stderr)
                sys.exit(1)
            pool = load_train_pool(args.train_split, args.train_benchmark,
                                   limit=args.train_pool_limit)
            print(f"Train pool: {len(pool)} records from {args.train_split}")
            pool, audit = audit_contamination(pool, qa_subset)
            print(f"  contamination audit: {audit}")
            (args.out_dir).mkdir(parents=True, exist_ok=True)
            (args.out_dir / "contamination_audit.json").write_text(
                json.dumps(audit, indent=2), encoding="utf-8")
            selector = build_selector(
                args.example_source, pool=pool, sim_backend=args.sim_backend,
                match_base_format=args.match_base_format,
            )

    suffix = ""
    if args.restratify_lr is not None:
        suffix += f"_restrat{args.restratify_lr}"
    if args.gw:
        suffix += "_gw"
    if selector is not None:
        suffix += f"_src-{args.example_source}_shots{args.n_shots}"
        if args.match_base_format:
            suffix += "_bfmt"

    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url=OPENROUTER_BASE_URL)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for base, ctx in [("ket", ket_ctx), ("lightrag", lr_ctx)]:
        for method in args.methods:
            out_path = args.out_dir / f"{base}_{method}{suffix}.jsonl"
            print(f"\n=== Cell: {base} + {method}{suffix} ===")
            rows = _run_cell(client, qa_subset, ctx, base, method, out_path,
                             selector=selector, n_shots=args.n_shots,
                             k=args.k, temperature=args.temperature)
            correct = sum(1 for r in rows if r["verdict"] == "correct")
            summary[f"{base}_{method}{suffix}"] = {
                "n": len(rows), "acc": correct / max(len(rows), 1),
            }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:30s}  acc={v['acc']:.3f}  n={v['n']}")
    (args.out_dir / "summary_fewshot.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
