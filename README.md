# Faithfulness probes for small-model Graph-RAG

This repository holds the experiment scripts for a study of *faithfulness and
calibration* failures in small (8B) language models used for graph-structured
retrieval-augmented question answering. The probes diagnose where the small-model
accuracy gap comes from (parametric leakage, retrieval recall, or failure to use
retrieved evidence) and test grounding + calibrated re-asking as a recovery
mechanism, across the HotpotQA, MuSiQue, and 2WikiMultiHopQA multi-hop
benchmarks.

The scripts in `probe/` are the analysis and run harness only. They do **not**
vendor the benchmark data or the base QA pipeline; both are provided by the prior
work this study builds on (see Prerequisites).

## Prerequisites

1. **Paper 1 base repository (arXiv 2603.14045).** The core scripts import the
   prior work's `qa_pipeline` module (prompt construction + answer evaluation),
   read its `.env`, and load its benchmark data, SPARQL-CoT result CSVs, and
   supporting-fact coverage files. Clone that repository and point the
   `PAPER1_REPO` environment variable at its root (see Environment variables).

2. **A Groq API key.** The scripts that (re)generate model answers call the Groq
   API. Provide `GROQ_API_KEY` either in the Paper 1 repo's `.env` (the scripts
   read it from there automatically) or in your own environment. A few probes use
   an OpenAI-compatible endpoint instead and read `OPENROUTER_API_KEY`.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Environment variables

| Variable             | Required by                                   | Meaning                                                            |
| -------------------- | --------------------------------------------- | ------------------------------------------------------------------ |
| `PAPER1_REPO`        | the core `pilot_*` / `gen_*` / `run_*` scripts | Absolute path to the Paper 1 (arXiv 2603.14045) base repo root.    |
| `GROQ_API_KEY`       | scripts that call the Groq API                | Groq API key (also read from `$PAPER1_REPO/.env` if present).      |
| `OPENROUTER_API_KEY` | the OpenAI-compatible probe variants          | API key for the OpenAI-compatible endpoint.                        |

Example (PowerShell):

```powershell
$env:PAPER1_REPO = "C:\path\to\paper1-base-repo"
$env:GROQ_API_KEY = "..."
```

Example (bash):

```bash
export PAPER1_REPO=/path/to/paper1-base-repo
export GROQ_API_KEY=...
```

All scripts live in `probe/` and import sibling `probe_*` modules, so run them
from inside that directory (or with `probe/` on `PYTHONPATH`).

## Key scripts

Core diagnosis and recovery (read from / import the Paper 1 base repo):

- **`pilot_xtab_closedbook.py`** — builds the deployment-relevant cross-tab per
  benchmark, splitting each question into parametric-leak / RAG-win / use-fail /
  unanswerable buckets from closed-book, coverage, and open-book signals.
- **`pilot_decisive.py`** — the decisive accuracy test: grounding (and gated
  re-ask) vs the baseline at T=0 greedy on the full cell, with paired-bootstrap
  CIs and trustworthiness metrics (abstain, confident-wrong, grounded-span,
  precision).
- **`pilot_grounded_reask.py`** — earlier three-arm grounded + calibrated re-ask
  run covering all three benchmarks (superseded by `pilot_decisive.py` for the
  headline accuracy claim; retained for full-benchmark coverage).
- **`pilot_ablation_bridge.py`** — ablates the SPARQL scaffold under the grounding
  regime and bridges to the in-domain few-shot ladder on a fixed MuSiQue sample.
- **`pilot_baseline_structure.py`** — structural comparison of 8B vs 70B baseline
  SPARQL-CoT chains (triple-pattern count, trace length, parse rate) from existing
  result CSVs; no API calls.
- **`pilot_ceiling.py`** — adjudicates whether the 8B-vs-70B gap is a retrieval-
  recall wall or a use problem, by joining coverage CSVs with per-question
  verdicts; no API calls.
- **`pilot_spotcheck_dump.py`** — dumps a stratified sample for a frontier-model
  spot-check of the coverage-based bucket split; no API calls.
- **`gen_lr_coverage.py`** — recomputes entity-coverage over the LightRAG contexts
  (CPU-only), validating the port against existing coverage CSVs.
- **`run_hotpotqa_lr_baseline.py`** — 8B single-shot SPARQL-CoT baseline on the
  HotpotQA-LightRAG split, the missing ingredient for the HotpotQA-LR diagnosis
  row.

Supporting probe / run modules (imported by the above and used standalone for
the closed-book, few-shot, self-consistency, and structure analyses):
`probe_run.py`, `probe_run_sc.py`, `probe_run_fewshot.py`,
`probe_run_baselines.py`, `probe_closedbook.py`, `probe_chain_structure.py`,
`probe_compute_subset.py`, `probe_extract_benchmark.py`, and the other
`probe_*.py` files. Each script's behavior is documented in its module docstring.

## Reproduction notes

- The scripts cache intermediate model outputs to gitignored `probe/_*.jsonl`
  files and are resumable; re-running picks up where a run left off.
- Reproducing the full study requires the Paper 1 base repo's benchmark data and
  result CSVs in addition to the API keys above; the scripts here do not bundle
  that data.
