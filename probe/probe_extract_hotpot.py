"""
Targeted extractor for HotpotQA-only data from the merged HF index zips.

Both zips contain 3 benchmarks overlaid at the same Windows-style paths
(\\ separators). For each duplicate-path entry, the FIRST chronological
occurrence is HotpotQA (verified by qid prefix `5a76f45a...`).

Extracts only the files the probe needs into:
    experiments/hotpotqa/large_scale/
    experiments/hotpotqa/large_scale_lightrag/
"""
import csv
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
KET_ZIP = REPO / "hf_dl" / "ketrag_indexes_N500.zip"
LR_ZIP = REPO / "hf_dl" / "lightrag_indexes_N500.zip"

KET_OUT = REPO / "experiments" / "hotpotqa" / "large_scale"
LR_OUT = REPO / "experiments" / "hotpotqa" / "large_scale_lightrag"


def _norm_path(zip_path: str) -> str:
    """Replace Windows backslashes with forward slashes."""
    return zip_path.replace("\\", "/")


HOTPOT_QID = re.compile(r"^[a-f0-9]{24}$")
MUSIQUE_QID = re.compile(r"^\dhop\d*__")
TWIKI_QID = re.compile(r"^[a-f0-9]{32}$")


def _is_hotpot_qid(qid: str) -> bool:
    """HotpotQA qids are 24-char hex strings."""
    return bool(qid and HOTPOT_QID.match(qid))


def _detect_benchmark_from_first_qid(qid: str) -> str:
    if _is_hotpot_qid(qid):
        return "hotpot"
    if MUSIQUE_QID.match(qid or ""):
        return "musique"
    if TWIKI_QID.match(qid or ""):
        return "2wiki"
    return "unknown"


def _benchmark_of_jsonish_entry(zf, info) -> str:
    """For .json/.jsonl files, peek at the first record to identify benchmark."""
    name = _norm_path(info.filename)
    try:
        with zf.open(info) as f:
            if name.endswith(".jsonl"):
                first_line = f.readline().decode("utf-8", errors="ignore")
                rec = json.loads(first_line)
                qid = str(rec.get("id") or rec.get("qid") or "")
                return _detect_benchmark_from_first_qid(qid)
            elif name.endswith(".json"):
                # Read just enough; for huge files, parse incrementally would be
                # nicer, but lightrag_contexts.json is dict-shaped and json.load
                # is the safest path.
                data = json.load(f)
                if isinstance(data, dict) and data:
                    qid = str(next(iter(data.keys())))
                    return _detect_benchmark_from_first_qid(qid)
                if isinstance(data, list) and data:
                    qid = str(data[0].get("id", "") if isinstance(data[0], dict) else "")
                    return _detect_benchmark_from_first_qid(qid)
    except Exception as e:
        print(f"    [warn] could not detect benchmark for {name}: {e}")
    return "unknown"


def extract_hotpot_match(zf: zipfile.ZipFile, suffix_unix: str, dest: Path,
                         use_content_detection: bool = True):
    """Extract the HotpotQA variant of a duplicate-path entry.

    If `use_content_detection` and the file is json/jsonl, identify benchmark
    by sampling the first qid. Otherwise fall back to chronologically-first.
    """
    matches = []
    for info in zf.infolist():
        norm = _norm_path(info.filename)
        if norm.endswith(suffix_unix):
            matches.append(info)
    if not matches:
        print(f"  [MISS] {suffix_unix} — not in zip")
        return False

    chosen = None
    method = ""
    if use_content_detection and (suffix_unix.endswith(".json") or
                                   suffix_unix.endswith(".jsonl")):
        for info in matches:
            bench = _benchmark_of_jsonish_entry(zf, info)
            if bench == "hotpot":
                chosen = info
                method = "content-detected"
                break
        if chosen is None:
            print(f"  [MISS] {suffix_unix} — no HotpotQA variant found by content")
            return False
    else:
        matches.sort(key=lambda i: i.date_time)
        chosen = matches[0]
        method = "chronological-first"

    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(chosen) as src, open(dest, "wb") as dst:
        dst.write(src.read())
    print(f"  [OK ] {suffix_unix} <- {chosen.filename} "
          f"(date={chosen.date_time}, size={chosen.file_size}, via {method})")
    return True


# Keep the old name as an alias for non-content-detected uses
extract_first_match = extract_hotpot_match


def extract_hotpot_baseline_csv_to_jsonl(zf: zipfile.ZipFile, dest_jsonl: Path):
    """The KET-RAG zip has hotpotqa baseline as CSV in results_baseline_groq/.
    Convert the EARLIEST hotpotqa 8B normalized run to JSONL with the schema
    expected by probe_compute_subset.py: {id, eval_verdict}.
    """
    candidates = []
    for info in zf.infolist():
        norm = _norm_path(info.filename).lower()
        if ("results_baseline_groq" in norm
                and "hotpotqa" in norm
                and "8b-instant" in norm
                and "normalized" in norm
                and norm.endswith(".csv")):
            candidates.append(info)
    if not candidates:
        print("  [MISS] no hotpotqa 8B normalized baseline CSV")
        return False
    candidates.sort(key=lambda i: i.date_time)
    chosen = candidates[0]
    print(f"  [OK ] reading baseline CSV: {chosen.filename}")
    with zf.open(chosen) as f:
        text = io.TextIOWrapper(f, encoding="utf-8", newline="")
        reader = csv.DictReader(text)
        rows = list(reader)
    print(f"        n_rows={len(rows)}")
    n_correct = sum(1 for r in rows if r.get("eval_verdict") == "correct")
    print(f"        n_correct={n_correct}  acc={n_correct/len(rows):.3f}")
    dest_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            out = {
                "id": r["id"],
                "eval_verdict": r.get("eval_verdict", ""),
                "final_pred": r.get("final_pred", ""),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"        wrote: {dest_jsonl}")
    return True


def main():
    if not KET_ZIP.exists() or not LR_ZIP.exists():
        print("ERROR: zips not found in hf_dl/", file=sys.stderr); sys.exit(1)

    print("=== Extracting HotpotQA from KET-RAG zip ===")
    with zipfile.ZipFile(KET_ZIP) as z:
        # Required by probe_compute_subset.py + probe_run.py
        extract_first_match(z, "qa-pairs/qa-pairs.json",
                            KET_OUT / "qa-pairs" / "qa-pairs.json")
        extract_first_match(z, "output/large_scale-keyword-0.5.json",
                            KET_OUT / "output" / "large_scale-keyword-0.5.json")
        # Optional but useful for sanity
        extract_first_match(z, "settings.yaml",
                            KET_OUT / "settings.yaml")
        # Baseline JSONL (build from CSV since no JSONL in zip for HotpotQA)
        extract_hotpot_baseline_csv_to_jsonl(
            z, KET_OUT / "checkpoints" / "baseline.jsonl")

    print()
    print("=== Extracting HotpotQA from LightRAG zip ===")
    with zipfile.ZipFile(LR_ZIP) as z:
        extract_first_match(z, "lightrag_contexts.json",
                            LR_OUT / "lightrag_contexts.json")
        extract_first_match(z, "results/Baseline_8B.jsonl",
                            LR_OUT / "results" / "Baseline_8B.jsonl")

    print()
    print("=== Verification ===")
    expected = [
        KET_OUT / "qa-pairs" / "qa-pairs.json",
        KET_OUT / "output" / "large_scale-keyword-0.5.json",
        KET_OUT / "checkpoints" / "baseline.jsonl",
        LR_OUT / "lightrag_contexts.json",
        LR_OUT / "results" / "Baseline_8B.jsonl",
    ]
    for p in expected:
        ok = p.exists()
        sz = p.stat().st_size if ok else 0
        print(f"  {'OK ' if ok else 'NO '}  {sz:>12}  {p.relative_to(REPO)}")

    # Confirm HotpotQA by sampling first qid
    qa_path = KET_OUT / "qa-pairs" / "qa-pairs.json"
    if qa_path.exists():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if isinstance(qa, list) and qa:
            qid = qa[0].get("id", "")
            is_hotpot = bool(re.match(r"^[a-f0-9]{24}$", qid))
            print(f"\n  qa-pairs[0].id = {qid!r}  HotpotQA-shape: {is_hotpot}  n={len(qa)}")


if __name__ == "__main__":
    main()
