"""
Targeted extractor for one benchmark from the merged HF index zips.

Generalization of probe_extract_hotpot.py — supports HotpotQA, MuSiQue,
and 2WikiMultiHopQA. Uses content-based benchmark detection (qid format)
for files that have multiple variants overlaid at the same Windows-style
path inside the zip.

Identifies benchmark by first qid format:
    HotpotQA      : 24-char hex     (e.g. 5a76f45a5542994aec3b719b)
    MuSiQue       : Nhop... pattern (e.g. 3hop1__390673_228453_86925)
    2WikiMHQA     : 32-char hex     (e.g. 9d054e980bdd11eba7f7acde48001122)

Usage:
    python probe_extract_benchmark.py --benchmark hotpotqa
    python probe_extract_benchmark.py --benchmark musique
    python probe_extract_benchmark.py --benchmark 2wikimultihopqa
"""
import argparse
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
KET_ZIP = REPO / "hf_dl" / "ketrag_indexes_N500.zip"
LR_ZIP = REPO / "hf_dl" / "lightrag_indexes_N500.zip"

QID_PATTERNS = {
    "hotpotqa":        re.compile(r"^[a-f0-9]{24}$"),
    "musique":         re.compile(r"^\dhop\d*__"),
    "2wikimultihopqa": re.compile(r"^[a-f0-9]{32}$"),
}

# Substring that appears in result CSV / JSONL filenames.
CSV_NAME_SUBSTR = {
    "hotpotqa":        "hotpotqa",
    "musique":         "musique",
    "2wikimultihopqa": "2wikimultihopqa",
}


def _norm_path(zip_path: str) -> str:
    return zip_path.replace("\\", "/")


def _detect_benchmark_from_qid(qid: str) -> str:
    for name, pat in QID_PATTERNS.items():
        if pat.match(qid or ""):
            return name
    return "unknown"


def _benchmark_of_jsonish_entry(zf: zipfile.ZipFile, info, target_bench: str) -> bool:
    """True iff the entry's first qid matches target_bench."""
    name = _norm_path(info.filename)
    try:
        with zf.open(info) as f:
            if name.endswith(".jsonl"):
                first_line = f.readline().decode("utf-8", errors="ignore")
                rec = json.loads(first_line)
                qid = str(rec.get("id") or rec.get("qid") or "")
                return _detect_benchmark_from_qid(qid) == target_bench
            elif name.endswith(".json"):
                data = json.load(f)
                if isinstance(data, dict) and data:
                    qid = str(next(iter(data.keys())))
                    return _detect_benchmark_from_qid(qid) == target_bench
                if isinstance(data, list) and data:
                    qid = str(data[0].get("id", "") if isinstance(data[0], dict) else "")
                    return _detect_benchmark_from_qid(qid) == target_bench
    except Exception as e:
        print(f"    [warn] could not detect benchmark for {name}: {e}")
    return False


def extract_match(zf: zipfile.ZipFile, suffix_unix: str, dest: Path,
                  target_bench: str, use_content: bool = True):
    matches = [info for info in zf.infolist()
               if _norm_path(info.filename).endswith(suffix_unix)]
    if not matches:
        print(f"  [MISS] {suffix_unix} — not in zip")
        return False

    chosen = None
    method = ""
    if use_content and (suffix_unix.endswith(".json") or suffix_unix.endswith(".jsonl")):
        for info in matches:
            if _benchmark_of_jsonish_entry(zf, info, target_bench):
                chosen = info
                method = f"content-detected ({target_bench})"
                break
        if chosen is None:
            print(f"  [MISS] {suffix_unix} — no {target_bench} variant found")
            return False
    else:
        # For non-content files, we have no easy way to attribute. The settings
        # files are tiny and benchmark-agnostic anyway, so chronological-first
        # is fine for those. (Caller chooses use_content.)
        matches.sort(key=lambda i: i.date_time)
        chosen = matches[0]
        method = "chronological-first"

    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(chosen) as src, open(dest, "wb") as dst:
        dst.write(src.read())
    print(f"  [OK ] {suffix_unix} <- {chosen.filename} "
          f"(date={chosen.date_time}, size={chosen.file_size}, via {method})")
    return True


def extract_baseline_csv_to_jsonl(zf: zipfile.ZipFile, target_bench: str,
                                  dest_jsonl: Path):
    bench_substr = CSV_NAME_SUBSTR[target_bench]
    candidates = []
    for info in zf.infolist():
        norm = _norm_path(info.filename).lower()
        if ("results_baseline_groq" in norm
                and bench_substr in norm
                and "8b-instant" in norm
                and "normalized" in norm
                and norm.endswith(".csv")):
            candidates.append(info)
    if not candidates:
        print(f"  [MISS] no {target_bench} 8B normalized baseline CSV")
        return False
    candidates.sort(key=lambda i: i.date_time)
    chosen = candidates[0]
    print(f"  [OK ] reading baseline CSV: {chosen.filename}")
    with zf.open(chosen) as f:
        text = io.TextIOWrapper(f, encoding="utf-8", newline="")
        reader = csv.DictReader(text)
        rows = list(reader)
    n_correct = sum(1 for r in rows if r.get("eval_verdict") == "correct")
    print(f"        n_rows={len(rows)} n_correct={n_correct} "
          f"acc={n_correct/len(rows):.3f}")
    dest_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "id": r["id"],
                "eval_verdict": r.get("eval_verdict", ""),
                "final_pred": r.get("final_pred", ""),
            }, ensure_ascii=False) + "\n")
    print(f"        wrote: {dest_jsonl}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True,
                   choices=list(QID_PATTERNS.keys()))
    args = p.parse_args()

    bench = args.benchmark
    ket_out = REPO / "experiments" / bench / "large_scale"
    lr_out = REPO / "experiments" / bench / "large_scale_lightrag"

    if not KET_ZIP.exists() or not LR_ZIP.exists():
        print("ERROR: zips not found in hf_dl/", file=sys.stderr); sys.exit(1)

    print(f"=== Extracting {bench} from KET-RAG zip ===")
    with zipfile.ZipFile(KET_ZIP) as z:
        extract_match(z, "qa-pairs/qa-pairs.json",
                      ket_out / "qa-pairs" / "qa-pairs.json", bench)
        extract_match(z, "output/large_scale-keyword-0.5.json",
                      ket_out / "output" / "large_scale-keyword-0.5.json", bench)
        extract_match(z, "settings.yaml", ket_out / "settings.yaml",
                      bench, use_content=False)
        extract_baseline_csv_to_jsonl(z, bench,
                                      ket_out / "checkpoints" / "baseline.jsonl")

    print()
    print(f"=== Extracting {bench} from LightRAG zip ===")
    with zipfile.ZipFile(LR_ZIP) as z:
        extract_match(z, "lightrag_contexts.json",
                      lr_out / "lightrag_contexts.json", bench)
        extract_match(z, "results/Baseline_8B.jsonl",
                      lr_out / "results" / "Baseline_8B.jsonl", bench)

    print()
    print("=== Verification ===")
    expected = [
        ket_out / "qa-pairs" / "qa-pairs.json",
        ket_out / "output" / "large_scale-keyword-0.5.json",
        ket_out / "checkpoints" / "baseline.jsonl",
        lr_out / "lightrag_contexts.json",
        lr_out / "results" / "Baseline_8B.jsonl",
    ]
    for p in expected:
        ok = p.exists()
        sz = p.stat().st_size if ok else 0
        print(f"  {'OK ' if ok else 'NO '}  {sz:>12}  {p.relative_to(REPO)}")

    qa_path = ket_out / "qa-pairs" / "qa-pairs.json"
    if qa_path.exists():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if isinstance(qa, list) and qa:
            qid = qa[0].get("id", "")
            print(f"\n  qa-pairs[0].id = {qid!r}  matches {bench}: "
                  f"{_detect_benchmark_from_qid(qid) == bench}  n={len(qa)}")


if __name__ == "__main__":
    main()
