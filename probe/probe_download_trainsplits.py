"""
Fetch the benchmark TRAIN splits used to build the in-domain few-shot pool.

These are the official train splits — disjoint from the eval N=500 (which are
drawn from the dev/distractor sets). The in-domain study sources worked
examples ONLY from here; probe_indomain_examples.audit_contamination then drops
any residual overlap against the eval set before use.

Usage:
    python3 probe_download_trainsplits.py --benchmark musique
    python3 probe_download_trainsplits.py --benchmark hotpotqa
    python3 probe_download_trainsplits.py --benchmark 2wikimultihopqa --limit 5000

Writes train_splits/<benchmark>_train.(jsonl|json).

If a direct download is unavailable in your environment, the script prints the
canonical source URL so you can place the file manually at the expected path.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "train_splits"

# Canonical public sources. Mirror locally if your network blocks these.
SOURCES = {
    "musique": {
        "url": "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_train.jsonl",
        "dest": "musique_train.jsonl",
        "note": "MuSiQue-Answerable train; has question_decomposition + paragraphs.",
    },
    "hotpotqa": {
        "url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
        "dest": "hotpotqa_train.json",
        "note": "HotpotQA train; has type (bridge/comparison) + supporting_facts.",
    },
    "2wikimultihopqa": {
        "url": "https://www.dropbox.com/s/ms2m13252h6xsubs/data_ids_april7.zip?dl=1",
        "dest": "2wikimultihopqa_train.json",
        "note": "2WikiMultiHopQA train; evidences are (subj,rel,obj) triples. "
                "NOTE: distributed as a zip — extract train.json manually.",
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=list(SOURCES.keys()))
    p.add_argument("--limit", type=int, default=None,
                   help="Optional: truncate after N records (jsonl only).")
    args = p.parse_args()

    src = SOURCES[args.benchmark]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / src["dest"]

    if dest.exists():
        print(f"[skip] {dest} already exists ({dest.stat().st_size} bytes)")
        return

    print(f"=== {args.benchmark} ===")
    print(f"  note: {src['note']}")
    print(f"  source: {src['url']}")
    print(f"  dest:   {dest}")
    try:
        if args.benchmark == "2wikimultihopqa":
            raise RuntimeError("zip archive — manual extraction required")
        if args.limit and src["dest"].endswith(".jsonl"):
            _download_jsonl_limited(src["url"], dest, args.limit)
        else:
            urllib.request.urlretrieve(src["url"], dest)
        print(f"  [OK] wrote {dest} ({dest.stat().st_size} bytes)")
    except Exception as e:
        print(f"  [MANUAL] automatic download failed: {e}", file=sys.stderr)
        print(f"  Please download {src['url']} and place it at {dest}",
              file=sys.stderr)
        sys.exit(2)


def _download_jsonl_limited(url, dest, limit):
    with urllib.request.urlopen(url) as resp, open(dest, "w", encoding="utf-8") as out:
        n = 0
        for line in resp:
            out.write(line.decode("utf-8"))
            n += 1
            if n >= limit:
                break
    print(f"  [OK] truncated to {limit} records")


if __name__ == "__main__":
    main()
