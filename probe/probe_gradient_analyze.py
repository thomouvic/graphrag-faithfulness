"""
Analyse the in-domain few-shot improvement gradient (R1 -> R6).

Reads the per-cell JSONL files produced by probe_run_fewshot.py for one
(base, method) combination, groups them by example-source, and reports:

  * overall accuracy per source,
  * paired-bootstrap 95% CI on the delta vs the generic (R1) baseline,
  * the same, stratified by hop-count and question-type,

so you can read off where on the gradient the gains appear (deep hops?
comparison questions? LightRAG cells?).

Stdlib-only (uses random for the bootstrap) so it runs anywhere.

Usage:
    python3 probe_gradient_analyze.py --probe-dir probe_results --base ket --method fewshot
    python3 probe_gradient_analyze.py --probe-dir probe_results --base lightrag --method fewshot_sc \
        --out probe_results/gradient_lightrag.json
"""
import argparse
import json
import random
import re
from pathlib import Path

from probe_indomain_examples import hop_count_from_qid, classify_question_type

random.seed(20260531)  # deterministic CIs

# Canonical ordering for display. A/B/C are the paper variants (plan.md);
# train/dynamic/structure are appendix/future ablations.
_RUNG_ORDER = {"generic": 1, "domain": 2, "teacher": 3,
               "train": 4, "dynamic": 5, "structure": 6}
_VARIANT_LABEL = {"generic": "A (generic)", "domain": "B (in-domain)",
                  "teacher": "C (teacher)"}


def _source_of(fname: str, base: str, method: str) -> str:
    """Extract the example-source label from a cell filename."""
    stem = Path(fname).stem
    rest = stem[len(f"{base}_{method}"):]  # suffix after base_method
    m = re.search(r"src-([a-z]+)", rest)
    if m:
        return m.group(1)
    # legacy files with no src suffix == generic R1 baseline
    return "generic"


def _shots_of(fname: str) -> int:
    m = re.search(r"shots(\d+)", Path(fname).stem)
    return int(m.group(1)) if m else 3


def load_cell(path: Path) -> dict:
    """Return {qid: 1/0 correct} for a cell JSONL."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[str(r["id"])] = 1 if r.get("verdict") == "correct" else 0
    return out


def paired_bootstrap(a: dict, b: dict, n_boot: int = 5000):
    """Paired bootstrap on shared qids. Returns (mean_delta, lo, hi).

    delta = acc(b) - acc(a). CI is the 2.5/97.5 percentiles of resampled delta.
    """
    ids = sorted(set(a) & set(b))
    if not ids:
        return 0.0, 0.0, 0.0
    diffs = [b[i] - a[i] for i in ids]
    n = len(diffs)
    mean = sum(diffs) / n
    boots = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += diffs[random.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return mean, lo, hi


def acc(cell: dict, ids=None) -> float:
    items = [cell[i] for i in (ids or cell.keys()) if i in cell]
    return sum(items) / len(items) if items else 0.0


def strata(qmeta: dict):
    """Yield (label, predicate) strata over qids."""
    yield "all", lambda qid: True
    for h in (2, 3, 4):
        yield f"hop={h}", lambda qid, h=h: qmeta[qid]["hop"] == h
    for t in ("bridge", "comparison"):
        yield f"type={t}", lambda qid, t=t: qmeta[qid]["type"] == t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-dir", required=True, type=Path)
    p.add_argument("--base", required=True, choices=["ket", "lightrag"])
    p.add_argument("--method", required=True,
                   choices=["fewshot", "fewshot_sc"])
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    pattern = f"{args.base}_{args.method}*.jsonl"
    files = sorted(args.probe_dir.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {pattern} in {args.probe_dir}")

    # group by (source, shots); collapse to source for the main gradient,
    # keep shots separate only if multiple shot counts are present.
    cells = {}
    qmeta = {}
    questions = {}
    for f in files:
        src = _source_of(f.name, args.base, args.method)
        shots = _shots_of(f.name)
        key = src if shots == 3 else f"{src}@{shots}shot"
        cell = load_cell(f)
        cells[key] = cell
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            qid = str(r["id"])
            qmeta.setdefault(qid, {
                "hop": hop_count_from_qid(qid),
                "type": classify_question_type(r.get("question", "")),
            })

    if "generic" not in cells:
        print("[warn] no generic (R1) baseline cell found; deltas vs first cell")
    base_key = "generic" if "generic" in cells else sorted(cells)[0]
    base_cell = cells[base_key]

    # ---- report ----
    order = sorted(cells, key=lambda k: (_RUNG_ORDER.get(k.split("@")[0], 99), k))
    report = {"base": args.base, "method": args.method,
              "baseline": base_key, "rungs": {}}

    print("=" * 96)
    print(f"IN-DOMAIN FEW-SHOT GRADIENT  —  base={args.base}  method={args.method}")
    print(f"baseline rung = {base_key}")
    print("=" * 96)
    header = f"{'rung':<18}{'n':>5}{'acc':>8}{'Δ vs base':>12}{'95% CI':>22}{'sig':>5}"
    print(header)
    print("-" * 96)

    for key in order:
        cell = cells[key]
        a_all = acc(cell)
        mean, lo, hi = paired_bootstrap(base_cell, cell, args.n_boot)
        sig = "✓" if (lo > 0 or hi < 0) else ""
        if key == base_key:
            sig = "—"
        print(f"{key:<18}{len(cell):>5}{a_all:>8.3f}{mean:>+12.3f}"
              f"{f'[{lo:+.3f},{hi:+.3f}]':>22}{sig:>5}")
        # stratified
        strat = {}
        for label, pred in strata(qmeta):
            ids = [i for i in cell if pred(i)]
            if label == "all":
                continue
            base_ids = {i: base_cell[i] for i in ids if i in base_cell}
            cur_ids = {i: cell[i] for i in ids if i in base_cell}
            if len(cur_ids) < 10:
                continue
            m2, lo2, hi2 = paired_bootstrap(base_ids, cur_ids, args.n_boot)
            strat[label] = {"n": len(cur_ids), "acc": acc(cell, ids),
                            "delta": m2, "ci": [lo2, hi2]}
        report["rungs"][key] = {"n": len(cell), "acc": a_all,
                                "delta": mean, "ci": [lo, hi], "strata": strat}

    # stratified gradient table (delta vs base per stratum)
    print("\n" + "-" * 96)
    print("STRATIFIED Δ vs baseline (only strata with n>=10):")
    strata_labels = ["type=bridge", "type=comparison", "hop=2", "hop=3", "hop=4"]
    hdr = f"{'rung':<18}" + "".join(f"{s:>16}" for s in strata_labels)
    print(hdr)
    for key in order:
        if key == base_key:
            continue
        row = f"{key:<18}"
        for s in strata_labels:
            st = report["rungs"][key]["strata"].get(s)
            cell_txt = f"{st['delta']:+.3f}" if st else "—"
            row += f"{cell_txt:>16}"
        print(row)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
