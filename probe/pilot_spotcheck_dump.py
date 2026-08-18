"""
Dump a stratified sample for a frontier-model (Claude) spot-check of the
coverage-based bucket split, NO API. For each sampled question it writes the
question, gold answer, the coverage flag (all_entities_present), the open/closed
-book verdicts, and a readable slice of the retrieved KET context (entities +
relationships + any window where the gold string appears).

Claude then reads the dump and judges, per question, whether the gold answer is
genuinely DERIVABLE from the shown context, and we compare that human-grade
judgment to the cheap coverage flag that built the useFail / unans buckets.
"""
import os, sys, json, csv, glob, re, random
from pathlib import Path

UP = Path(os.environ["PAPER1_REPO"])  # Paper 1 (arXiv 2603.14045) base repo
HERE = Path(__file__).parent
sys.path.insert(0, str(UP))
import qa_pipeline as qp
csv.field_size_limit(10_000_000)

# how many per (bench, bucket)
PLAN = {"musique": 10, "2wikimultihopqa": 5}


def open_book(bench):
    f = sorted(glob.glob(str(UP / "experiments" / bench / "large_scale" /
                              "results_sparql_groq" / "*llama-3.1-8b*.csv")))[-1]
    return {str(r["id"]): r["eval_verdict"].strip().lower() == "correct"
            for r in csv.DictReader(open(f, encoding="utf-8"))}


def closed_book(bench):
    c = HERE / f"_closedbook_{bench}_8b.jsonl"
    return {str(json.loads(l)["id"]): json.loads(l)["correct"]
            for l in c.read_text(encoding="utf-8").splitlines() if l.strip()}


def coverage(bench):
    out = {}
    for r in csv.DictReader(open(UP / "revision/coverage" / f"{bench}_coverage.csv",
                                 encoding="utf-8")):
        out[str(r["id"])] = r["all_entities_present"].strip().lower() == "true"
    return out


def excerpt(context, gold_strs, max_ent=3200, max_rel=1500):
    """Readable slice: entities section + relationships section + gold windows."""
    parts = {}
    for header in ["-----Entities-----", "-----Relationships-----",
                   "-----Sources-----", "-----Text source that may be relevant-----"]:
        i = context.find(header)
        parts[header] = i
    out = []
    # entities
    ei = context.find("-----Entities-----")
    ri = context.find("-----Relationships-----")
    if ei >= 0:
        end = ri if ri > ei else ei + max_ent
        out.append("ENTITIES:\n" + context[ei:min(end, ei + max_ent)].strip())
    if ri >= 0:
        si = context.find("-----Sources-----", ri)
        end = si if si > ri else ri + max_rel
        out.append("RELATIONSHIPS:\n" + context[ri:min(end, ri + max_rel)].strip())
    # gold windows (does the answer literally appear, and in what relation)
    low = context.lower()
    wins = []
    for g in gold_strs:
        gl = g.lower().strip()
        if len(gl) < 2:
            continue
        k = low.find(gl)
        if k >= 0:
            wins.append(f"  [gold '{g}' FOUND] ...{context[max(0,k-120):k+len(g)+120]}...")
        else:
            wins.append(f"  [gold '{g}' NOT found verbatim]")
    out.append("GOLD-STRING CHECK:\n" + "\n".join(wins))
    return "\n\n".join(out)


def main():
    random.seed(7)
    lines = []
    for bench, k in PLAN.items():
        ctx = {d["id"]: d["context"] for d in json.load(
            open(UP / f"experiments/{bench}/large_scale/output/large_scale-keyword-0.5.json",
                 encoding="utf-8"))}
        qa = {q["id"]: q for q in json.load(
            open(UP / f"experiments/{bench}/large_scale/qa-pairs/qa-pairs.json",
                 encoding="utf-8"))}
        ob, cb, cov = open_book(bench), closed_book(bench), coverage(bench)
        ids = [i for i in qa if i in ob and i in cb and i in cov and i in ctx]
        useFail = [i for i in ids if not cb[i] and not ob[i] and cov[i]]
        unans = [i for i in ids if not cb[i] and not ob[i] and not cov[i]]
        random.shuffle(useFail); random.shuffle(unans)
        for bucket, pool in [("useFail(ent=present)", useFail[:k]),
                             ("unans(ent=absent)", unans[:k])]:
            for qid in pool:
                q = qa[qid]
                gold = qp.get_gold(q)
                gold_strs = qp._gold_candidates(gold) if hasattr(qp, "_gold_candidates") else [str(gold)]
                lines.append("=" * 90)
                lines.append(f"BENCH={bench}  BUCKET={bucket}  ID={qid}")
                lines.append(f"COVERAGE_FLAG all_entities_present={cov[qid]}")
                lines.append(f"Q: {q['question']}")
                lines.append(f"GOLD: {gold}")
                lines.append("-" * 40)
                lines.append(excerpt(ctx[qid], [str(s) for s in gold_strs]))
                lines.append("")
    out = HERE / "_spotcheck.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    n = sum(PLAN.values()) * 2
    print(f"wrote {out}  ({n} cases)  {len(''.join(lines))} chars")


if __name__ == "__main__":
    main()
