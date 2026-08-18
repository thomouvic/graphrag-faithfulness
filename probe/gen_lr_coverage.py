"""
Generate entity-coverage (all required bridging entities present) for the
LightRAG contexts, mirroring the Paper 1 supporting_fact_coverage.py but pointing at
lightrag_contexts.json instead of the KET keyword-0.5 file. CPU-only, no API.

Validates the port by also re-deriving KET coverage and diffing all_entities_present
against the existing revision/coverage/{ds}_coverage.csv.

Outputs (gitignored data): probe/_lightrag_coverage_{ds}.csv
"""
import os, json, re, string, csv
from pathlib import Path

UP = Path(os.environ["PAPER1_REPO"])  # Paper 1 (arXiv 2603.14045) base repo
DATASET = UP / "HippoRAG" / "reproduce" / "dataset"
EXP = UP / "experiments"
OUT = Path(__file__).parent
csv.field_size_limit(10_000_000)

# ---- copied verbatim from supporting_fact_coverage.py ----
def squad_norm(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

def entity_present(entity, lower_ctx, norm_ctx):
    if not entity:
        return False
    if entity.lower().strip() in lower_ctx:
        return True
    e_norm = squad_norm(entity)
    return bool(e_norm and e_norm in norm_ctx)

def gold_answer_substring(answer, lower_ctx, aliases=None):
    for a in [answer] + (aliases or []):
        if a and a.lower().strip() in lower_ctx:
            return True
    return False

def hotpotqa_required_entities(item):
    titles = set()
    for title, _ in item["supporting_facts"]:
        clean = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
        if clean:
            titles.add(clean)
    return list(titles) + [item["answer"]]

def twiki_required_entities(item):
    out = set()
    for s, _, o in item.get("evidences", []):
        if s: out.add(s)
        if o: out.add(o)
    return list(out)

def musique_required_entities(item):
    return [st["answer"] for st in item["question_decomposition"] if st.get("answer")]
# ----------------------------------------------------------

def load_dataset(name):
    return json.load(open(DATASET / f"{name}.json", encoding="utf-8"))

def load_qa(ds):
    return json.load(open(EXP / ds / "large_scale/qa-pairs/qa-pairs.json", encoding="utf-8"))

def load_ctx(ds, base):
    if base == "lightrag":
        d = json.load(open(EXP / ds / "large_scale_lightrag/lightrag_contexts.json", encoding="utf-8"))
        return {str(k): v for k, v in d.items()}
    return {str(c["id"]): c["context"] for c in
            json.load(open(EXP / ds / "large_scale/output/large_scale-keyword-0.5.json", encoding="utf-8"))}

def compute(name, ds, extractor, base):
    full = {it.get("_id", it.get("id")): it for it in load_dataset(name)}
    qa = load_qa(ds); ctx = load_ctx(ds, base)
    rows = []
    for q in qa:
        qid = str(q["id"]); item = full.get(qid)
        if item is None:
            continue
        craw = ctx.get(qid, "") or ""
        norm_ctx = squad_norm(craw); lower_ctx = " ".join(craw.lower().split())
        ents = extractor(item)
        npres = sum(1 for e in ents if entity_present(e, lower_ctx, norm_ctx))
        rows.append({
            "id": qid, "question": q["question"], "gold": q["answer"],
            "n_entities": len(ents), "n_entities_present": npres,
            "all_entities_present": len(ents) > 0 and npres == len(ents),
            "substring_covered": gold_answer_substring(q["answer"], lower_ctx, q.get("answers", [])),
            "context_chars": len(craw),
        })
    return rows

CONFIGS = [("hotpotqa", "hotpotqa", hotpotqa_required_entities),
           ("musique", "musique", musique_required_entities),
           ("2wikimultihopqa", "2wikimultihopqa", twiki_required_entities)]


def main():
    # 1) validate the port: re-derive KET coverage, diff vs existing CSV
    print("=== VALIDATION: re-derived KET vs existing coverage CSV ===")
    for name, ds, ex in CONFIGS:
        try:
            mine = {r["id"]: r["all_entities_present"] for r in compute(name, ds, ex, "ket")}
        except Exception as e:
            print(f"  {ds}: KET recompute failed ({e})"); continue
        existing = {}
        p = UP / "revision/coverage" / f"{ds}_coverage.csv"
        if p.exists():
            for r in csv.DictReader(open(p, encoding="utf-8")):
                existing[str(r["id"])] = r["all_entities_present"].strip().lower() == "true"
        common = [i for i in mine if i in existing]
        agree = sum(1 for i in common if mine[i] == existing[i])
        print(f"  {ds:16} KET match: {agree}/{len(common)} "
              f"({'OK' if agree == len(common) else 'MISMATCH'})  my-cover={sum(mine.values())*100//max(len(mine),1)}%")

    # 2) generate LR coverage
    print("\n=== LightRAG entity-coverage ===")
    for name, ds, ex in CONFIGS:
        try:
            rows = compute(name, ds, ex, "lightrag")
        except FileNotFoundError as e:
            print(f"  {ds}: no LR contexts ({e})"); continue
        out = OUT / f"_lightrag_coverage_{ds}.csv"
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        n = len(rows)
        ent = sum(1 for r in rows if r["all_entities_present"])
        sub = sum(1 for r in rows if r["substring_covered"])
        print(f"  {ds:16} n={n}  entity-cover={100*ent//n}%  substring-cover={100*sub//n}%  -> {out.name}")


if __name__ == "__main__":
    main()
