"""
8B single-shot SPARQL-CoT baseline on HotpotQA-LightRAG (the one missing
ingredient for the HotpotQA-LR diagnosis row in Table 1). T=0, resumable.

Saves probe/_hotpotqa_lr_baseline.jsonl {id, ok}. Then computes the diagnosis
row from leak (closed-book cache) + LR entity-coverage + this verdict.
"""
import os, sys, json, csv
from pathlib import Path

UP = Path(os.environ["PAPER1_REPO"])  # Paper 1 (arXiv 2603.14045) base repo
HERE = Path(__file__).parent
sys.path.insert(0, str(UP))
for line in (UP / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from groq import Groq
import qa_pipeline as qp

csv.field_size_limit(10_000_000)
MODEL = "llama-3.1-8b-instant"
client = Groq(api_key=os.environ["GROQ_API_KEY"])

qa = json.load(open(UP / "experiments/hotpotqa/large_scale/qa-pairs/qa-pairs.json",
                    encoding="utf-8"))
ctx = json.load(open(UP / "experiments/hotpotqa/large_scale_lightrag/lightrag_contexts.json",
                     encoding="utf-8"))
ctx = {str(k): v for k, v in ctx.items()}

out = HERE / "_hotpotqa_lr_baseline.jsonl"
done = {}
if out.exists():
    for l in out.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l); done[str(r["id"])] = r["ok"]

todo = [q for q in qa if str(q["id"]) not in done]
print(f"HotpotQA-LR baseline: {len(done)} done, {len(todo)} todo")
fh = open(out, "a", encoding="utf-8")
for i, q in enumerate(todo):
    qid = str(q["id"]); c = ctx.get(qid, "")
    try:
        pred, _ = qp.answer_with_sparql_cot(client, MODEL, q["question"], c, temperature=0.0)
        v = qp.eval_once(client, MODEL, q["question"], qp.get_gold(q), pred)["verdict"]
        ok = int(v.strip().lower() == "correct")
    except Exception as e:
        ok = 0
    done[qid] = ok
    fh.write(json.dumps({"id": qid, "ok": ok}) + "\n"); fh.flush()
    if (i + 1) % 50 == 0:
        print(f"  +{i+1}/{len(todo)} (total {len(done)}/{len(qa)})")
fh.close()

# ---- diagnosis row ----
leak = {str(json.loads(l)["id"]): json.loads(l)["correct"]
        for l in (HERE / "_closedbook_hotpotqa_8b.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
ent = {str(r["id"]): (r["all_entities_present"].strip().lower() == "true")
       for r in csv.DictReader(open(HERE / "_lightrag_coverage_hotpotqa.csv", encoding="utf-8"))}
ids = [str(q["id"]) for q in qa if str(q["id"]) in done and str(q["id"]) in leak and str(q["id"]) in ent]
n = len(ids)
bk = {"leak": 0, "RAGwin": 0, "useFail": 0, "unans": 0}; corr = 0
for i in ids:
    if done[i]: corr += 1
    if leak[i]: bk["leak"] += 1
    elif done[i]: bk["RAGwin"] += 1
    elif ent[i]: bk["useFail"] += 1
    else: bk["unans"] += 1
nonleak = n - bk["leak"]
print(f"\n=== HotpotQA-LR diagnosis (entity, 8B single-shot, n={n}) ===")
for k in ["leak", "RAGwin", "useFail", "unans"]:
    print(f"  {k:8} {round(100*bk[k]/n)}%")
print(f"  raw      {round(100*corr/n)}%")
print(f"  deploy   {round(100*bk['RAGwin']/nonleak)}%")
