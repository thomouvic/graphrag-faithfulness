"""
Independent-judge validation of the LightRAG Reask gain (mirrors pilot_judge_check
for KET-RAG, but on LightRAG contexts).

Re-generates the three arms on the LightRAG MuSiQue *deployment slice*
(present & non-leak, taken verbatim from the decisive rows so the slice is the
exact cell the paper reports) at T=0, saves the answer TEXT, and scores each
answer with BOTH judges:
  - 8B  (llama-3.1-8b-instant)   -- the original self-judge
  - 70B (llama-3.3-70b-versatile) -- independent

If Reask's gain over baseline survives the 70B judge (paired-bootstrap 95% CI
excludes 0), it is not an artifact of 8B judging 8B.

Resumable + crash-proof. Reuses the arm functions from pilot_decisive.
"""
import os, sys, json
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import pilot_decisive as pd  # sets up env, Groq client, arms, helpers
import qa_pipeline as qp

UP = pd.UP
client = pd.client
JUDGES = {"ok8": "llama-3.1-8b-instant", "ok70": "llama-3.3-70b-versatile"}
ARMS = ["A_baseline", "Gp_ground", "C_ground+reask"]
BENCH = "musique"


def judge(model, q, gold, pred):
    res = pd.safe(lambda: qp.eval_once(client, model, q, gold, pred),
                  default={"verdict": "incorrect"})
    v = (res["verdict"] if isinstance(res, dict) else "incorrect").strip().lower()
    return int(v == "correct")


def main():
    b = BENCH
    # LightRAG contexts (same source pilot_decisive --base lightrag used)
    raw = json.load(open(UP / f"experiments/{b}/large_scale_lightrag/"
                         "lightrag_contexts.json", encoding="utf-8"))
    ctx = {str(k): v for k, v in raw.items()}
    qa = {str(q["id"]): q for q in json.load(
        open(UP / f"experiments/{b}/large_scale/qa-pairs/qa-pairs.json",
             encoding="utf-8"))}

    # Deployment slice = present & non-leak, taken verbatim from the LR decisive
    # rows so this is the exact cell the paper reports (8B base/ground/reask =
    # 43.5/50.6/56.2). present/leak flags carried over for the analysis slices.
    dec = {str(json.loads(l)["id"]): json.loads(l)
           for l in (HERE / f"_decisive_{b}_lightrag_rows.jsonl").read_text(
               encoding="utf-8").splitlines() if l.strip()}
    dep_ids = [i for i in sorted(dec)
               if dec[i]["present"] and not dec[i]["leak"] and i in ctx]

    out = HERE / f"_judgecheck_{b}_lightrag_rows.jsonl"
    done = set()
    if out.exists():
        done = {str(json.loads(l)["id"])
                for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [i for i in dep_ids if i not in done]
    print(f"{b} LR deployment: N={len(dep_ids)} done={len(done)} todo={len(todo)}",
          flush=True)

    fh = open(out, "a", encoding="utf-8")
    for j, qid in enumerate(todo):
        q, c = qa[qid]["question"], ctx[qid]
        gold = qp.get_gold(qa[qid]); ctoks = set(pd.toks(c))
        preds = {"A_baseline": pd.safe(pd.ask_baseline, q, c),
                 "Gp_ground": pd.safe(pd.ask_g_plain, q, c)}
        gp = preds["Gp_ground"]
        if qp.is_abstain(gp) or not pd.grounded(gp, ctoks):
            r2 = pd.safe(pd.ask_reask, q, c)
            preds["C_ground+reask"] = r2 if (not qp.is_abstain(r2)
                and pd.grounded(r2, ctoks)) else (
                gp if not qp.is_abstain(gp) else "I don't know")
        else:
            preds["C_ground+reask"] = gp
        row = {"id": qid, "present": bool(dec[qid]["present"]),
               "leak": bool(dec[qid]["leak"])}
        for arm, p in preds.items():
            row[arm] = {"pred": p[:200],
                        "ok8": judge(JUDGES["ok8"], q, gold, p),
                        "ok70": judge(JUDGES["ok70"], q, gold, p)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        if (j + 1) % 25 == 0:
            print(f"  +{j+1}/{len(todo)} (total {len(done)+j+1}/{len(dep_ids)})",
                  flush=True)
    fh.close()

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    n = len(rows)
    print(f"\n=== LR JUDGE CHECK ({b}, deployment slice, N={n}, T=0) ===")
    for jk, jmodel in JUDGES.items():
        print(f"\n## judge = {jmodel}")
        base = [r["A_baseline"][jk] for r in rows]
        for arm in ARMS:
            arr = [r[arm][jk] for r in rows]
            acc = sum(arr) / n if n else float("nan")
            if arm == "A_baseline":
                print(f"   {arm:16} acc={acc:.3f} (ref)")
            else:
                lo, hi = pd.boot_ci(base, arr)
                sig = "  *REAL*" if not (lo <= 0 <= hi) else ""
                print(f"   {arm:16} acc={acc:.3f}  d={acc-sum(base)/n:+.3f} "
                      f"[{lo:+.3f},{hi:+.3f}]{sig}")
    print(f"\n  KEY: does the LR Reask gain survive the 70B judge? rows -> {out}")


if __name__ == "__main__":
    main()
