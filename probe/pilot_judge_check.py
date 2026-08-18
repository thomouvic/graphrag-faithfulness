"""
Independent-judge validation of the +9 (gates everything).

Re-generates the three arms on MuSiQue at T=0 (deterministic, so accuracy
reproduces pilot_decisive), saves the answer TEXT, and scores each answer with
BOTH judges:
  - 8B  (llama-3.1-8b-instant)  -- the original self-judge
  - 70B (llama-3.3-70b-versatile) -- independent

If C's gain over baseline survives the 70B judge, it's real; if it collapses,
the 8B judge was inflating it (form bias toward verbatim-span answers).

Resumable + crash-proof. Reuses the arm functions from pilot_decisive.
"""
import os, sys, json, csv, random
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import pilot_decisive as pd  # sets up env, Groq client, arms, helpers
import qa_pipeline as qp

UP = pd.UP
client = pd.client
JUDGES = {"ok8": "llama-3.1-8b-instant", "ok70": "llama-3.3-70b-versatile"}
ARMS = ["A_baseline", "Gp_ground", "C_ground+reask"]


def judge(model, q, gold, pred):
    res = pd.safe(lambda: qp.eval_once(client, model, q, gold, pred),
                  default={"verdict": "incorrect"})
    v = (res["verdict"] if isinstance(res, dict) else "incorrect").strip().lower()
    return int(v == "correct")


def main():
    b = "musique"
    ctx = {d["id"]: d["context"] for d in json.load(
        open(UP / f"experiments/{b}/large_scale/output/large_scale-keyword-0.5.json",
             encoding="utf-8"))}
    qa = {q["id"]: q for q in json.load(
        open(UP / f"experiments/{b}/large_scale/qa-pairs/qa-pairs.json",
             encoding="utf-8"))}
    cov = {str(r["id"]): r["substring_covered"].strip().lower() == "true"
           for r in csv.DictReader(
               open(UP / f"revision/coverage/{b}_coverage.csv", encoding="utf-8"))}
    leak = {str(json.loads(l)["id"]): json.loads(l)["correct"]
            for l in (HERE / f"_closedbook_{b}_8b.jsonl").read_text(
                encoding="utf-8").splitlines() if l.strip()}
    ids = [i for i in sorted(qa) if i in ctx]

    out = HERE / f"_judgecheck_{b}_rows.jsonl"
    done = set()
    if out.exists():
        done = {str(json.loads(l)["id"])
                for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [i for i in ids if i not in done]
    print(f"{b}: N={len(ids)} done={len(done)} todo={len(todo)}")

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
        row = {"id": qid, "present": bool(cov.get(qid)), "leak": bool(leak.get(qid, False))}
        for arm, p in preds.items():
            row[arm] = {"pred": p[:200],
                        "ok8": judge(JUDGES["ok8"], q, gold, p),
                        "ok70": judge(JUDGES["ok70"], q, gold, p)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        if (j + 1) % 25 == 0:
            print(f"  +{j+1}/{len(todo)} (total {len(done)+j+1}/{len(ids)})")
    fh.close()

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    SLICES = [("ALL", rows), ("PRESENT", [r for r in rows if r["present"]]),
              ("GENUINE-RAG", [r for r in rows if r["present"] and not r["leak"]])]
    print(f"\n=== JUDGE CHECK ({b}, N={n}, T=0) ===")
    for jk, jmodel in JUDGES.items():
        print(f"\n## judge = {jmodel}")
        for label, sl in SLICES:
            m = len(sl); base = [r["A_baseline"][jk] for r in sl]
            print(f"  -- {label} (n={m}) --")
            for arm in ARMS:
                arr = [r[arm][jk] for r in sl]; acc = sum(arr)/m if m else float("nan")
                if arm == "A_baseline":
                    print(f"     {arm:16} acc={acc:.3f} (ref)")
                else:
                    lo, hi = pd.boot_ci(base, arr)
                    sig = "  *REAL*" if not (lo <= 0 <= hi) else ""
                    print(f"     {arm:16} acc={acc:.3f}  d={acc-sum(base)/m:+.3f} "
                          f"[{lo:+.3f},{hi:+.3f}]{sig}")
    print(f"\n  KEY: does C's gain survive the 70B judge? rows -> {out}")


if __name__ == "__main__":
    main()
