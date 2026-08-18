"""
Decisive accuracy test: does grounding(+re-ask) actually beat the baseline,
once we remove the generation noise that made earlier N=120 runs swing +-6pp?

Design:
  - temperature 0 (greedy => deterministic => zero run-to-run generation noise;
    remaining variance is across questions, captured by paired bootstrap).
  - full cell (N=500), so per-question noise averages out.
  - paired bootstrap 95% CI on each arm's accuracy delta vs baseline.

Arms (same questions):
  A  : baseline SPARQL-CoT (reference)
  Gp : grounding+calibration, NO SPARQL, NO re-ask
  C  : grounding+calibration (plain) + one focused re-ask when abstain/ungrounded

Also reports trustworthiness metrics (abstain, confident-wrong, grounded-span,
precision) and an answer-present / answer-absent breakdown.
"""
import os, sys, json, csv, re, random, argparse
from pathlib import Path

UP = Path(os.environ["PAPER1_REPO"])  # Paper 1 (arXiv 2603.14045) base repo
sys.path.insert(0, str(UP))
for line in (UP / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from groq import Groq
import qa_pipeline as qp

csv.field_size_limit(10_000_000)
MODEL = "llama-3.1-8b-instant"
TEMP = 0.0
client = Groq(api_key=os.environ["GROQ_API_KEY"])

STOP = set("the a an of to in for and or is was are were be by with on at as from "
           "into that this which who whose what".split())
def toks(s):
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower())
            if t not in STOP and len(t) > 1]
def grounded(pred, ctx_tokens, thr=0.5):
    pt = toks(pred)
    return bool(pt) and sum(1 for t in pt if t in ctx_tokens) / len(pt) >= thr
def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", str(s).lower())).strip()
def final(raw):
    m = re.findall(r"(?i)FINAL\s*ANSWER\s*:\s*(.+)", raw)
    if m and not m[-1].strip().startswith("<"):
        return m[-1].strip().split("\n")[0].strip()
    return raw
def call(prompt, mt=512):
    raw = qp.call_groq_chat(client, MODEL, [{"role": "user", "content": prompt}],
                            max_tokens=mt, temperature=TEMP)
    return final(raw)


def ask_baseline(q, c):
    a, _ = qp.answer_with_sparql_cot(client, MODEL, q, c, temperature=TEMP)
    return a

def ask_g_plain(q, c):
    return call(
        "You are answering a multi-hop question using ONLY the provided context.\n\n"
        "Think briefly, step by step, about which facts in the context connect the\n"
        "question to an answer, quoting the exact supporting text for each step.\n"
        "Your FINAL ANSWER must be an entity name copied VERBATIM from the context.\n"
        "Be conservative: only answer when the answer is supported by the context.\n"
        "If you cannot support an answer with context evidence, write\n"
        "FINAL ANSWER: I don't know\n\n"
        f"CONTEXT:\n{c}\n\nQUESTION:\n{q}")

def ask_reask(q, c):
    return call(
        "Re-read the CONTEXT carefully. The answer is very likely present in it.\n"
        "Identify the single entity, copied VERBATIM from the context, that best\n"
        "answers the QUESTION. In one or two sentences, trace the connection while\n"
        "quoting the context. Then write FINAL ANSWER: <verbatim entity>.\n"
        "Only if the answer is truly absent, write FINAL ANSWER: I don't know\n\n"
        f"CONTEXT:\n{c}\n\nQUESTION:\n{q}", mt=400)


def boot_ci(a, b, iters=10000, seed=1):
    """Paired bootstrap 95% CI for mean(b)-mean(a). a,b are 0/1 lists."""
    rnd = random.Random(seed)
    n = len(a)
    deltas = []
    for _ in range(iters):
        idxs = [rnd.randint(0, n - 1) for _ in range(n)]
        da = sum(a[i] for i in idxs) / n
        db = sum(b[i] for i in idxs) / n
        deltas.append(db - da)
    deltas.sort()
    return deltas[int(0.025 * iters)], deltas[int(0.975 * iters)]


ARM_KEYS = ["A_baseline", "Gp_ground", "C_ground+reask"]


def safe(fn, *a, default=""):
    try:
        return fn(*a)
    except Exception as e:
        return f"ERROR {e}"[:80] if default == "" else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="musique")
    ap.add_argument("--base", default="ket", choices=["ket", "lightrag"])
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    b = args.base

    if args.base == "lightrag":
        raw = json.load(open(UP / f"experiments/{args.bench}/large_scale_lightrag/"
                             "lightrag_contexts.json", encoding="utf-8"))
        ctx = {str(k): v for k, v in raw.items()}
    else:  # ket
        ctx = {str(d["id"]): d["context"] for d in json.load(
            open(UP / f"experiments/{args.bench}/large_scale/output/"
                 "large_scale-keyword-0.5.json", encoding="utf-8"))}
    qa = {str(q["id"]): q for q in json.load(
        open(UP / f"experiments/{args.bench}/large_scale/qa-pairs/qa-pairs.json",
             encoding="utf-8"))}
    # leak = closed-book correct (base-independent; cache on disk)
    cbpath = Path(__file__).parent / f"_closedbook_{args.bench}_8b.jsonl"
    leak = {str(json.loads(l)["id"]): json.loads(l)["correct"]
            for l in cbpath.read_text(encoding="utf-8").splitlines() if l.strip()}
    ids = [i for i in sorted(qa) if i in ctx][:args.n]
    outpath = Path(__file__).parent / f"_decisive_{args.bench}_{args.base}_rows.jsonl"
    done = set()
    if outpath.exists():
        for l in outpath.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(str(json.loads(l)["id"]))
    todo = [i for i in ids if i not in done]
    print(f"{b}: N={len(ids)}  done={len(done)}  todo={len(todo)}  (T=0, resumable)\n")

    fh = open(outpath, "a", encoding="utf-8")
    for j, qid in enumerate(todo):
        q, c = qa[qid]["question"], ctx[qid]
        gold = qp.get_gold(qa[qid])
        ctoks = set(toks(c)); nctx = norm(c)
        gstr = norm(qa[qid].get("answer") or "")
        present = bool(gstr) and gstr in nctx  # answer-in-context, this base
        outs = {"A_baseline": safe(ask_baseline, q, c),
                "Gp_ground": safe(ask_g_plain, q, c)}
        gp = outs["Gp_ground"]
        if qp.is_abstain(gp) or not grounded(gp, ctoks):
            r2 = safe(ask_reask, q, c)
            cpred = r2 if (not qp.is_abstain(r2) and grounded(r2, ctoks)) else (
                gp if not qp.is_abstain(gp) else "I don't know")
        else:
            cpred = gp
        outs["C_ground+reask"] = cpred

        row = {"id": qid, "present": present,
               "leak": bool(leak.get(qid, False))}
        for name, pred in outs.items():
            res = safe(lambda: qp.eval_once(client, MODEL, q, gold, pred),
                       default={"verdict": "incorrect"})
            v = (res["verdict"] if isinstance(res, dict) else "incorrect").strip().lower()
            ans = 0 if qp.is_abstain(pred) else 1
            row[name] = {"ok": int(v == "correct"), "ans": ans,
                         "span": int(ans and norm(pred) and norm(pred) in nctx)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        if (j + 1) % 25 == 0:
            print(f"  +{j+1}/{len(todo)} (total {len(done)+j+1}/{len(ids)})")
    fh.close()

    # ---- analyze from the full file (resume-safe) ----
    rows = [json.loads(l) for l in outpath.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    n = len(rows)
    SLICES = [("ALL (full cell)", rows),
              ("ANSWER-PRESENT", [r for r in rows if r["present"]]),
              ("GENUINE-RAG (non-leak & present) <- deployment",
               [r for r in rows if r["present"] and not r["leak"]])]
    print(f"\n=== DECISIVE ({b}, N={n}, T=0 greedy) ===")
    for label, sl in SLICES:
        m = len(sl)
        base = [r["A_baseline"]["ok"] for r in sl]
        print(f"\n-- {label}  (n={m}) --")
        for k in ARM_KEYS:
            arr = [r[k]["ok"] for r in sl]
            acc = sum(arr) / m if m else float("nan")
            if k == "A_baseline":
                print(f"   {k:16} acc={acc:.3f}   (reference)")
            else:
                lo, hi = boot_ci(base, arr)
                sig = "" if (lo <= 0 <= hi) else "  *REAL*"
                print(f"   {k:16} acc={acc:.3f}   delta={acc-sum(base)/m:+.3f} "
                      f"[{lo:+.3f},{hi:+.3f}]{sig}")
    print(f"\n-- trustworthiness (all {n}) --")
    print(f"   {'arm':16}{'abstain':>9}{'confWrong':>11}{'span':>7}{'prec':>7}")
    for k in ARM_KEYS:
        ans = sum(r[k]["ans"] for r in rows)
        corr = sum(r[k]["ok"] for r in rows)
        spn = sum(r[k]["span"] for r in rows)
        abst = (n - ans) / n
        cw = (ans - corr) / n
        span = spn / ans if ans else float("nan")
        prec = corr / ans if ans else float("nan")
        print(f"   {k:16}{abst:>9.2f}{cw:>11.2f}{span:>7.2f}{prec:>7.3f}")
    print(f"\n   * = 95% CI excludes 0. GENUINE-RAG slice is the private-corpus proxy.")
    print(f"   per-question rows saved -> {outpath}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
