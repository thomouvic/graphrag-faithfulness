"""
SUPERSEDED by pilot_decisive.py. This early version ran at T=0.3, N=120, with no
confidence intervals; its accuracy deltas (~+11/+12 pp) were within the run-to-run
noise of that setup. Use pilot_decisive.py (full 500, T=0 greedy, paired-bootstrap
CIs) for the accuracy claim. Kept only because it is currently the only grounded+
re-ask run covering all three benchmarks; retire once pilot_decisive is extended.

Confirming experiment: does a calibration-gated grounded re-ask recover the
8B's confident-wrong-but-answer-present failures on MuSiQue?

Three arms on the SAME sampled answer-present questions, same Groq judge:
  A baseline       : faithful repro of the published SPARQL-CoT prompt.
  B grounded-strict: SPARQL-CoT + verbatim-grounding + conservative abstain.
  C grounded+reask : B, then one focused re-extract when B abstains/ungrounded.

Reports accuracy, abstain rate, precision (correct | answered), and the
A->C confusion (wrong->right recovery vs right->wrong harm). Uses the upstream
Paper 1 base repo's qa_pipeline (prompt + eval) and .env (GROQ_API_KEY).
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
client = Groq(api_key=os.environ["GROQ_API_KEY"])

STOP = set("the a an of to in for and or is was are were be by with on at as from "
           "into that this which who whose what".split())
def toks(s):
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower())
            if t not in STOP and len(t) > 1]

def grounded(pred, context, thr=0.5):
    pt = toks(pred)
    if not pt:
        return False
    ctx = set(toks(context))
    return sum(1 for t in pt if t in ctx) / len(pt) >= thr

_PARADISE = ('  SELECT ?answer WHERE {\n    ?x name "Paradise Creek" .\n'
             "    ?x tributaryOf ?y .\n    ?y tributaryOf ?answer .\n  }\n")

def ask_grounded(q, c, temperature=0.3):
    prompt = (
        "You are answering a multi-hop question using ONLY the provided context.\n\n"
        "Step 1: Write a simple SPARQL query (max 4 triple patterns, plain English\n"
        "predicates, NO URIs, NO FILTER, NO subqueries). Example:\n" + _PARADISE +
        "Step 2: Follow the chain step by step through the context. For EACH step,\n"
        "quote the exact entity or sentence FROM THE CONTEXT that supports it.\n"
        "Step 3: The FINAL ANSWER must be an entity name copied VERBATIM from the\n"
        "context. Be conservative: only answer when every hop is supported by a\n"
        "verbatim quote from the context. If you cannot fully support the answer\n"
        "with context evidence, write FINAL ANSWER: I don't know\n\n"
        f"CONTEXT:\n{c}\n\nQUESTION:\n{q}"
    )
    raw = qp.call_groq_chat(client, MODEL, [{"role": "user", "content": prompt}],
                            max_tokens=512, temperature=temperature)
    m = re.findall(r"(?i)FINAL\s*ANSWER\s*:\s*(.+)", raw)
    ans = raw
    if m and not m[-1].strip().startswith("<"):
        ans = m[-1].strip().split("\n")[0].strip()
    return ans, raw

def ask_reask(q, c, temperature=0.6):
    prompt = (
        "Re-read the CONTEXT carefully. The answer is very likely present in it.\n"
        "Identify the single entity, copied VERBATIM from the context, that best\n"
        "answers the QUESTION. In one or two sentences, trace the connection while\n"
        "quoting the context. Then write FINAL ANSWER: <verbatim entity>.\n"
        "Only if the answer is truly absent, write FINAL ANSWER: I don't know\n\n"
        f"CONTEXT:\n{c}\n\nQUESTION:\n{q}"
    )
    raw = qp.call_groq_chat(client, MODEL, [{"role": "user", "content": prompt}],
                            max_tokens=400, temperature=temperature)
    m = re.findall(r"(?i)FINAL\s*ANSWER\s*:\s*(.+)", raw)
    ans = raw
    if m and not m[-1].strip().startswith("<"):
        ans = m[-1].strip().split("\n")[0].strip()
    return ans, raw

def verdict(q, gold, pred):
    return qp.eval_once(client, MODEL, q, gold, pred)["verdict"].strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="musique",
                    choices=["hotpotqa", "musique", "2wikimultihopqa"])
    ap.add_argument("--pool", default="full", choices=["full", "present"],
                    help="sample from the full cell or only answer-present")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    b = args.bench

    ctx = {d["id"]: d["context"] for d in json.load(
        open(UP / f"experiments/{b}/large_scale/output/large_scale-keyword-0.5.json",
             encoding="utf-8"))}
    qa = {q["id"]: q for q in json.load(
        open(UP / f"experiments/{b}/large_scale/qa-pairs/qa-pairs.json",
             encoding="utf-8"))}
    cov = {str(r["id"]): r["substring_covered"].strip().lower() == "true"
           for r in csv.DictReader(
               open(UP / f"revision/coverage/{b}_coverage.csv", encoding="utf-8"))}

    pool = sorted(i for i in qa if i in ctx and
                  (args.pool == "full" or cov.get(i)))
    random.seed(args.seed)
    random.shuffle(pool)
    sample = pool[:args.n]
    n_present = sum(1 for i in sample if cov.get(i))
    print(f"=== {b}  pool={args.pool} ({len(pool)})  sample={len(sample)}  "
          f"answer-present in sample={n_present} ===\n")

    rec = {"A": [], "B": [], "C": []}
    # stratified records: accuracy within answer-present vs answer-absent
    strat = {"present": {"A": [], "B": [], "C": []},
             "absent": {"A": [], "B": [], "C": []}}
    conf = {"RR": 0, "RW": 0, "WR": 0, "Wabs": 0, "WW": 0}
    abst = {"A": 0, "B": 0, "C": 0}
    for idx, qid in enumerate(sample):
        q, c = qa[qid]["question"], ctx[qid]
        gold = qp.get_gold(qa[qid])

        pa, _ = qp.answer_with_sparql_cot(client, MODEL, q, c, temperature=0.3)
        pb, _ = ask_grounded(q, c)
        # arm C: re-ask only when B abstains or is poorly grounded
        if qp.is_abstain(pb) or not grounded(pb, c):
            pc, _ = ask_reask(q, c)
            if qp.is_abstain(pc) or not grounded(pc, c):
                pc = pb if not qp.is_abstain(pb) else "I don't know"
        else:
            pc = pb

        va, vb, vc = verdict(q, gold, pa), verdict(q, gold, pb), verdict(q, gold, pc)
        s = "present" if cov.get(qid) else "absent"
        for k, v, p in [("A", va, pa), ("B", vb, pb), ("C", vc, pc)]:
            rec[k].append(v == "correct")
            strat[s][k].append(v == "correct")
            if qp.is_abstain(p):
                abst[k] += 1
        # A->C confusion
        a_ok, c_ok = va == "correct", vc == "correct"
        if a_ok and c_ok: conf["RR"] += 1
        elif a_ok and not c_ok: conf["RW"] += 1
        elif not a_ok and c_ok: conf["WR"] += 1
        elif not a_ok and qp.is_abstain(pc): conf["Wabs"] += 1
        else: conf["WW"] += 1

        if (idx + 1) % 10 == 0 or idx + 1 == len(sample):
            acc = lambda k: sum(rec[k]) / len(rec[k])
            print(f"  [{idx+1}/{len(sample)}] A={acc('A'):.2f} B={acc('B'):.2f} "
                  f"C={acc('C'):.2f}")

    n = len(sample)
    print(f"\n=== RESULTS ({b}, pool={args.pool}, 8B, n={n}) ===")
    for k, name in [("A", "baseline"), ("B", "grounded-strict"),
                    ("C", "grounded+reask")]:
        correct = sum(rec[k]); ab = abst[k]; answered = n - ab
        prec = correct / answered if answered else float("nan")
        pres = strat["present"][k]; absn = strat["absent"][k]
        ap = sum(pres) / len(pres) if pres else float("nan")
        aa = sum(absn) / len(absn) if absn else float("nan")
        print(f"  {name:16} acc={correct/n:.3f}  abstain={ab/n:.2f}  "
              f"prec(answered)={prec:.3f}   [present n={len(pres)} acc={ap:.3f} | "
              f"absent n={len(absn)} acc={aa:.3f}]")
    print(f"\n  A->C confusion: right->right={conf['RR']}  right->wrong={conf['RW']}  "
          f"wrong->right={conf['WR']}  wrong->abstain={conf['Wabs']}  "
          f"wrong->wrong={conf['WW']}")
    print(f"  net accuracy change A->C: {(sum(rec['C'])-sum(rec['A']))/n:+.3f}")


if __name__ == "__main__":
    main()
