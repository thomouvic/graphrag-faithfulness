"""
Two questions, one run, same MuSiQue answer-present sample (seed 0, N=120):

(1) SPARQL ablation: is the SPARQL scaffold load-bearing once we add the
    grounding+calibration instruction? Compare:
        G_sparql : grounding+calibration WITH the SPARQL Step-1/2/3 format
        G_plain  : grounding+calibration WITHOUT SPARQL (free-form brief reason)
    If they tie, SPARQL is cosmetic under the grounding regime.

(2) Bridge to the few-shot ladder: does in-domain few-shot work via OUR
    mechanism? Run the variant-A few-shot prompt and see if it moves the
    faithfulness profile (confident-wrong rate down, grounded-span rate up,
    abstention up) toward the 70B's profile, relative to the plain baseline.

Per-arm metrics: accuracy, abstain rate, confident-wrong rate, grounded-span
rate (answer is an exact normalized substring of the context), precision.
8B baseline and 70B references (answer-present MuSiQue) for context:
  8B confident-wrong ~82%, grounded-span(of wrong) ~31%; 70B abstain ~60%,
  grounded-span(of wrong) ~70%.
"""
import os, sys, json, csv, re, random
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
TEMP = 0.3
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Variant-A few-shot examples (pulled from the sibling probe_fewshot_sparql.py).
_src = (Path(__file__).resolve().parent / "probe_fewshot_sparql.py").read_text(encoding="utf-8")
FEW_SHOT = re.search(r'_FEW_SHOT_EXAMPLES = """(.*?)"""', _src, re.DOTALL).group(1)

_PARADISE = ('  SELECT ?answer WHERE {\n    ?x name "Paradise Creek" .\n'
             "    ?x tributaryOf ?y .\n    ?y tributaryOf ?answer .\n  }\n")


def _final(raw):
    m = re.findall(r"(?i)FINAL\s*ANSWER\s*:\s*(.+)", raw)
    if m and not m[-1].strip().startswith("<"):
        return m[-1].strip().split("\n")[0].strip()
    return raw


def _call(prompt, max_tokens=512):
    raw = qp.call_groq_chat(client, MODEL, [{"role": "user", "content": prompt}],
                            max_tokens=max_tokens, temperature=TEMP)
    return _final(raw), raw


def arm_baseline(q, c):
    return qp.answer_with_sparql_cot(client, MODEL, q, c, temperature=TEMP)


def arm_g_sparql(q, c):
    p = ("You are answering a multi-hop question using ONLY the provided context.\n\n"
         "Step 1: Write a simple SPARQL query (max 4 triple patterns, plain English\n"
         "predicates, NO URIs, NO FILTER, NO subqueries). Example:\n" + _PARADISE +
         "Step 2: Follow the chain step by step through the context. For EACH step,\n"
         "quote the exact entity or sentence FROM THE CONTEXT that supports it.\n"
         "Step 3: The FINAL ANSWER must be an entity name copied VERBATIM from the\n"
         "context. Be conservative: only answer when every hop is supported by a\n"
         "verbatim quote from the context. If you cannot fully support the answer\n"
         "with context evidence, write FINAL ANSWER: I don't know\n\n"
         f"CONTEXT:\n{c}\n\nQUESTION:\n{q}")
    return _call(p)


def arm_g_plain(q, c):
    p = ("You are answering a multi-hop question using ONLY the provided context.\n\n"
         "Think briefly, step by step, about which facts in the context connect the\n"
         "question to an answer, quoting the exact supporting text for each step.\n"
         "Your FINAL ANSWER must be an entity name copied VERBATIM from the context.\n"
         "Be conservative: only answer when the answer is supported by the context.\n"
         "If you cannot support an answer with context evidence, write\n"
         "FINAL ANSWER: I don't know\n\n"
         f"CONTEXT:\n{c}\n\nQUESTION:\n{q}")
    return _call(p)


def arm_fewshot(q, c):
    p = ("You are answering a multi-hop question using ONLY the provided context.\n"
         "First, study these examples of how to answer correctly.\n\n"
         f"{FEW_SHOT}"
         "Now answer the following question in the SAME format "
         "(Step 1 SPARQL -> Step 2 trace -> Step 3 FINAL ANSWER).\n"
         "If the answer is not in the context, write FINAL ANSWER: I don't know\n\n"
         f"CONTEXT:\n{c}\n\nQUESTION:\n{q}")
    return _call(p)


ARMS = [("baseline(SPARQL)", arm_baseline),
        ("G_sparql", arm_g_sparql),
        ("G_plain(no SPARQL)", arm_g_plain),
        ("fewshot-A(SPARQL)", arm_fewshot)]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", str(s).lower())).strip()


def main():
    bench = "musique"
    ctx = {d["id"]: d["context"] for d in json.load(
        open(UP / f"experiments/{bench}/large_scale/output/large_scale-keyword-0.5.json",
             encoding="utf-8"))}
    qa = {q["id"]: q for q in json.load(
        open(UP / f"experiments/{bench}/large_scale/qa-pairs/qa-pairs.json",
             encoding="utf-8"))}
    cov = {str(r["id"]): r["substring_covered"].strip().lower() == "true"
           for r in csv.DictReader(
               open(UP / f"revision/coverage/{bench}_coverage.csv", encoding="utf-8"))}
    pool = sorted(i for i in qa if i in ctx and cov.get(i))
    random.seed(0); random.shuffle(pool)
    sample = pool[:120]
    print(f"MuSiQue answer-present sample N={len(sample)}\n")

    rec = {name: {"corr": 0, "abst": 0, "cwrong": 0, "span": 0, "ans": 0}
           for name, _ in ARMS}
    for idx, qid in enumerate(sample):
        q, c = qa[qid]["question"], ctx[qid]
        gold = qp.get_gold(qa[qid])
        nctx = norm(c)
        for name, fn in ARMS:
            try:
                ans, _ = fn(q, c)
            except Exception as e:
                ans = f"ERROR {e}"
            v = qp.eval_once(client, MODEL, q, gold, ans)["verdict"].strip().lower()
            ab = qp.is_abstain(ans)
            r = rec[name]
            if v == "correct": r["corr"] += 1
            if ab: r["abst"] += 1
            else:
                r["ans"] += 1
                if v != "correct": r["cwrong"] += 1
                na = norm(ans)
                if na and na in nctx: r["span"] += 1
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(sample)} "
                  + " ".join(f"{n.split('(')[0]}={rec[n]['corr']/(idx+1):.2f}"
                             for n, _ in ARMS))

    n = len(sample)
    print(f"\n=== ARMS (MuSiQue answer-present, N={n}, 8B, T={TEMP}) ===")
    print(f"{'arm':22}{'acc':>7}{'abstain':>9}{'confWrong':>11}"
          f"{'spanRate':>10}{'prec':>7}")
    for name, _ in ARMS:
        r = rec[name]
        acc = r["corr"] / n
        ab = r["abst"] / n
        cw = r["cwrong"] / n
        span = r["span"] / r["ans"] if r["ans"] else float("nan")
        prec = r["corr"] / r["ans"] if r["ans"] else float("nan")
        print(f"{name:22}{acc:>7.3f}{ab:>9.2f}{cw:>11.2f}{span:>10.2f}{prec:>7.3f}")
    print("\nSPARQL ablation -> compare G_sparql vs G_plain.")
    print("Bridge -> compare baseline vs fewshot-A on confWrong/spanRate/abstain.")
    print("70B ref (answer-present): abstain~0.60 of-wrong, span-of-wrong~0.70; "
          "8B baseline of-wrong span~0.31, confident~0.82.")


if __name__ == "__main__":
    main()
