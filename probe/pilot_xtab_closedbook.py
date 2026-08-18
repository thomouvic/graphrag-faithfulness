"""
Build the deployment-relevant cross-tab for the 8B, per benchmark.

Three signals, joined per question:
  - closed-book verdict : 8B answers with NO context (regenerated here, same
    prompt as probe_closedbook.py). Correct => the model already knew it =>
    parametric LEAK (won't transfer to a private corpus).
  - coverage            : are the supporting entities in the retrieved context?
    (upstream all_entities_present; substring_covered also loaded.)
  - open-book verdict   : the published no-few-shot SPARQL-CoT baseline verdict
    from the upstream results_sparql_groq CSVs (8B).

Buckets (per question):
  LEAK         : closed-book correct.
  RAG_WIN      : non-leak AND open-book correct (context did the work).
  USE_FAIL     : non-leak AND open-book wrong AND entities present
                 (answerable from context, model failed) -> intervention target.
  UNANSWERABLE : non-leak AND open-book wrong AND entities absent
                 (facts not retrieved) -> right behavior is abstention.

Closed-book answers are cached to probe/_closedbook_<bench>_8b.jsonl (resumable).
Uses the Paper 1 base repo .env (GROQ_API_KEY).
"""
import os, sys, json, csv, glob
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
BENCHES = ["hotpotqa", "musique", "2wikimultihopqa"]


def ask_closed_book(question):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system",
                   "content": "Answer in 1-5 words. If you do not know, reply: I don't know"},
                  {"role": "user", "content": f"QUESTION: {question}\nANSWER:"}],
        max_tokens=20, temperature=0.0)
    t = r.choices[0].message.content.strip()
    for p in ("ANSWER:", "Answer:", "answer:"):
        if t.startswith(p):
            t = t[len(p):].strip()
    return t.split("\n")[0].strip().strip('"\'')


def open_book_verdicts(bench):
    f = sorted(glob.glob(str(UP / "experiments" / bench / "large_scale" /
                              "results_sparql_groq" / "*llama-3.1-8b*.csv")))[-1]
    return {str(r["id"]): r["eval_verdict"].strip().lower() == "correct"
            for r in csv.DictReader(open(f, encoding="utf-8"))}


def coverage(bench):
    out = {}
    for r in csv.DictReader(open(UP / "revision/coverage" / f"{bench}_coverage.csv",
                                 encoding="utf-8")):
        out[str(r["id"])] = {
            "ent": r["all_entities_present"].strip().lower() == "true",
            "substr": r["substring_covered"].strip().lower() == "true"}
    return out


def closed_book(bench):
    """Run (resumable) or load cached closed-book verdicts for all questions."""
    qa = json.load(open(UP / "experiments" / bench / "large_scale" /
                        "qa-pairs/qa-pairs.json", encoding="utf-8"))
    cache = HERE / f"_closedbook_{bench}_8b.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); done[str(r["id"])] = r["correct"]
    todo = [q for q in qa if str(q["id"]) not in done]
    if todo:
        print(f"  [{bench}] closed-book: {len(done)} cached, running {len(todo)}...")
        with open(cache, "a", encoding="utf-8") as fh:
            for i, q in enumerate(todo):
                pred = ask_closed_book(q["question"])
                v = qp.eval_once(client, MODEL, q["question"],
                                 qp.get_gold(q), pred)["verdict"].strip().lower()
                ok = v == "correct"
                done[str(q["id"])] = ok
                fh.write(json.dumps({"id": str(q["id"]), "pred": pred,
                                     "correct": ok}) + "\n")
                if (i + 1) % 50 == 0:
                    print(f"    {bench} {i+1}/{len(todo)}")
    return done


def main():
    print(f"{'bench':16}{'N':>5}{'leak':>7}{'RAGwin':>8}{'useFail':>9}"
          f"{'unans':>7}{'baseAcc':>9}{'deployAcc':>10}")
    print("-" * 80)
    for bench in BENCHES:
        cb = closed_book(bench)
        ob = open_book_verdicts(bench)
        cov = coverage(bench)
        ids = [i for i in cb if i in ob and i in cov]
        n = len(ids)
        b = {"LEAK": 0, "RAG_WIN": 0, "USE_FAIL": 0, "UNANS": 0}
        for i in ids:
            if cb[i]:
                b["LEAK"] += 1
            elif ob[i]:
                b["RAG_WIN"] += 1
            elif cov[i]["ent"]:
                b["USE_FAIL"] += 1
            else:
                b["UNANS"] += 1
        nonleak = n - b["LEAK"]
        base_acc = sum(ob[i] for i in ids) / n
        # deployment proxy: accuracy on non-leak questions (RAG_WIN / nonleak)
        deploy_acc = b["RAG_WIN"] / nonleak if nonleak else float("nan")
        print(f"{bench:16}{n:>5}{b['LEAK']/n:>7.0%}{b['RAG_WIN']/n:>8.0%}"
              f"{b['USE_FAIL']/n:>9.0%}{b['UNANS']/n:>7.0%}"
              f"{base_acc:>9.0%}{deploy_acc:>10.0%}")
    print("\nleak=knew-it-no-context  RAGwin=context-flipped-to-correct  "
          "useFail=facts-present-but-wrong(target)  unans=facts-not-retrieved")
    print("baseAcc=raw open-book acc (incl leak)  "
          "deployAcc=acc on non-leak questions only")


if __name__ == "__main__":
    main()
