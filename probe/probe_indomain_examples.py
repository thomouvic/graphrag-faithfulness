"""
In-domain few-shot example pool + selection strategies.

This module is the engine behind the "better few-shot examples" study
(Phase 7). It builds a pool of worked SPARQL-CoT examples from the *train
splits* of the benchmarks (never the eval samples), audits the pool for
contamination against the eval set, and exposes several selection strategies
that form an improvement gradient:

    R1  GenericSelector        — the 3 fixed generic examples (current FS)
    R3  DomainStaticSelector   — fixed examples, domain-flavoured surface forms
    R4  TrainStaticSelector    — a fixed set sampled from the train pool
    R5  DynamicSelector        — per-question nearest-neighbour from the pool
    R6  StructureSelector       — per-question NN, filtered to matching
                                  (hop-count, question-type[, base-format])

(R0 = zero-shot and R2 = generic with a swept shot count are produced by the
existing probe with --n-shots, so they need no new selector here.)

Design constraints honoured:
  * Selection is LLM-free (lexical or sentence-transformers similarity), so the
    "1 LLM call / question" cost story is preserved.
  * The example pool is built offline from gold decompositions — examples are
    *templated*, not LLM-generated, so they are faithful and reproducible.
  * STDLIB-ONLY at import time. sentence-transformers is an optional backend
    selected explicitly; without it a deterministic lexical backend is used.
    This keeps the module unit-testable without the heavy probe environment.

The worked-example *format* mirrors probe_fewshot_sparql._FEW_SHOT_EXAMPLES
exactly (Step 1 SPARQL -> Step 2 trace -> Step 3 FINAL ANSWER) so the only
variable that changes across the gradient is *which* examples are shown.
"""
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Question metadata: hop count and type.
# ---------------------------------------------------------------------------

_MUSIQUE_HOP = re.compile(r"^(\d)hop")
_COMPARISON_CUES = (
    " or ", "earlier", "later", "older", "younger", "longer", "shorter",
    "larger", "smaller", "bigger", "more", "less", "first", "last",
    "same ", "both ", "which one", "who is older", "who was born",
)


def hop_count_from_qid(qid: str) -> int:
    """MuSiQue qids encode hop count ('3hop1__...'). Otherwise default to 2."""
    m = _MUSIQUE_HOP.match(qid or "")
    if m:
        return int(m.group(1))
    return 2


def classify_question_type(question: str) -> str:
    """Lightweight heuristic: 'comparison' vs 'bridge'.

    Deliberately self-contained (does not depend on qa_pipeline's classifier)
    so this module imports with stdlib only and stays unit-testable.
    """
    q = (question or "").lower()
    for cue in _COMPARISON_CUES:
        if cue in q:
            return "comparison"
    return "bridge"


# ---------------------------------------------------------------------------
# Text normalisation + similarity backends.
# ---------------------------------------------------------------------------

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "is", "are",
    "was", "were", "who", "what", "which", "where", "when", "whom", "whose",
    "did", "do", "does", "and", "or", "that", "this", "with", "from", "as",
    "it", "its", "his", "her", "their", "s",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> list:
    return [t for t in _norm(s).split() if t and t not in _STOP]


def _ngrams(s: str, n: int = 8) -> set:
    toks = _norm(s).split()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)} if len(toks) >= n else set()


class LexicalSimilarity:
    """Stdlib TF-IDF-ish cosine over question tokens. No third-party deps.

    Fit on the pool's question texts to get IDF weights, then score any query
    against any pool item. Deterministic — good for reproducible runs and for
    unit tests in a bare-python environment.
    """

    def __init__(self):
        self.idf = {}
        self._fitted = False

    def fit(self, docs):
        df = Counter()
        n = 0
        for d in docs:
            n += 1
            for t in set(_tokens(d)):
                df[t] += 1
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self._fitted = True
        return self

    def _vec(self, text):
        tf = Counter(_tokens(text))
        return {t: c * self.idf.get(t, 1.0) for t, c in tf.items()}

    def similarity(self, a, b):
        va, vb = self._vec(a), self._vec(b)
        if not va or not vb:
            return 0.0
        dot = sum(va[t] * vb.get(t, 0.0) for t in va)
        na = math.sqrt(sum(v * v for v in va.values()))
        nb = math.sqrt(sum(v * v for v in vb.values()))
        return dot / (na * nb) if na and nb else 0.0


class STSimilarity:
    """Optional sentence-transformers backend (cosine over MiniLM embeddings).

    Imported lazily so the module never hard-depends on sentence-transformers.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy
        self._model = SentenceTransformer(model_name)
        self._cache = {}

    def fit(self, docs):  # embeddings are query-independent; nothing to fit
        return self

    def _emb(self, text):
        if text not in self._cache:
            self._cache[text] = self._model.encode(text, normalize_embeddings=True)
        return self._cache[text]

    def similarity(self, a, b):
        ea, eb = self._emb(a), self._emb(b)
        return float(sum(x * y for x, y in zip(ea, eb)))


def make_similarity(backend: str):
    if backend == "st":
        return STSimilarity()
    if backend == "lexical":
        return LexicalSimilarity()
    raise ValueError(f"unknown similarity backend: {backend!r}")


# ---------------------------------------------------------------------------
# Train-split loaders -> normalised example records.
#
# A normalised record is:
#   {id, question, answer, type, hops,
#    steps:  [{"q": subquestion, "a": subanswer}, ...],   # reasoning chain
#    triples:[(subj, pred, obj), ...],                    # for SPARQL body
#    snippet: "short supporting-context string"}
#
# Loaders are defensive about field names so they tolerate minor format drift
# across dataset releases.
# ---------------------------------------------------------------------------


def _first(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _subquestion_to_pred(subq: str) -> str:
    """Turn a decomposition sub-question into a short plain-English predicate.

    'Who is the director of #1 ?' -> 'directorOf'
    'What country is X located in?' -> 'locatedIn'
    Heuristic and illustrative — examples only need to *teach the format*.
    """
    toks = _tokens(subq)
    # drop placeholder refs like '#1'
    toks = [t for t in toks if not t.startswith("#") and not re.fullmatch(r"\d+", t)]
    if not toks:
        return "relatedTo"
    # camelCase the two most contentful tokens
    head = toks[:2]
    pred = head[0] + "".join(w.capitalize() for w in head[1:])
    return pred


def _musique_record(rec: dict) -> dict:
    qid = str(_first(rec, "id", "_id", default=""))
    decomp = rec.get("question_decomposition", []) or []
    steps = [{"q": d.get("question", ""), "a": d.get("answer", "")} for d in decomp]
    triples = []
    prev = None
    for d in decomp:
        subj = prev if prev else (steps[0]["q"].split()[0] if steps else "?x")
        pred = _subquestion_to_pred(d.get("question", ""))
        obj = d.get("answer", "")
        triples.append((subj, pred, obj))
        prev = obj
    # short supporting snippet from supporting paragraphs
    sup = [p for p in rec.get("paragraphs", []) if p.get("is_supporting")]
    snippet = " ".join(p.get("paragraph_text", "")[:240] for p in sup[:2]).strip()
    return {
        "id": qid,
        "question": rec.get("question", ""),
        "answer": _first(rec, "answer", default=""),
        "type": classify_question_type(rec.get("question", "")),
        "hops": hop_count_from_qid(qid) or max(len(steps), 2),
        "steps": steps,
        "triples": triples,
        "snippet": snippet,
        "source": "musique",
    }


def _hotpot_record(rec: dict) -> dict:
    qid = str(_first(rec, "_id", "id", default=""))
    ctx = {title: sents for title, sents in rec.get("context", [])}
    sup_titles = []
    snippet_parts = []
    for title, idx in rec.get("supporting_facts", []):
        if title not in sup_titles:
            sup_titles.append(title)
        sents = ctx.get(title, [])
        if 0 <= idx < len(sents):
            snippet_parts.append(sents[idx])
    qtype = rec.get("type") or classify_question_type(rec.get("question", ""))
    return {
        "id": qid,
        "question": rec.get("question", ""),
        "answer": _first(rec, "answer", default=""),
        "type": "comparison" if qtype == "comparison" else "bridge",
        "hops": 2,
        "steps": [],  # HotpotQA train has no explicit decomposition
        "triples": [],
        "snippet": " ".join(snippet_parts[:3]).strip(),
        "source": "hotpotqa",
        "sup_titles": sup_titles,
    }


def _twowiki_record(rec: dict) -> dict:
    qid = str(_first(rec, "_id", "id", default=""))
    ev = rec.get("evidences", []) or []
    triples = [(e[0], _camel(e[1]), e[2]) for e in ev if len(e) == 3]
    ctx = {title: sents for title, sents in rec.get("context", [])}
    snippet_parts = []
    for title, idx in rec.get("supporting_facts", []):
        sents = ctx.get(title, [])
        if 0 <= idx < len(sents):
            snippet_parts.append(sents[idx])
    qtype = rec.get("type", "")
    return {
        "id": qid,
        "question": rec.get("question", ""),
        "answer": _first(rec, "answer", default=""),
        "type": "comparison" if "comparison" in str(qtype) else "bridge",
        "hops": max(len(triples), 2),
        "steps": [{"q": f"{s} {p}?", "a": o} for s, p, o in triples],
        "triples": triples,
        "snippet": " ".join(snippet_parts[:3]).strip(),
        "source": "2wikimultihopqa",
    }


def _camel(rel: str) -> str:
    toks = _tokens(rel)
    if not toks:
        return "relatedTo"
    return toks[0] + "".join(w.capitalize() for w in toks[1:])


_LOADERS = {
    "musique": _musique_record,
    "hotpotqa": _hotpot_record,
    "2wikimultihopqa": _twowiki_record,
}


def load_pool_json(path: Path) -> list:
    """Load a pre-built example pool (e.g. a teacher-harvested variant-C pool).

    Accepts either a bare list of records or {"benchmark": ..., "examples": [...]}.
    Records are expected to carry at least `question`, `answer`, `type`, and
    either a `rendered` worked example or triples/steps to render from.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    recs = data.get("examples", data) if isinstance(data, dict) else data
    out = []
    for r in recs:
        r.setdefault("type", classify_question_type(r.get("question", "")))
        r.setdefault("hops", hop_count_from_qid(str(r.get("id", ""))))
        r.setdefault("steps", [])
        r.setdefault("triples", [])
        r.setdefault("snippet", "")
        out.append(r)
    return out


def load_train_pool(path: Path, benchmark: str, limit: int = None) -> list:
    """Load a train split (.json list or .jsonl) into normalised records."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        raw = [json.loads(l) for l in text.splitlines() if l.strip()]
    else:
        raw = json.loads(text)
    fn = _LOADERS[benchmark]
    pool = []
    for r in raw:
        try:
            rec = fn(r)
        except Exception:
            continue
        if rec["question"] and rec["answer"]:
            pool.append(rec)
        if limit and len(pool) >= limit:
            break
    return pool


# ---------------------------------------------------------------------------
# Worked-example rendering (same format as the generic FS examples).
# ---------------------------------------------------------------------------


# Default cap on an example's CONTEXT block (~300-400 tokens), per plan.md §10
# ("default trimmed"). 1 token ~= 4 chars, so ~1600 chars.
MAX_CTX_CHARS = 1600


def render_example(rec: dict, idx: int, base_format: str = None,
                   max_ctx_chars: int = MAX_CTX_CHARS) -> str:
    """Render one normalised record as a Step1/Step2/Step3 worked example.

    base_format: optionally shape the CONTEXT block to mimic a base's context
    style — 'ket' = a chunk dump, 'lightrag' = an entity-relation summary.
    None = plain sentences (matches the generic examples).

    If the record carries a pre-rendered worked example (`rendered`, e.g. a
    teacher-harvested variant-C example whose Step-2 prose we keep verbatim),
    emit it directly with only the "Example N (type):" header normalised.
    """
    if rec.get("rendered"):
        body = re.sub(r"^Example\s*\d*\s*(\([^)]*\))?\s*:?\s*\n?", "",
                      rec["rendered"].strip(), count=1)
        label = rec.get("type", "bridge")
        return f"Example {idx} ({label}):\n{body}\n"

    label = rec.get("type", "bridge")
    ctx = _format_context(rec, base_format, max_ctx_chars=max_ctx_chars)

    # Build the SPARQL body from triples, falling back to steps.
    triples = rec.get("triples") or []
    if not triples and rec.get("steps"):
        prev = "?x0"
        for i, st in enumerate(rec["steps"]):
            triples.append((prev if i else f'"{_seed_entity(rec)}"',
                            _subquestion_to_pred(st["q"]),
                            st["a"] or f"?x{i+1}"))
            prev = f"?x{i+1}"
    sparql = _render_sparql(rec, triples)
    trace = _render_trace(rec, triples)
    ans = rec.get("answer", "")

    return (
        f"Example {idx} ({label}):\n"
        f"CONTEXT:\n{ctx}\n"
        f"QUESTION:\n{rec['question']}\n"
        f"Step 1: Write a simple SPARQL query (max 4 triple patterns, plain "
        f"English predicates, NO URIs, NO FILTER, NO subqueries).\n"
        f"{sparql}\n"
        f"Step 2: Trace each variable through the context.\n"
        f"{trace}\n"
        f"Step 3: FINAL ANSWER: {ans}\n"
    )


def _seed_entity(rec: dict) -> str:
    if rec.get("triples"):
        return rec["triples"][0][0]
    q = rec.get("question", "")
    caps = re.findall(r"[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*", q)
    return caps[0] if caps else "entity"


def _render_sparql(rec, triples) -> str:
    if not triples:
        return ("SELECT ?answer WHERE {\n  # (no gold decomposition available)\n}")
    var = "?answer"
    lines = ["SELECT %s WHERE {" % var]
    for s, p, o in triples[:4]:
        s_r = s if s.startswith("?") else f'"{s}"'
        o_r = o if (o.startswith("?")) else f'"{o}"'
        lines.append(f"  {s_r} {p} {o_r} .")
    lines.append("}")
    return "\n".join(lines)


def _render_trace(rec, triples) -> str:
    if rec.get("steps"):
        parts = []
        for st in rec["steps"]:
            if st.get("a"):
                parts.append(f"{st['q'].strip()} -> {st['a']}.")
        if parts:
            return " ".join(parts) + f" So the answer is {rec.get('answer','')}."
    if triples:
        chain = "; ".join(f"{s} {p} {o}" for s, p, o in triples[:4])
        return f"From context: {chain}. So the answer is {rec.get('answer','')}."
    return f"From context, the answer is {rec.get('answer','')}."


def _format_context(rec, base_format, max_ctx_chars: int = MAX_CTX_CHARS) -> str:
    snippet = rec.get("snippet") or ""
    if not snippet:
        # synthesise a minimal context from triples/steps
        if rec.get("triples"):
            snippet = ". ".join(f"{s} {p} {o}" for s, p, o in rec["triples"][:3]) + "."
        elif rec.get("steps"):
            snippet = ". ".join(f"{st['q']} {st['a']}" for st in rec["steps"][:3])
    if max_ctx_chars and len(snippet) > max_ctx_chars:
        snippet = snippet[:max_ctx_chars].rsplit(" ", 1)[0] + " ..."
    if base_format == "ket":
        return f"[chunk-1] {snippet}"
    if base_format == "lightrag":
        ents = []
        for s, p, o in (rec.get("triples") or [])[:3]:
            ents.append(f"  {s} -> {o}: {p}")
        rels = "\n".join(ents) if ents else f"  {snippet}"
        return f"=== Entities ===\n{rels}"
    return snippet


# ---------------------------------------------------------------------------
# Contamination audit.
# ---------------------------------------------------------------------------


def audit_contamination(pool: list, eval_qa: list, ngram: int = 8) -> tuple:
    """Drop train examples that overlap the eval set; report overlap stats.

    Drops a pool record if:
      * its qid is present in the eval set, OR
      * its normalised gold answer equals an eval gold answer AND it shares an
        8-gram of question text with that eval question (same fact, same phrasing).

    Returns (clean_pool, stats).
    """
    eval_ids = {str(q.get("id")) for q in eval_qa}
    eval_golds = Counter()
    eval_ngrams = set()
    for q in eval_qa:
        for g in _gold_list(q):
            eval_golds[_norm(g)] += 1
        eval_ngrams |= _ngrams(q.get("question", ""), ngram)

    clean, dropped_id, dropped_overlap = [], 0, 0
    for rec in pool:
        if rec["id"] in eval_ids:
            dropped_id += 1
            continue
        gnorm = _norm(rec["answer"])
        shares_ngram = bool(_ngrams(rec["question"], ngram) & eval_ngrams)
        if gnorm in eval_golds and shares_ngram:
            dropped_overlap += 1
            continue
        clean.append(rec)

    # residual question-ngram overlap across the cleaned pool (reportable number)
    pool_ngrams = set()
    for rec in clean:
        pool_ngrams |= _ngrams(rec["question"], ngram)
    residual = len(pool_ngrams & eval_ngrams)

    stats = {
        "pool_in": len(pool),
        "pool_clean": len(clean),
        "dropped_qid_match": dropped_id,
        "dropped_gold+ngram": dropped_overlap,
        "residual_question_%dgram_overlaps" % ngram: residual,
    }
    return clean, stats


def _gold_list(q: dict) -> list:
    out = []
    a = q.get("answer")
    if isinstance(a, str):
        out.append(a)
    elif isinstance(a, list):
        out.extend(a)
    aa = q.get("answers")
    if isinstance(aa, list):
        out.extend([x for x in aa if isinstance(x, str)])
    return [s for s in out if s and s.strip()]


# ---------------------------------------------------------------------------
# Selectors.  Each .select(question, qid, base, n_shots) -> example block str.
# ---------------------------------------------------------------------------


class _BaseSelector:
    name = "base"

    def select(self, question, qid, base, n_shots) -> str:
        raise NotImplementedError

    @staticmethod
    def _join(recs, base_format=None):
        return "".join(
            render_example(r, i + 1, base_format=base_format) + "\n"
            for i, r in enumerate(recs)
        )


class GenericSelector(_BaseSelector):
    """R1: the original 3 generic examples, truncated/repeated to n_shots."""
    name = "generic"

    def __init__(self):
        # imported lazily to avoid pulling qa_pipeline at module import
        from probe_fewshot_sparql import _FEW_SHOT_EXAMPLES
        self._block = _FEW_SHOT_EXAMPLES
        # split into the 3 individual examples for shot-count control
        self._examples = [
            "Example" + e for e in _FEW_SHOT_EXAMPLES.split("Example")[1:]
        ]

    def select(self, question, qid, base, n_shots) -> str:
        ex = self._examples[:n_shots] if n_shots else self._examples
        # renumber
        out = []
        for i, e in enumerate(ex):
            e = re.sub(r"^Example \d+", f"Example {i+1}", e)
            out.append(e)
        return "".join(out)


class StaticSelector(_BaseSelector):
    """R3/R4: a fixed example set, identical for every question.

    Built once from a candidate pool by taking the highest-quality example of
    each question type (comparison + bridge), so the fixed set still covers the
    type slots the generic set covered.
    """
    name = "static"

    def __init__(self, pool, base_format=None, seed_order=None):
        self.base_format = base_format
        # deterministic pick: longest reasoning chain per type (most instructive)
        by_type = {}
        for r in sorted(pool, key=lambda x: (-len(x.get("steps") or x.get("triples") or []), x["id"])):
            by_type.setdefault(r["type"], []).append(r)
        ordered = []
        # bridge first, then comparison, then fill from bridge — matches the
        # generic set's emphasis (2 bridge-ish + 1 comparison)
        for t in ("bridge", "comparison", "bridge"):
            bucket = by_type.get(t, [])
            for r in bucket:
                if r not in ordered:
                    ordered.append(r)
                    break
        # pad from anything remaining
        for r in pool:
            if r not in ordered:
                ordered.append(r)
        self._ordered = ordered

    def select(self, question, qid, base, n_shots) -> str:
        n = n_shots or 3
        recs = self._ordered[:n]
        return self._join(recs, base_format=self.base_format)


class DynamicSelector(_BaseSelector):
    """R5: per-question nearest neighbours from the pool by question similarity."""
    name = "dynamic"

    def __init__(self, pool, sim, base_format=None):
        self.pool = pool
        self.sim = sim.fit([r["question"] for r in pool])
        self.base_format = base_format

    def _rank(self, question):
        scored = [(self.sim.similarity(question, r["question"]), r) for r in self.pool]
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return scored

    def select(self, question, qid, base, n_shots) -> str:
        n = n_shots or 3
        recs = [r for _, r in self._rank(question)[:n]]
        return self._join(recs, base_format=self.base_format)


class StructureSelector(DynamicSelector):
    """R6: nearest neighbours *filtered* to matching (hop-count, type).

    Falls back to relaxing constraints (type-only, then unconstrained) if the
    matched bucket can't supply n_shots — so it never returns fewer than the
    dynamic selector would.  When `match_base_format` is set, the example
    CONTEXT blocks are also rendered in the target base's style.
    """
    name = "structure"

    def __init__(self, pool, sim, match_base_format=False):
        super().__init__(pool, sim, base_format=None)
        self.match_base_format = match_base_format

    def select(self, question, qid, base, n_shots) -> str:
        n = n_shots or 3
        want_hops = hop_count_from_qid(qid)
        want_type = classify_question_type(question)
        ranked = self._rank(question)

        def pick(pred):
            return [r for _, r in ranked if pred(r)]

        tiers = [
            lambda r: r["hops"] == want_hops and r["type"] == want_type,
            lambda r: r["type"] == want_type,
            lambda r: True,
        ]
        chosen, seen = [], set()
        for pred in tiers:
            for r in pick(pred):
                if r["id"] in seen:
                    continue
                chosen.append(r); seen.add(r["id"])
                if len(chosen) >= n:
                    break
            if len(chosen) >= n:
                break
        bf = base if self.match_base_format else None
        return self._join(chosen[:n], base_format=bf)


# ---------------------------------------------------------------------------
# Factory: build the selector for a given --example-source.
# ---------------------------------------------------------------------------


def build_selector(source: str, *, pool=None, sim_backend="lexical",
                   base_format=None, match_base_format=False):
    """source ∈ {generic, domain, teacher, train, dynamic, structure}.

    Paper variants (plan.md): generic=A, domain=B, teacher=C.
    'domain', 'teacher', and 'train' all use a fixed StaticSelector; the
    difference is the pool passed in (hand-written domain pool / teacher-harvested
    per-benchmark pool / raw train-split pool). 'generic' needs no pool.
    'dynamic'/'structure' are per-question selectors kept as appendix/future
    ablations (see ideas.md) — not part of the A/B/C headline ladder.
    """
    if source == "generic":
        return GenericSelector()
    if pool is None or len(pool) == 0:
        raise ValueError(f"source={source!r} requires a non-empty example pool")
    if source in ("domain", "teacher", "train"):
        return StaticSelector(pool, base_format=base_format)
    if source == "dynamic":
        return DynamicSelector(pool, make_similarity(sim_backend), base_format=base_format)
    if source == "structure":
        return StructureSelector(pool, make_similarity(sim_backend),
                                 match_base_format=match_base_format)
    raise ValueError(f"unknown example source: {source!r}")
