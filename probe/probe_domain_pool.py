"""
Hand-written DOMAIN-FLAVOURED example pool for the R3 rung of the gradient.

These are crafted to look like the benchmark domains (Wikipedia biographies,
films, geography, compositional entity chains) — richer surface forms and real
(subject, predicate, object) triples — while remaining hand-authored generic
world knowledge, so they carry the same no-contamination guarantee as the
original generic examples.

R3's role in the gradient: hold example *count* and *structure* fixed but make
the surface forms domain-realistic. Comparing R3 vs R1 (generic) isolates
"does domain surface-form matter?"; comparing R4 (real train examples) vs R3
isolates "do genuine in-domain examples beat hand-flavoured ones?".

Records use the same normalised schema as probe_indomain_examples.load_train_pool.
"""

_DOMAIN_RECORDS = [
    # --- bridge (single) -------------------------------------------------
    {
        "id": "domain_bridge_1",
        "question": "Which novel, written by the author of A Study in Scarlet, "
                    "features the detective Sherlock Holmes returning at the "
                    "Reichenbach Falls?",
        "answer": "The Adventure of the Empty House",
        "type": "bridge",
        "hops": 2,
        "steps": [
            {"q": "Who wrote A Study in Scarlet?", "a": "Arthur Conan Doyle"},
            {"q": "Which Arthur Conan Doyle story is set at the Reichenbach Falls return?",
             "a": "The Adventure of the Empty House"},
        ],
        "triples": [
            ("A Study in Scarlet", "writtenBy", "Arthur Conan Doyle"),
            ("?work", "writtenBy", "Arthur Conan Doyle"),
            ("?work", "setting", "Reichenbach Falls return"),
        ],
        "snippet": "A Study in Scarlet is a detective novel by Arthur Conan "
                   "Doyle. The Adventure of the Empty House, by Doyle, depicts "
                   "Sherlock Holmes's return after the Reichenbach Falls.",
        "source": "domain",
    },
    # --- comparison ------------------------------------------------------
    {
        "id": "domain_comparison_1",
        "question": "Between the University of Bologna and the University of "
                    "Oxford, which was established earlier?",
        "answer": "University of Bologna",
        "type": "comparison",
        "hops": 2,
        "steps": [
            {"q": "When was the University of Bologna established?", "a": "1088"},
            {"q": "When was the University of Oxford established?", "a": "1096"},
        ],
        "triples": [
            ("University of Bologna", "establishedIn", "1088"),
            ("University of Oxford", "establishedIn", "1096"),
        ],
        "snippet": "The University of Bologna was established in 1088. The "
                   "University of Oxford traces teaching to 1096.",
        "source": "domain",
    },
    # --- multi-hop bridge (3-hop) ---------------------------------------
    {
        "id": "domain_bridge_3hop_1",
        "question": "What is the official currency of the country where the "
                    "headquarters of the company that produces the PlayStation "
                    "is located?",
        "answer": "Japanese yen",
        "type": "bridge",
        "hops": 3,
        "steps": [
            {"q": "Which company produces the PlayStation?", "a": "Sony"},
            {"q": "In which country is Sony headquartered?", "a": "Japan"},
            {"q": "What is the official currency of Japan?", "a": "Japanese yen"},
        ],
        "triples": [
            ("PlayStation", "producedBy", "Sony"),
            ("Sony", "headquarteredIn", "Japan"),
            ("Japan", "officialCurrency", "Japanese yen"),
        ],
        "snippet": "The PlayStation is produced by Sony, which is headquartered "
                   "in Japan. The official currency of Japan is the Japanese yen.",
        "source": "domain",
    },
    # --- multi-hop bridge (4-hop), for deep-hop matching ----------------
    {
        "id": "domain_bridge_4hop_1",
        "question": "What is the capital of the country whose national football "
                    "team won the FIFA World Cup hosted by the country where the "
                    "Eiffel Tower is located, in the year that tournament was held?",
        "answer": "Paris",
        "type": "bridge",
        "hops": 4,
        "steps": [
            {"q": "In which country is the Eiffel Tower located?", "a": "France"},
            {"q": "Which FIFA World Cup did France host?", "a": "1998 FIFA World Cup"},
            {"q": "Which national team won the 1998 FIFA World Cup?", "a": "France"},
            {"q": "What is the capital of France?", "a": "Paris"},
        ],
        "triples": [
            ("Eiffel Tower", "locatedIn", "France"),
            ("France", "hosted", "1998 FIFA World Cup"),
            ("1998 FIFA World Cup", "wonBy", "France"),
            ("France", "capital", "Paris"),
        ],
        "snippet": "The Eiffel Tower is in France. France hosted the 1998 FIFA "
                   "World Cup and won it. The capital of France is Paris.",
        "source": "domain",
    },
    # --- comparison (numeric/dates) -------------------------------------
    {
        "id": "domain_comparison_2",
        "question": "Who was born first, the composer of the Ninth Symphony "
                    "known as the Choral, or the composer of The Magic Flute?",
        "answer": "Wolfgang Amadeus Mozart",
        "type": "comparison",
        "hops": 3,
        "steps": [
            {"q": "Who composed the Ninth Symphony (Choral)?", "a": "Ludwig van Beethoven"},
            {"q": "Who composed The Magic Flute?", "a": "Wolfgang Amadeus Mozart"},
            {"q": "Who was born first, Beethoven (1770) or Mozart (1756)?",
             "a": "Wolfgang Amadeus Mozart"},
        ],
        "triples": [
            ("Ninth Symphony (Choral)", "composedBy", "Ludwig van Beethoven"),
            ("The Magic Flute", "composedBy", "Wolfgang Amadeus Mozart"),
            ("Ludwig van Beethoven", "bornIn", "1770"),
            ("Wolfgang Amadeus Mozart", "bornIn", "1756"),
        ],
        "snippet": "Beethoven composed the Choral Symphony and was born in 1770. "
                   "Mozart composed The Magic Flute and was born in 1756.",
        "source": "domain",
    },
    {
        "id": "domain_bridge_2",
        "question": "In which mountain range is the highest peak of the country "
                    "that borders both France and Italy to its north located?",
        "answer": "Alps",
        "type": "bridge",
        "hops": 2,
        "steps": [
            {"q": "Which country borders France and Italy and contains Monte Rosa?",
             "a": "Switzerland"},
            {"q": "In which range is Switzerland's highest peak?", "a": "Alps"},
        ],
        "triples": [
            ("Switzerland", "borders", "France"),
            ("Switzerland", "highestPeakRange", "Alps"),
        ],
        "snippet": "Switzerland borders France and Italy; its highest peaks lie "
                   "in the Alps.",
        "source": "domain",
    },
]


def domain_pool(benchmark: str = None) -> list:
    """Return the hand-written domain-flavoured pool (benchmark arg reserved
    for future per-benchmark sets; currently one shared multi-hop-QA pool)."""
    return [dict(r) for r in _DOMAIN_RECORDS]
