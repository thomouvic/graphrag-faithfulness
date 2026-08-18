"""
Strip credit-error-poisoned rows from probe JSONL checkpoints so the resumable
runners will pick them up on the next invocation.

Detection: any row whose `sc_norms` joined string contains "code 402" or
"insufficient credits", or whose `pred` / `final_pred` field starts with
"ERROR:" or contains the same OpenRouter credit-failure phrasing.

Writes the cleaned JSONL back to the same path (with a .bak backup of the
original alongside).

Usage:
    python probe_unpoison.py
"""
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent

TARGETS = [
    REPO / "probe_results"  / "lightrag_sparql_sc.jsonl",
    REPO / "probe_results"  / "lightrag_ircot.jsonl",
    REPO / "probe_musique" / "lightrag_sparql_sc.jsonl",
    REPO / "probe_musique" / "lightrag_ircot.jsonl",
]


def is_polluted(r: dict) -> bool:
    sc_norms = r.get("sc_norms", [])
    joined = " ".join(sc_norms) if isinstance(sc_norms, list) else str(sc_norms)
    pred = (r.get("pred", "") or "") + (r.get("final_pred", "") or "")
    blob = (joined + " " + pred).lower()
    if "code 402" in blob: return True
    if "insufficient credits" in blob: return True
    if pred.startswith("ERROR:"): return True
    return False


def main():
    for path in TARGETS:
        if not path.exists():
            print(f"[skip] {path} (not found)")
            continue
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy(path, backup)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        clean = [r for r in rows if not is_polluted(r)]
        n_removed = len(rows) - len(clean)
        with open(path, "w", encoding="utf-8") as f:
            for r in clean:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {path.relative_to(REPO)}: kept {len(clean)} / {len(rows)}, removed {n_removed}  "
              f"(backup at {backup.name})")


if __name__ == "__main__":
    main()
