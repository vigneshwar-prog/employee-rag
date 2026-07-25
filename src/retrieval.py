"""
Phase 3 — Hybrid retrieval.

The core learning of the project. A single semantic search can't answer
"who reports to Dhruv Sharma?" or "list everyone on MPLS" — those are EXACT,
COMPLETE set queries, not fuzzy similarity. So we combine two strategies:

    SEMANTIC  (embeddings)  -> fuzzy, meaning-based ("who knows automation-ish stuff?")
    METADATA  (Chroma where)-> exact, complete      ("everyone whose manager == Dhruv Sharma")

How we DECIDE which to use — and this works because our data is tiny and
categorical (7 managers, 9 real projects, 12 technologies, 97 names, all known):

    scan the question for any KNOWN value.
      - found a known manager/project/technology/name?  -> exact metadata filter
      - found nothing to filter on?                       -> semantic search

No ML classifier, no guessing. Just "does the question mention something we know?"

Run it to try questions live:
    python src/retrieval.py
"""

from index import get_store


# --- Which metadata field is a value's "home", and how we detect it ---------
# We scan the question for known values. Order matters: NAME first (most specific),
# then manager/project/technology. "Unassigned" is deliberately NOT detectable —
# nobody asks "who is on the Unassigned project", and it would false-match.
FILTERABLE_FIELDS = ["name", "manager", "project", "technology"]
SKIP_VALUES = {"Unassigned"}

# Intent cues: phrases that reveal WHICH field the user means, independent of the
# value. "who reports to Dhruv" and "Dhruv's team" both mean manager==Dhruv, even
# though "Dhruv Sharma" is ALSO a person's name. When a cue is present we prefer its
# field, so we don't mistake "reports to Dhruv" for a lookup of Dhruv's own record.
FIELD_CUES = {
    "manager": ["reports to", "report to", "reporting to", "managed by",
                "manager of", "'s team", "s team", "under ", "who does "],
    "project": ["project", "working on the", "on the"],
    "technology": ["technology", "tech", "skill", "specializ", "works with",
                   "knows", "expert in"],
}


def _known_values(store) -> dict[str, list[str]]:
    """Read the distinct values actually stored in Chroma, per field.

    We build the detector from the DATA itself (not a hand-typed list) so it can
    never drift out of sync with what's in the index.
    """
    metas = store.get()["metadatas"]
    values: dict[str, list[str]] = {}
    for field in FILTERABLE_FIELDS:
        distinct = {m[field] for m in metas if m.get(field) not in SKIP_VALUES}
        # Longest-first so "R&S (Routing & Switching)" is tried before a short token.
        values[field] = sorted(distinct, key=len, reverse=True)
    return values


def detect_filter(question: str, known: dict[str, list[str]]) -> tuple[str, str] | None:
    """Return (field, value) if the question mentions a known value, else None.

    Two-step so we get the RIGHT field, not just any field:
      1. If an intent cue is present (e.g. "reports to" -> manager), try to match a
         known value for THAT field first. This resolves the ambiguity where a
         person's name is also a manager's name ("who reports to Dhruv Sharma?").
      2. Otherwise, scan all fields (name first = most specific).
    Matching is case-insensitive substring, so "mpls" matches "MPLS".
    """
    q = question.lower()

    # Step 1 — cue-guided: does the phrasing point at a specific field?
    for field, cues in FIELD_CUES.items():
        if any(cue in q for cue in cues):
            for value in known[field]:
                if value.lower() in q:
                    return field, value

    # Step 2 — fallback: scan every field, most-specific (name) first.
    for field in FILTERABLE_FIELDS:
        for value in known[field]:
            if value.lower() in q:
                return field, value
    return None


def retrieve(question: str, store=None, k: int = 4, debug: bool = True):
    """Hybrid retrieve: pick metadata filter OR semantic search for this question.

    Returns the list of matched Documents. When debug=True, prints which path was
    taken and why — so the retriever's decision is never a black box.
    """
    if store is None:
        store = get_store()

    known = _known_values(store)
    hit = detect_filter(question, known)

    if hit is not None:
        field, value = hit
        # EXACT path: Chroma `where` returns ALL records with this value — complete,
        # no ranking, no "top-k cutoff". Perfect for "list everyone / who reports to".
        results = store.get(where={field: value})
        docs = _get_result_to_docs(results)
        if debug:
            print(f"  [route] METADATA filter  where {field} == {value!r}  "
                  f"-> {len(docs)} exact match(es)")
        return docs

    # SEMANTIC path: no known value mentioned -> fall back to fuzzy similarity.
    docs = store.similarity_search(question, k=k)
    if debug:
        print(f"  [route] SEMANTIC search  (no known value in question)  "
              f"-> top {len(docs)}")
    return docs


def _get_result_to_docs(results: dict):
    """store.get() returns parallel lists; rebuild lightweight Document-likes."""
    from langchain_core.documents import Document
    docs = []
    for content, meta in zip(results["documents"], results["metadatas"]):
        docs.append(Document(page_content=content, metadata=meta))
    return docs


if __name__ == "__main__":
    store = get_store()
    print("Hybrid retrieval — type a question ('q' to quit).")
    print("Try: 'who reports to Dhruv Sharma?'  vs  'who keeps the network reliable?'\n")
    while True:
        question = input("Question: ")
        if question.lower() == "q":
            break
        docs = retrieve(question, store=store)
        for d in docs:
            m = d.metadata
            print(f"    - {m['name']:22s} | mgr={m['manager']:16s} "
                  f"| proj={m['project']:16s} | {m['technology']}")
        print()
