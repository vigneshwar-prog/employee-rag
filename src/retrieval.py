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


# ============================================================================
# LLM-ROUTER (Phase 3.5 — the upgrade from the playground finding)
# ----------------------------------------------------------------------------
# The rule-based detect_filter() above only matches EXACT stored values, so
# "Reyan" (partial) or "reyen" (typo) fell through to semantic and gave wrong
# answers. Fix: let the LLM READ the question and propose {route, field, value} —
# it handles partials, typos, and synonyms. BUT the LLM can hallucinate a value
# not in the data ("Vignesh" != "Vigneshwar B"), which would filter to 0 results.
# So we NEVER trust its value blindly: we validate it against the known lists and
# fuzzy-snap to the nearest real value, else fall back to semantic.
# ============================================================================
import json
import difflib

# Below this similarity ratio (0..1) we don't trust a fuzzy snap and use semantic.
FUZZY_SNAP_THRESHOLD = 0.6


def _router_system_prompt(known: dict[str, list[str]]) -> str:
    """Give the LLM the known vocabulary and ask for a strict JSON routing decision."""
    return (
        "You are a query router for an employee database.\n"
        f"Known managers: {known['manager']}\n"
        f"Known projects: {known['project']}\n"
        f"Known technologies: {known['technology']}\n\n"
        "Decide how to answer the user's question. Reply with ONLY a JSON object:\n"
        '{"route": "metadata" | "semantic" | "aggregate", '
        '"field": "name" | "manager" | "project" | "technology" | null, '
        '"value": "<the exact known value, fixing typos/partials to the closest known value>" | null}\n\n'
        "Use metadata when the question asks WHO/WHICH people match a specific "
        "manager/project/technology/person (it lists them).\n"
        "Use aggregate when the question asks HOW MANY / the COUNT / the TOTAL number of people "
        "(optionally within a manager/project/technology). Use field=null, value=null for a grand total.\n"
        "Use semantic for fuzzy/meaning questions with no specific known entity.\n"
        "Output only the JSON, nothing else."
    )


def _parse_router_json(text: str) -> dict | None:
    """Extract the JSON object from the LLM reply, tolerating stray prose/markdown."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def resolve_value(field: str, value: str, known: dict[str, list[str]]) -> str | None:
    """Guardrail: map the LLM's proposed value to a REAL stored value, or None.

    - exact (case-insensitive) match  -> that stored value
    - otherwise fuzzy-snap to the closest known value if similar enough
    - else None (caller falls back to semantic)
    """
    if not value or field not in known:
        return None
    candidates = known[field]
    # 1. exact, case-insensitive
    for c in candidates:
        if c.lower() == value.lower():
            return c
    # 2. fuzzy: closest known value above the threshold
    match = difflib.get_close_matches(value, candidates, n=1, cutoff=FUZZY_SNAP_THRESHOLD)
    return match[0] if match else None


def llm_route(question: str, known: dict[str, list[str]]):
    """Ask the LLM to route; return a PLAN dict, or None for semantic/fallback.

    Plan shapes:
      {"route": "metadata",  "field": <f>, "value": <real>}   -> list matching people
      {"route": "aggregate", "field": <f>, "value": <real>}   -> COUNT within that filter
      {"route": "aggregate", "field": None, "value": None}    -> COUNT everyone (grand total)

    Any failure (gate down, bad JSON, unvalidated value) returns None so the caller
    degrades gracefully to the rule-based detector / semantic search. As always, we
    NEVER trust the LLM's value blindly — it's resolved against the known lists first.
    """
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url="http://localhost:8080/v1", api_key="unused",
            model="claude-sonnet-5", temperature=0,
        )
        reply = llm.invoke(
            [("system", _router_system_prompt(known)), ("human", question)]
        ).content
    except Exception:
        return None  # gate unreachable -> let caller fall back

    decision = _parse_router_json(reply)
    if not decision:
        return None
    route = decision.get("route")
    if route not in ("metadata", "aggregate"):
        return None  # semantic (or unknown) -> caller falls back

    field, raw_value = decision.get("field"), decision.get("value")

    # Grand-total count: "how many employees in total?" — no field to resolve.
    if route == "aggregate" and not field:
        return {"route": "aggregate", "field": None, "value": None}

    # Otherwise resolve the value against real stored data (guardrail).
    if field == "name":
        real = resolve_value("name", raw_value, {"name": known["name"]})
    else:
        real = resolve_value(field, raw_value, known)
    if not real:
        return None  # couldn't tie it to real data -> fall back
    return {"route": route, "field": field, "value": real}


def retrieve_with_plan(question: str, store=None, k: int = 4, debug: bool = True, use_llm: bool = True):
    """Route the question and return (docs, plan).

    `plan` tells the caller WHAT happened so it can phrase the answer:
      {"route": "metadata",  "field", "value"}                 -> docs = matching people
      {"route": "aggregate", "field", "value", "count": int}   -> docs = matched people, count = how many
      {"route": "semantic"}                                    -> docs = top-k similar

    Routing order (each step degrades gracefully to the next):
      1. LLM router  -> understands partials/typos/synonyms, validated to a real value
      2. rule-based  -> exact-substring detector (works even if the gate is down)
      3. semantic    -> fuzzy fallback when nothing filterable is found
    """
    if store is None:
        store = get_store()

    known = _known_values(store)

    # 1. LLM router (primary) -> a plan dict. 2. rule-based detector (fallback) -> a tuple
    #    we normalize into a metadata plan.
    plan = None
    reason = ""
    if use_llm:
        plan = llm_route(question, known)
        if plan:
            reason = "LLM router"
    if plan is None:
        hit = detect_filter(question, known)
        if hit:
            plan = {"route": "metadata", "field": hit[0], "value": hit[1]}
            reason = "rule-based"

    # --- AGGREGATE: count, don't list. Python owns the number (LLM is bad at counting). ---
    if plan and plan["route"] == "aggregate":
        field, value = plan["field"], plan["value"]
        if field:  # count within a filter, e.g. manager == Dhruv Sharma
            results = store.get(where={field: value})
            docs = _get_result_to_docs(results)
            where_txt = f"where {field} == {value!r}"
        else:       # grand total: count everyone
            results = store.get()
            docs = _get_result_to_docs(results)
            where_txt = "ALL employees"
        plan["count"] = len(docs)
        if debug:
            print(f"  [route] AGGREGATE ({reason})  count {where_txt}  -> {plan['count']}")
        return docs, plan

    # --- METADATA: exact filter, list the people. ---
    if plan and plan["route"] == "metadata":
        field, value = plan["field"], plan["value"]
        results = store.get(where={field: value})
        docs = _get_result_to_docs(results)
        if debug:
            print(f"  [route] METADATA ({reason})  where {field} == {value!r}  "
                  f"-> {len(docs)} exact match(es)")
        return docs, plan

    # --- SEMANTIC fallback. ---
    docs = store.similarity_search(question, k=k)
    if debug:
        print(f"  [route] SEMANTIC search  (no filterable entity found)  -> top {len(docs)}")
    return docs, {"route": "semantic"}


def retrieve(question: str, store=None, k: int = 4, debug: bool = True, use_llm: bool = True):
    """Backwards-compatible wrapper: return just the docs (used by __main__ and older callers)."""
    docs, _plan = retrieve_with_plan(question, store=store, k=k, debug=debug, use_llm=use_llm)
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
