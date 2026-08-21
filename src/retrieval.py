"""
Phase 3 (+ Phase 9) — Hybrid retrieval, now SCHEMA-AGNOSTIC.

The core learning of the project. A single semantic search can't answer
"who reports to Dhruv Sharma?" or "list everyone on MPLS" — those are EXACT,
COMPLETE set queries, not fuzzy similarity. So we combine strategies:

    SEMANTIC  (embeddings)   -> fuzzy, meaning-based ("who knows automation-ish?")
    METADATA  (Chroma where) -> exact, complete      ("everyone whose Manager == Dhruv")
    AGGREGATE (Python)       -> count / avg / min / max / sum over a column
    RANGE     (Chroma $gt/$lt, Phase 9) -> "salary > 20 lakhs", "joined after 2022"

PHASE 9 — what changed vs. the old fixed-4-column version
---------------------------------------------------------
Nothing about the ROUTES is hardcoded to a domain anymore. We load the persisted
SchemaProfile (index.get_schema) and drive everything from it:
  - the "known values" = every CATEGORICAL column's distinct values (not a fixed 4).
  - the LLM router PROMPT is generated from the schema (columns + roles + values),
    so the model maps "who earns the most" -> the numeric `salary` column, or
    "grade A" -> the categorical `Grade` column, with no per-domain code.
  - a NEW `range` route handles numeric/date comparisons the old system couldn't,
    and `aggregate` generalizes from count-only to avg/min/max/sum.

The guardrail philosophy is unchanged: we NEVER trust the LLM's proposed value
blindly — categoricals are snapped to a real stored value (fuzzy), numerics/dates
must parse, else we fall back to semantic.

Run it to try questions live:
    python src/retrieval.py
"""

from __future__ import annotations

import json
import difflib

from index import get_store, get_schema
from schema import CATEGORICAL, NUMERIC, DATE


# Below this similarity ratio (0..1) we don't trust a fuzzy snap and use semantic.
FUZZY_SNAP_THRESHOLD = 0.6

# Categorical placeholder we never want to match on.
SKIP_VALUES = {"Unassigned"}


# ============================================================================
# SCHEMA-DRIVEN "known values" (generalizes the old fixed-4 _known_values)
# ============================================================================
def _known_values(store, schema=None) -> dict[str, list[str]]:
    """{categorical_column: [distinct values]} — the vocabulary the router validates
    against. Read straight from the inferred schema (Phase 9). Falls back to reading
    the store's string metadata if no schema was persisted (older index).

    Also includes the identity column under its real name AND under the stable key
    "name", so name lookups keep working regardless of what the label column is called.
    """
    values: dict[str, list[str]]
    if schema is not None:
        values = {col: [v for v in vals if v not in SKIP_VALUES]
                  for col, vals in schema.distinct_values().items()}
    else:
        metas = store.get()["metadatas"]
        acc: dict[str, set] = {}
        for m in metas:
            for k, v in m.items():
                if k != "name" and isinstance(v, str) and v not in SKIP_VALUES:
                    acc.setdefault(k, set()).add(v)
        values = {k: sorted(vs, key=len, reverse=True) for k, vs in acc.items()}

    # The identity values live in metadata under "name" (stable key). Expose them so
    # the router can filter/lookup by the row label.
    names = sorted({m["name"] for m in store.get()["metadatas"] if m.get("name")},
                   key=len, reverse=True)
    values["name"] = names
    return values


# ============================================================================
# GENERIC rule-based fallback (used when the gate is down — no domain cues)
# ============================================================================
def detect_filter(question: str, known: dict[str, list[str]]) -> tuple[str, str] | None:
    """Return (column, value) if the question mentions a known categorical/identity
    value, else None. Domain-agnostic: just case-insensitive substring scan, most
    specific (identity/name) first, then the longest values. No hardcoded field cues."""
    q = question.lower()
    # identity/name first (most specific)
    ordered = (["name"] if "name" in known else []) + \
              [k for k in known if k != "name"]
    for field in ordered:
        for value in known[field]:
            if value and value.lower() in q:
                return field, value
    return None


# ============================================================================
# LLM ROUTER — now generated FROM the schema
# ============================================================================
def _router_system_prompt(schema, known: dict[str, list[str]]) -> str:
    """Describe THIS sheet's columns/roles/values to the LLM and ask for a strict
    JSON routing decision. Fully generic — works for any schema."""
    cat_lines = []
    for col in schema.categorical:
        vals = known.get(col, [])
        shown = vals[:25] + (["..."] if len(vals) > 25 else [])
        cat_lines.append(f'  - "{col}" (categorical) values: {shown}')
    num_lines = [f'  - "{c.name}" (numeric) range {c.minimum}..{c.maximum}'
                 for c in schema.columns.values() if c.role == NUMERIC]
    date_lines = [f'  - "{c.name}" (date)' for c in schema.columns.values() if c.role == DATE]
    cols_desc = "\n".join(cat_lines + num_lines + date_lines) or "  (none)"

    return (
        "You are a query router for a table of records.\n"
        f"The identity (label) column is \"{schema.identity}\"; refer to it as \"name\".\n"
        "Filterable columns and their types:\n"
        f"{cols_desc}\n\n"
        "Decide how to answer the user's question. Reply with ONLY a JSON object:\n"
        '{"route": "metadata" | "semantic" | "aggregate" | "range", '
        '"field": "<column name, or \\"name\\" for the identity>" | null, '
        '"value": "<categorical: closest known value, fixing typos/partials; '
        'range: the threshold as a plain number, or YYYY-MM-DD for a date>" | null, '
        '"op": ">" | "<" | ">=" | "<=" | "between" | null, '
        '"value2": "<upper bound if op==between>" | null, '
        '"agg": "count" | "avg" | "min" | "max" | "sum" | null}\n\n'
        "Route guide:\n"
        "- metadata: WHO/WHICH records match an exact categorical (or name) value; "
        "lists them. field=that column, value=the known value.\n"
        "- range: numeric/date comparison (greater/less/after/before/between). "
        "field=the numeric or date column, op=the comparator, value=the threshold.\n"
        "- aggregate: HOW MANY / AVERAGE / MIN / MAX / TOTAL. agg=the operation; "
        "field=the numeric column for avg/min/max/sum (null for a plain count); "
        "optionally value=a categorical value to scope the aggregate.\n"
        "- semantic: fuzzy/meaning questions with no specific column/value.\n"
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
    """Guardrail: map the LLM's proposed CATEGORICAL/name value to a REAL stored
    value, or None (caller falls back to semantic).
      - exact (case-insensitive) match  -> that stored value
      - otherwise fuzzy-snap to the closest known value if similar enough
    """
    if not value or field not in known:
        return None
    candidates = known[field]
    for c in candidates:
        if c.lower() == value.lower():
            return c
    match = difflib.get_close_matches(value, candidates, n=1, cutoff=FUZZY_SNAP_THRESHOLD)
    return match[0] if match else None


def _parse_threshold(value, schema, field) -> float | None:
    """Parse a range threshold into the numeric form we STORED in metadata.
    Numeric columns -> float. Date columns -> epoch seconds (matches _meta_value).
    Returns None if it can't parse -> caller falls back to semantic."""
    role = schema.columns[field].role if field in schema.columns else None
    if role == DATE:
        import pandas as pd
        dt = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(dt) else float(dt.timestamp())
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def llm_route(question: str, known: dict[str, list[str]], schema):
    """Ask the LLM to route; return a PLAN dict, or None for semantic/fallback.

    Plan shapes:
      {"route":"metadata",  "field", "value"}                          -> list matches
      {"route":"range",     "field", "op", "value"[, "value2"]}        -> numeric/date filter
      {"route":"aggregate", "agg":"count", "field"|None, "value"|None} -> count
      {"route":"aggregate", "agg":avg|min|max|sum, "field",
                            "scope_field", "scope_value"}               -> numeric aggregate
    Any failure returns None so the caller degrades to rule-based / semantic.
    """
    if schema is None:
        return None
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(base_url="http://localhost:8080/v1", api_key="unused",
                         model="claude-sonnet-5", temperature=0)
        reply = llm.invoke([("system", _router_system_prompt(schema, known)),
                            ("human", question)]).content
    except Exception:
        return None  # gate unreachable -> let caller fall back

    decision = _parse_router_json(reply)
    if not decision:
        return None
    route = decision.get("route")
    field = decision.get("field")

    # --- RANGE (numeric/date comparison) ---
    if route == "range":
        if field not in schema.columns or schema.columns[field].role not in (NUMERIC, DATE):
            return None
        op = decision.get("op")
        if op not in (">", "<", ">=", "<=", "between"):
            return None
        lo = _parse_threshold(decision.get("value"), schema, field)
        if lo is None:
            return None
        plan = {"route": "range", "field": field, "op": op, "value": lo}
        if op == "between":
            hi = _parse_threshold(decision.get("value2"), schema, field)
            if hi is None:
                return None
            plan["value2"] = hi
        return plan

    # --- AGGREGATE (count / avg / min / max / sum) ---
    if route == "aggregate":
        agg = decision.get("agg") or "count"
        raw_value = decision.get("value")
        if agg in ("avg", "min", "max", "sum"):
            if field not in schema.columns or schema.columns[field].role != NUMERIC:
                return None
            scope_field, scope_value = None, None
            if raw_value:  # optional categorical scope, e.g. "avg salary in MPLS"
                for cat in schema.categorical:
                    real = resolve_value(cat, raw_value, known)
                    if real:
                        scope_field, scope_value = cat, real
                        break
            return {"route": "aggregate", "agg": agg, "field": field,
                    "scope_field": scope_field, "scope_value": scope_value}
        # plain COUNT
        if not field:
            # LLM sometimes gives value but not field (e.g. field=null, value="Aditya
            # Sharma"). Recover the scope column by snapping the value to a categorical.
            if raw_value:
                for cat in schema.categorical:
                    real = resolve_value(cat, raw_value, known)
                    if real:
                        return {"route": "aggregate", "agg": "count", "field": cat, "value": real}
            if any(cue in question.lower() for cue in ("total", "all ", " all", "everyone",
                    "entire team", "whole team", "how many people", "how many employees",
                    "how many records", "how many rows")):
                return {"route": "aggregate", "agg": "count", "field": None, "value": None}
            return None  # underspecified -> fall back (don't silently count everyone)
        real = resolve_value(field, raw_value, known)
        if not real:
            return None
        return {"route": "aggregate", "agg": "count", "field": field, "value": real}

    # --- METADATA (exact categorical / name filter) ---
    if route == "metadata":
        real = resolve_value(field, decision.get("value"), known)
        if not real:
            return None
        return {"route": "metadata", "field": field, "value": real}

    return None  # semantic / unknown -> caller falls back


def retrieve_with_plan(question: str, store=None, k: int = 4, debug: bool = True,
                       use_llm: bool = True, schema=None):
    """Route the question and return (docs, plan). Degrades gracefully:
      1. LLM router (schema-driven) 2. generic rule-based 3. semantic."""
    if store is None:
        store = get_store()
    if schema is None:
        schema = get_schema()

    known = _known_values(store, schema)

    plan, reason = None, ""
    if use_llm and schema is not None:
        plan = llm_route(question, known, schema)
        if plan:
            reason = "LLM router"
    if plan is None:  # gate down / no schema -> generic substring detector
        hit = detect_filter(question, known)
        if hit:
            plan = {"route": "metadata", "field": hit[0], "value": hit[1]}
            reason = "rule-based"

    return _execute_plan(plan, question, store, k, debug, reason, schema)


def _execute_plan(plan, question: str, store, k: int, debug: bool, reason: str = "", schema=None):
    """Run ONE resolved plan -> (docs, plan). plan=None -> semantic fallback."""

    # --- AGGREGATE: count / avg / min / max / sum. Python owns the number. ---
    if plan and plan["route"] == "aggregate":
        agg = plan.get("agg", "count")
        if agg == "count":
            field, value = plan.get("field"), plan.get("value")
            results = store.get(where={field: value}) if field else store.get()
            docs = _get_result_to_docs(results)
            plan["count"] = len(docs)
            if debug:
                where_txt = f"where {field} == {value!r}" if field else "ALL records"
                print(f"  [route] AGGREGATE count ({reason})  {where_txt}  -> {plan['count']}")
            return docs, plan
        # numeric aggregate (avg/min/max/sum) over plan['field'], optional scope
        field = plan["field"]
        sf, sv = plan.get("scope_field"), plan.get("scope_value")
        results = store.get(where={sf: sv}) if sf else store.get()
        nums = [m[field] for m in results["metadatas"]
                if isinstance(m.get(field), (int, float)) and not isinstance(m.get(field), bool)]
        plan["agg_value"] = _aggregate(agg, nums)
        if debug:
            scope = f" in {sf}=={sv!r}" if sf else ""
            print(f"  [route] AGGREGATE {agg}({field}){scope} ({reason}) -> {plan['agg_value']}")
        return _get_result_to_docs(results), plan

    # --- RANGE: numeric/date comparison via Chroma $gt/$lt. ---
    if plan and plan["route"] == "range":
        field, op, val = plan["field"], plan["op"], plan["value"]
        results = store.get(where=_range_where(field, op, val, plan.get("value2")))
        docs = _get_result_to_docs(results)
        if debug:
            hi = f"..{plan.get('value2')}" if op == "between" else ""
            print(f"  [route] RANGE ({reason})  {field} {op} {val}{hi}  -> {len(docs)} match(es)")
        return docs, plan

    # --- METADATA: exact categorical / name filter, list the records. ---
    if plan and plan["route"] == "metadata":
        field, value = plan["field"], plan["value"]
        results = store.get(where={field: value})
        docs = _get_result_to_docs(results)
        if debug:
            print(f"  [route] METADATA ({reason})  where {field} == {value!r}  "
                  f"-> {len(docs)} exact match(es)")
        return docs, plan

    # --- SEMANTIC fallback. --- (use the sub-plan's own text when decomposed)
    q = (plan or {}).get("question", question)
    docs = store.similarity_search(q, k=k)
    if debug:
        print(f"  [route] SEMANTIC search  (no filterable entity found)  -> top {len(docs)}")
    return docs, {"route": "semantic", "question": q}


def _aggregate(agg: str, nums: list[float]):
    """Compute a numeric aggregate in Python (never trust the LLM to do math)."""
    if not nums:
        return None
    if agg == "avg":
        return round(sum(nums) / len(nums), 2)
    if agg == "min":
        return min(nums)
    if agg == "max":
        return max(nums)
    if agg == "sum":
        return sum(nums)
    return len(nums)


def _range_where(field, op, val, val2=None):
    """Build a Chroma `where` clause for a range comparison on numeric/date metadata."""
    if op == "between" and val2 is not None:
        lo, hi = sorted([val, val2])
        return {"$and": [{field: {"$gte": lo}}, {field: {"$lte": hi}}]}
    opmap = {">": "$gt", "<": "$lt", ">=": "$gte", "<=": "$lte"}
    return {field: {opmap.get(op, "$gt"): val}}


# ============================================================================
# QUERY DECOMPOSITION (Phase 8a) — unchanged logic, now schema-aware sub-routing
# ============================================================================
def _decompose_system_prompt() -> str:
    return (
        "You split a user's question into independent sub-questions for a table of records.\n"
        "If the question asks for SEVERAL independent things, return a JSON list of "
        "self-contained sub-questions. If it asks ONE thing, return a 1-element list.\n"
        "Each sub-question MUST stand alone: resolve shared verbs and pronouns so it "
        "reads on its own.\n"
        "Reply with ONLY a JSON list of strings, nothing else."
    )


def llm_decompose(question: str) -> list[str] | None:
    """Split a compound question into standalone sub-questions (length 1 for simple),
    or None on any failure so the caller falls back to the single path."""
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(base_url="http://localhost:8080/v1", api_key="unused",
                         model="claude-sonnet-5", temperature=0)
        reply = llm.invoke([("system", _decompose_system_prompt()),
                            ("human", question)]).content
    except Exception:
        return None
    start, end = reply.find("["), reply.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        subs = json.loads(reply[start:end + 1])
    except json.JSONDecodeError:
        return None
    subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
    return subs or None


def llm_route_many(question: str, known: dict[str, list[str]], schema) -> list[dict] | None:
    """Decompose, then route EACH sub-question with the schema-driven llm_route().
    Per-sub fallback mirrors the single path: llm_route -> detect_filter -> semantic."""
    subs = llm_decompose(question)
    if not subs or len(subs) == 1:
        return None
    plans = []
    for sub in subs:
        plan = llm_route(sub, known, schema) if schema is not None else None
        if plan is None:
            hit = detect_filter(sub, known)
            plan = {"route": "metadata", "field": hit[0], "value": hit[1]} if hit \
                else {"route": "semantic"}
        plan["question"] = sub
        plans.append(plan)
    return plans


def retrieve_multi(question: str, store=None, k: int = 4, debug: bool = True,
                   use_llm: bool = True) -> list[dict]:
    """Return a list of {"plan","docs"} sub-results (1 element for a simple question)."""
    if store is None:
        store = get_store()
    schema = get_schema()

    plans = None
    if use_llm and schema is not None:
        plans = llm_route_many(question, _known_values(store, schema), schema)

    if not plans:  # not compound (or gate down / no schema) -> single path
        docs, plan = retrieve_with_plan(question, store=store, k=k, debug=debug,
                                        use_llm=use_llm, schema=schema)
        return [{"plan": plan, "docs": docs}]

    if debug:
        print(f"  [decompose] {len(plans)} sub-questions")
    results = []
    for p in plans:
        docs, plan = _execute_plan(p, p["question"], store, k, debug, reason="sub", schema=schema)
        results.append({"plan": plan, "docs": docs})
    return results


def retrieve(question: str, store=None, k: int = 4, debug: bool = True, use_llm: bool = True):
    """Backwards-compatible wrapper: return just the docs."""
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
    schema = get_schema()
    if schema:
        print(schema.describe(), "\n")
    print("Hybrid retrieval — type a question ('q' to quit).\n")
    while True:
        question = input("Question: ")
        if question.lower() == "q":
            break
        docs = retrieve(question, store=store)
        for d in docs:
            m = d.metadata
            print(f"    - {str(m.get('name')):24s} | "
                  f"{ {k: v for k, v in m.items() if k != 'name'} }")
        print()
