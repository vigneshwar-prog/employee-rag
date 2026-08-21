# Phase 9 — Schema-agnostic (dynamic) RAG (PLANNED, not yet implemented)

**Status:** planned & parked. Implement later.
**Goal (user's words):** *"Let the user give any Excel sheet with any employee details — dob, joining
date, years of experience, salary, grade, notes, mentor, anything — and my RAG still works even if I
change the data."*

**One-line framing (resume-ready):** upgrade the flagship from a **fixed 4-column** RAG to a
**schema-agnostic** RAG that *introspects any spreadsheet at load time*, auto-classifies each column, and
routes questions (semantic / exact-match / **numeric range** / count) without any hardcoded field names.

---

## Why this is a strong next phase (interview value)

Today the system *knows* the domain: name / manager / project / technology are baked into `data.py` and
`retrieval.py`. That's why it's accurate — but it's also brittle: change the sheet and it breaks. Making it
dynamic is a genuine engineering problem (schema inference, type classification, numeric filtering, prompt
generalization) — exactly the "handle arbitrary input robustly" signal senior interviewers look for. It
also sets up the *personal-assistant* project (arbitrary personal data) with the same machinery.

---

## What's hardcoded today (the three places that must become dynamic)

**1. `src/data.py`** — assumes exactly 4 named columns:
- `COL_NAME / COL_MANAGER / COL_PROJECT / COL_TECHNOLOGY`, `SHEET_NAME = "Employee Mapping"`
- `row_to_card()` writes one fixed sentence ("X is managed by Y, works on Z, specializing in W")
- `build_documents()` writes a fixed 4-key metadata dict
- fills blanks only for Project/Technology

**2. `src/retrieval.py`** — assumes the same 4 fields:
- `FILTERABLE_FIELDS = ["name","manager","project","technology"]`
- `FIELD_CUES` — hand-written intent phrases ("reports to" → manager, "knows" → technology)
- `_router_system_prompt()` — literally injects `Known managers / projects / technologies`
- routes are only `metadata / semantic / aggregate` — **no numeric/range filtering exists**

**3. Implicit** — the "identity" column is always `name`. With arbitrary data the row's label could be
anything (Employee Name, Emp ID, Full Name…).

---

## The core idea: **infer the schema at load time, drive everything from it**

Instead of constants, build a **schema profile** by introspecting the DataFrame once, and pass it through
`data.py` → `index.py` → `retrieval.py` → the router prompt.

### Step A — Schema inference (`src/schema.py`, NEW)
Read the sheet, and for **each column** decide a **role**:

| Role | Detection heuristic | Retrieval use |
|---|---|---|
| **identity** | first text column, or the one with all-unique values | the row label ("who is X", cited as the name) |
| **categorical** | text column with **low cardinality** (e.g. ≤ ~30 distinct, or distinct/rows < 0.5) — manager, project, grade, mentor, department | **exact metadata filter** (`where col == value`) + **aggregate/count** |
| **numeric** | pandas dtype is int/float, or parseable (salary, years_experience, age) | **range filter** (`>`, `<`, `between`) + aggregate (avg/min/max/sum) |
| **date** | parseable as datetime (dob, joining date) | range/relative filter ("joined after 2022", "over 5 years tenure") |
| **freetext** | high-cardinality text (notes, comments, bio) | **semantic only** (embedded, not filtered) |

Output = a `SchemaProfile`: `{column_name: {role, dtype, distinct_values (if categorical), min/max (if numeric/date)}}`.
This is computed **from the data itself** — never hardcoded — so it can't drift.

### Step B — Dynamic cards & metadata (`src/data.py`)
- `row_to_card()` becomes generic: build a readable sentence/paragraph from **whatever columns exist**,
  e.g. *"{identity} — {col}: {val}; {col}: {val}; …"* (still prose-ish so it embeds well). Put freetext
  (notes) at the end so it's embedded verbatim.
- `metadata` dict = **every categorical + numeric + date column** (Chroma metadata must be scalar: str/
  int/float/bool — cast dates to ISO string or epoch; skip freetext to keep metadata lean).
- Blank-filling becomes generic: categoricals → "Unassigned", numerics → left null (don't fake a number).

### Step C — Schema-driven routing (`src/retrieval.py`)
- `FILTERABLE_FIELDS` and the "known values" come from the `SchemaProfile`, not constants.
- **Router prompt is generated from the schema:** inject the actual column list, each column's role, and
  (for categoricals) its distinct values — so the LLM routes against *this* sheet's real fields.
- **`FIELD_CUES` mostly retired:** with the schema in the prompt, the LLM maps "who earns the most" →
  `salary`, "5+ years" → `years_experience`, "reports to X" → whichever categorical looks like a manager.
  Keep a tiny optional synonym hint map, but it's no longer required for correctness.
- **NEW route: `range`** — `{route:"range", field, op:">"|"<"|"between", value(s)}`. Executed in Python
  over `store.get()` results (Chroma `where` supports `$gt`/`$lt` on numeric metadata — use it where
  possible, fall back to Python filtering). Reuse the **`resolve_value()` guardrail spirit**: validate the
  field exists and the value parses as a number/date before filtering; else fall back to semantic.
- **Aggregate generalizes:** not just count — `avg/min/max/sum` over a numeric column
  ("average salary in the MPLS project"), still **Python-owned** (LLMs can't be trusted to compute).

### Step D — Generation (`src/generate.py`) — mostly free
`generate.py` already consumes `retrieve_multi() → {plan, docs}` generically. Main change: the
answer-phrasing prompt should reference "the record" generically rather than assume employee fields, and
the count/aggregate phrasing should handle avg/min/max, not just count. No structural change.

### Step E — CLI/UX (`src/main.py`, `src/app.py`)
- On startup (or Streamlit upload), print/show the **inferred schema** ("Detected columns: salary
  [numeric], grade [categorical: A/B/C], notes [freetext]…") so the user *sees* what the system understood
  — a great demo moment and a debugging aid.
- **Streamlit:** add a **file-uploader** → user drops any .xlsx → we infer schema, rebuild the index, and
  chat against it. This is the headline "it's dynamic" demo.

---

## Files & changes (summary)

| File | Change |
|---|---|
| `src/schema.py` (NEW) | infer `SchemaProfile` from any DataFrame (roles + distinct values + numeric ranges) |
| `src/data.py` | generic `row_to_card()` + generic metadata dict, driven by the profile; generic blank-fill |
| `src/index.py` | pass the profile through; store it (e.g. alongside the collection) so retrieval can read it |
| `src/retrieval.py` | schema-driven known-values + router prompt; **new `range` route**; generalized aggregate (avg/min/max/sum); guardrail validates numeric/date parsing |
| `src/generate.py` | generic record phrasing; aggregate phrasing covers avg/min/max |
| `src/main.py` / `src/app.py` | show inferred schema; Streamlit **file-upload → reindex → chat** |
| `docs/*` | learnings §16 "Schema-agnostic RAG", architecture update (new `range` route + schema-inference box), learning-qa lesson on schema inference & numeric filtering |

## New capabilities this unlocks (demo queries on an arbitrary sheet)
- *"Who earns more than 20 lakhs?"* → **range** filter on `salary`
- *"Average years of experience on the MPLS project"* → **aggregate(avg)** on `years_experience` within a categorical filter
- *"Who joined after 2022?"* → **date range** on `joining_date`
- *"List everyone with grade A"* → **metadata** filter on an auto-detected categorical (`grade`)
- *"Who mentors Vigneshwar?"* → **metadata** on an auto-detected `mentor` column
- *"What does Vigneshwar's note say?"* → **semantic** over the `notes` freetext

## Hard parts / decisions to make at implementation time
1. **Categorical vs freetext threshold** — cardinality cutoff (distinct count and/or distinct/row ratio).
   Make it a tunable constant; show the inferred role so the user can sanity-check.
2. **Numeric parsing of messy cells** — "20L", "20,00,000", "5 yrs" — need light normalization; document
   that we handle clean numeric columns first, messy ones best-effort.
3. **Chroma metadata typing** — must be scalar; dates → ISO string *and* an epoch int (so `$gt`/`$lt`
   works). Decide the convention.
4. **Ambiguous column mapping** — two categoricals could both look like "manager"; the schema-in-prompt
   approach lets the LLM pick, but log the choice in `--debug`.
5. **Identity detection** — pick the label column deterministically (first all-unique text column), with a
   manual override option.

## Backwards compatibility
The current `employees.xlsx` (name/manager/project/technology) must still work **unchanged** — it's just
one instance of the general case (3 categoricals + 1 identity, no numerics). Verify the existing test
questions ("who reports to Dhruv Sharma", "how many under Aditya", counts, compound, memory) all still pass
after the refactor. That regression suite is the safety net for the rewrite.

## Scope estimate
One new module (`schema.py`), meaningful edits to `data.py` + `retrieval.py` (the two schema-bound files),
light edits to `index.py`/`generate.py`/`main.py`, a Streamlit uploader, and docs. Bigger than Phase 8x —
this is a proper phase. Sequence it: **schema.py → data.py → index.py → retrieval.py (range route last) →
CLI/Streamlit → docs.** Keep each step green against the existing sheet before adding numeric/date power.
