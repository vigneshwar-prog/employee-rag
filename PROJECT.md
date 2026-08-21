# PROJECT.md — Employee RAG System

> **Purpose of this file:** Persistent context so we can resume across multiple Claude Code sessions
> without re-explaining. At the start of any session, read this file first.
> At the end of a session, update the **Progress Tracker** and **Session Log**.

---

## 🎯 Goal
Build a **RAG (Retrieval-Augmented Generation)** system as a Gen AI learning project.

**Use case:** Given an Excel sheet of team employees (name, role, what they're working on, and
other fields), ask natural-language questions like:
- "Who is working on the billing migration?"
- "What is Kunal doing right now?"
- "List everyone with Python skills."
- "Who should I ask about Kubernetes?"

The system retrieves relevant employee records and an LLM generates a grounded answer.

---

## 👤 Learner Profile
- **Goal:** Learn Gen AI by building end-to-end projects.
- **Style:** Step-by-step, understand each concept before moving on.
- **Level:** Beginner-to-intermediate in Gen AI; comfortable with Python basics.

## 🛠️ Tooling — how we work (two AI assistants, split by job)
- **GitHub Copilot (in-editor):** learner's side-channel — inline explanations, quick questions,
  and **note-taking**. Owns the learner's own notes (keep them in a separate `NOTES.md`, *not* PROJECT.md).
- **Claude Code (terminal):** owns **development + PROJECT.md + `src/`**. Builds each phase *with* the
  learner — explains the "why," pauses on the concept — rather than dumping finished files.
- **Rules to avoid drift:** (1) Only Claude edits PROJECT.md and `src/` so the source of truth stays
  coherent. (2) Don't hand-edit a file while Claude is writing to it. (3) `git commit` at every phase
  boundary so either tool's mistakes roll back one phase, not the whole project.

---

## 🧠 RAG Concepts to Learn (the "why")
| Concept | What it is | Why it matters here |
|---|---|---|
| **Embeddings** | Turn text into vectors capturing meaning | Lets us search employees by meaning, not keywords |
| **Chunking** | Split data into retrievable units | Here: 1 employee = 1 chunk (simple, clean) |
| **Vector store** | DB that finds nearest vectors | Fast semantic retrieval |
| **Retrieval** | Fetch top-k relevant chunks for a query | Feeds the LLM only what's relevant |
| **Augmentation** | Stuff retrieved context into the prompt | Grounds the answer in *your* data |
| **Generation** | LLM writes the final answer | Natural language response |
| **Grounding / citations** | Answer cites which rows it used | Trust + avoids hallucination |

---

## 🏗️ Architecture (simple version)
```
Excel (.xlsx)
   │  1. Load & clean (pandas)
   ▼
Row → text "card" per employee
   │  2. Embed each card
   ▼
Vector store (Chroma / FAISS)  ← stored locally
   │
User question ──3. embed──► similarity search ──top-k rows──┐
                                                            ▼
                              4. Build prompt (question + retrieved rows)
                                                            ▼
                                        5. LLM generates grounded answer
```

---

## 🧰 Tech Stack (CONFIRMED)
- **Language:** Python 3.10+
- **Data:** pandas + openpyxl (read Excel)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`, local, free)
- **Vector store:** ChromaDB (local, persistent)
- **LLM (generation):** **Claude running locally at `http://localhost:8080`** via its
  OpenAI-compatible endpoint (no API key / no cost).
- **Orchestration:** **LangChain** — loaders, vector store wrapper, retriever, chains.
- **Interface:** CLI first → Streamlit web UI later

### ⚠️ Key design insight (why not "pure" RAG)
Our data is **small (97 rows) and structured/categorical** (7 managers, 9 projects, 12 techs).
Many real questions are *filters*, not semantic search:
- "Who reports to Dhruv Sharma?" → exact match on `Manager`
- "List everyone on MPLS" → exact match on `Technology`
- "What is Vivaan working on?" → semantic/lookup on name

So we build a **hybrid retriever**: semantic search (embeddings) **+ metadata filtering**
(Chroma `where` clause on manager/project/technology). This is the biggest learning payoff and
gives correct answers to "list all" style questions that pure vector search gets wrong.

---

## 🪜 Step-by-Step Plan

### Phase 0 — Setup ✅
- [x] Create virtualenv (python3.13) + `requirements.txt`
      (`langchain`, `langchain-community`, `langchain-openai`, `langchain-chroma`,
       `chromadb`, `langchain-huggingface`, `sentence-transformers`, `pandas`, `openpyxl`, `streamlit`)
- [x] Verify Claude endpoint at `http://localhost:8080` (chat works; **embeddings 404 — not served**)
- [x] Copy Excel into `data/employees.xlsx`
- [x] Confirm embeddings via local `all-MiniLM-L6-v2` (384-dim) — works, incl. fully offline from cache

### Phase 1 — Data ✅  (source: 97 rows, cols = Employee Name, Manager, Project, Technology)
- [x] Load with pandas; inspect (97 rows, 4 cols confirmed) — `src/data.py`
- [x] Handle 6 rows with missing Project/Technology → fill with "Unassigned" (they're managers, no project)
- [x] Build one **Document per employee**: readable card (`page_content`) + `metadata`
      = {name, manager, project, technology}. Text cells whitespace-stripped so exact filters work.

### Phase 2 — Embeddings & Indexing ✅
- [x] Embed cards with `all-MiniLM-L6-v2` (HuggingFaceEmbeddings) — `src/index.py`
- [x] Persist to Chroma on disk (`./chroma_db`, collection `employees`); metadata stored alongside
- [x] Re-runnable `index.py` (stable ids = employee name → upsert, no duplicates)
- [x] Verified: 97 vectors, semantic search works, store reopens without re-embedding

### Phase 3 — Hybrid Retrieval ✅  ← core learning
- [x] Semantic retriever: embed question, top-k similarity search (fallback path)
- [x] Metadata filter: detect known managers/projects/techs/names in the question →
      Chroma `where={field: value}` for exact, COMPLETE "list all" answers
- [x] Intent cues ("reports to"→manager, "on the"→project, "knows"→technology) resolve
      the name-vs-manager ambiguity (e.g. "reports to Dhruv" = her 22-person team, not her record)
- [x] Debug mode: prints which route (METADATA vs SEMANTIC) was taken and why
- [x] Detector built from the DATA's own distinct values (never drifts); "Unassigned" excluded

### Phase 4 — Generation (Claude @ localhost:8080) ✅
- [x] Wire `ChatOpenAI(base_url="http://localhost:8080/v1", api_key="unused", model="claude-sonnet-5")`
- [x] Prompt template: system rules ("answer ONLY from records, else 'I don't know'") + cards + question
- [x] Return grounded answer **+ list of source employees** (anti-hallucination)
- [x] `src/generate.py`: retrieve → augment (stuff cards) → generate → {answer, sources}
- [x] Verified: manager/project lookups answer in plain English; "reports to Dhruv" lists all 22;
      salary question correctly returns "I don't know based on the data" (guardrail works)

### Phase 5 — Interface (CLI) ✅
- [x] CLI Q&A loop (`main.py`) — single entry point, opens store once,
      `--debug` flag reveals the routing decision, clean quit + gate-down error

> Note on numbering: the `X.5` items (Phase 3.5 router, 4.5 friendly answers, 4.6 relevance
> fix) are **sub-improvements inserted into an existing phase**, not new phases — they upgraded
> the retrieval/generation that Phases 3 & 4 built. The whole-number phases below are the roadmap.

### Phase 6 — Streamlit UI ✅
- [x] Browser front-end (`src/app.py`) on top of the SAME `answer()` engine: text box → answer + sources panel.
- [x] `@st.cache_resource` opens the store ONCE (survives Streamlit's per-interaction reruns); debug checkbox mirrors `--debug`; honors the citation fix (sources panel only when cited); friendly gate-down error.
- [x] Run with `streamlit run src/app.py`.

### Phase 7 — Aggregation route (count / how-many) ✅
- [x] Third route `"aggregate"` in `llm_route` (now returns a plan dict). `retrieve_with_plan()` computes
      `count = len(store.get(where=...))` in PYTHON (LLM never counts); `generate._answer_count()` hands the
      LLM the exact number to phrase warmly. Covers total + filtered ("how many under X / on MPLS").
      Verified: total→97 (bug was "4"), Aditya→12, MPLS→8; listing & semantic unchanged; graceful fallback holds.
- [ ] (Later) Full framework-based tool/function calling for richer ops beyond count.

### Phase 8 — Experiments & polish (learning extensions) ⏳
- [ ] Robust name matching: phonetic (Soundex/Metaphone) so "vicky"/"vignsh" resolve reliably.
- [ ] Compare pure-semantic vs hybrid on "list everyone on MPLS"
- [ ] Try different k; try a bigger embedding model
- [ ] Add logging + simple eval questions

---

## 📄 Actual Excel Schema (CONFIRMED)
Source file: `/Users/vigneshwar/Downloads/Employee_Manager_Project_Mapping.xlsx`
Sheet: **Employee Mapping** · **97 rows** · 4 columns.

| Column | Example | Notes |
|---|---|---|
| `Employee Name` | Ishan Sharma | 0 missing |
| `Manager` | Dhruv Sharma | 7 unique managers, 0 missing |
| `Project` | Morgan Stanley | 9 unique projects, **6 missing** → fill "Unassigned" |
| `Technology` | R&S (Routing & Switching) | 12 unique, **6 missing** → fill "Unassigned" |

**12 technologies:** Automation, Cloud Networking, Data Center Networking, MPLS,
Network Monitoring (NMS), Optics, R&S (Routing & Switching), SD-WAN,
SDA (Software Defined Access), Security (Firewall/VPN), Voice/UC (Collaboration), Wireless (WLAN).

Copy the file to: `data/employees.xlsx`

### Example questions the system should answer
- "Who reports to Dhruv Sharma?"  (metadata filter → Manager)
- "List everyone working on MPLS." (metadata filter → Technology)
- "What is Vivaan Sharma working on?" (lookup by name)
- "Who is on the Telstra project?" (filter → Project)
- "Which technologies does Dhruv's team cover?" (filter + summarize)

---

## 📊 Progress Tracker
- Phase 0 — Setup: ✅ Done
- Phase 1 — Data: ✅ Done
- Phase 2 — Embeddings & Indexing: ✅ Done
- Phase 3 — Retrieval: ✅ Done
- Phase 4 — Generation: ✅ Done
- Phase 3.5 — LLM Router upgrade: ✅ Done (learner's finding + idea)
- Phase 4.5 — Friendlier answers (grounding kept): ✅ Done (learner's idea)
- Phase 4.6 — Relevance fix (don't recite irrelevant records): ✅ Done (learner's finding)
- Phase 5 — Interface (CLI `main.py`): ✅ Done
- Phase 6 — Streamlit UI: ✅ Done (`src/app.py`)
- Phase 7 — Aggregation route (count/how-many): ✅ Done (learner-found bug)
- Phase 8a — Query decomposition (multi-entity questions): ✅ Done (learner-found bug)
- Phase 8b — Conversational memory (in-session history, CLI): ✅ Done (learner-found gap)
- Phase 9 — Schema-agnostic RAG (infer any sheet; range route + generalized aggregate): ✅ Done (learner's idea)
- Phase 8 — Experiments & polish: ⏳ Pending  ← next (query rewriting, BM25 8c planned)

Legend: ✅ Done · 🔄 In Progress · ⏳ Pending

---

## 🗒️ Session Log
- **2026-07-24 — Session 0:** Scaffolded project. Inspected the real Excel
  (`Employee_Manager_Project_Mapping.xlsx`, 97 rows, 4 cols). Confirmed stack:
  LangChain + Chroma + sentence-transformers + **local Claude @ localhost:8080**. Chose a
  **hybrid retriever** (semantic + metadata filter) because the data is small/structured.
  **Next:** Phase 0 — create requirements.txt, verify the localhost:8080 endpoint, copy Excel in.
- **2026-07-24 — Session 1 (Phase 0 ✅):** Built venv on **python3.13** (system 3.9 too old),
  installed deps, copied Excel to `data/employees.xlsx`, gate re-verified (chat OK).
  **Embeddings decision settled:** gate returns **404 on `/v1/embeddings`** on every path/model
  despite `/v1/models` advertising them — the gate is chat-only. Switched embeddings to local
  **`all-MiniLM-L6-v2`** (384-dim), confirmed working and loadable **fully offline from cache**.
  Added `langchain-huggingface` + `sentence-transformers` to requirements.txt.
  **Two environment quirks captured** (see next section) — needed only for the first download on a
  machine behind the corporate SSL-inspecting proxy; irrelevant at runtime once cached.
  **Next:** Phase 1 — load Excel with pandas, fill missing → "Unassigned", build one Document/employee.
- **2026-07-25 — Session 2 (Phases 1 & 2 ✅):** Standardized on **`.venv` (Python 3.14)**, removed the
  duplicate `venv/` (both had full deps). Recorded the **tooling split** (Copilot=notes, Claude=dev).
  **Phase 1** `src/data.py`: load + whitespace-strip + fill 6 missing Project/Tech → "Unassigned"
  (the blanks are managers with no delivery project); build 97 `Document`s (card + metadata).
  **Phase 2** `src/index.py`: embed cards with `all-MiniLM-L6-v2`, persist to `./chroma_db`
  (collection `employees`), stable ids = employee name (upsert, no dupes). Verified 97 vectors,
  semantic search returns NMS people for "keeps the network reliable" (no keyword match — meaning match),
  and the store reopens without re-embedding. Fixed paths to be **project-root-anchored** so scripts
  run from any directory. **Next:** Phase 3 — hybrid retrieval (semantic + metadata filter).
- **2026-07-25 — Session 3 (Phases 3, 4 & 3.5 ✅):**
  **Phase 3** `src/retrieval.py`: hybrid router — detect known value → exact Chroma `where`;
  else semantic. Intent cues resolve name-vs-manager ("reports to Dhruv" = her 22-person team).
  **Phase 4** `src/generate.py`: `ChatOpenAI`→gate reads retrieved cards, answers grounded + sources;
  anti-hallucination guardrail verified ("salary?" → "I don't know based on the data").
  **Learner-driven finding & fix (Phase 3.5):** learner discovered pure-exact routing fails on
  partial names / typos ("Reyan"→3 not 20) and that semantic can't do "list all" (said "all four"
  of 8 R&S). Learner proposed **LLM-as-router**; built it with a **validation guardrail**: LLM proposes
  `{route, field, value}` → value validated against known lists, **fuzzy-snapped** (difflib) if not exact,
  else semantic. Graceful degrade: LLM router → rule-based detector → semantic (works even if gate down).
  Verified: "reyen"→all 20, "R & S"→all 8, "vignesh"→snaps to "Vigneshwar B" (no 0-results), synonyms
  ("airtel account"→Airtel). Docs: added `docs/learnings.html` (playground revision log, cross-linked).
  **Next:** Phase 5 — interface (CLI, then optional Streamlit).
- **2026-07-25 — Session 4 (Phase 5 ✅):**
  **Phase 5** `src/main.py`: the front door. One entry point (`python src/main.py`) that opens the
  vector store ONCE and reuses it, with a `--debug` flag to reveal the routing decision per question.
  UX hardening: empty-input skip, clean quit on `q`/Ctrl-C/Ctrl-D (no traceback), and a friendly
  gate-down error ("Is the local Claude gate up? bash scripts/check_gate.sh") instead of a stack trace.
  Verified end-to-end: "who reports to Dhruv Sharma?" → METADATA route → 22 complete; "who keeps the
  network reliable?" → SEMANTIC → honest "I don't know based on the data" (guardrail, not a wrong guess).
  **Next:** Phase 6 (improvements) or the optional Streamlit UI.
- **2026-07-25 — Session 5 (Phase 4.5 — friendlier answers):**
  Learner noticed answers were correct but curt, and asked (a) can we make them warmer, and (b) instead
  of a flat "I don't know", can the LLM fall back to its own knowledge? **(a) yes, (b) deliberately NO** —
  a free LLM fallback would hallucinate fake employees (e.g. invent a "Balamurugan"), defeating RAG.
  Fix is **prompt-only** in `generate.py`: warmer tone + phrase the "no" helpfully (suggest spelling /
  search by manager·project·tech), while keeping every anti-hallucination rule ("Do NOT make up an answer
  from general knowledge"). Verified on 4 questions: Pari/Aditya now friendly; "1+1" and "Balamurugan"
  stay grounded (no invented facts). Documented as learnings.html section 9. **Lesson: tone lives in the
  prompt; grounding is non-negotiable.** **Next:** Phase 6 or optional Streamlit UI.
- **2026-07-25 — Session 6 (citations + relevance + deep-dive Q&A):**
  Two more learner-found bugs, both fixed prompt-/filter-only:
  (1) **Misleading sources** — semantic always returns k=4, so chitchat/"I don't know" cited 4 unrelated
  names. Fixed with `_cited_sources()` in generate.py (refusal guard + name-mention filter) and main.py
  omits the Sources line when empty. (2) **"vicky" relevance bug** (Phase 4.6) — for a meaningless name the
  LLM recited the random retrieved cards as "the employees I know about". Fixed with a prompt rule: records
  may be irrelevant, use only genuine matches, else treat as not-found and never recite the rest.
  Also answered deep-dive questions (documented in learnings.html §10–11): how the LLM routes (decides in
  words, Python executes via if/else — hand-built "tool calling"); security (only ~28 vocab labels sent to
  router, not the 97 names; gate is local; risk = going cloud); name matching is **difflib character
  similarity, NOT semantic**, with a 0.6 threshold that genuinely fails on heavy typos; and clarified the
  "vicky" case is a **relevance bug, not a breach or prompt injection**. **Next:** Phase 6 or Streamlit UI.
- **2026-07-25 — Session 7 (Phase 6 — Streamlit UI ✅):**
  Renumbered the roadmap first (Streamlit=6, aggregation=7, experiments=8) to clear numbering confusion;
  noted that X.5 items are sub-improvements, not phases. Built `src/app.py`: single Q&A box on the SAME
  `generate.answer()` engine, so all prior fixes carry over. Key Streamlit concept taught: the script
  reruns top-to-bottom on every interaction, so the slow store/embedding load is wrapped in
  `@st.cache_resource` to run ONCE (web equivalent of the CLI's "open store once"). Debug checkbox mirrors
  `--debug`; sources shown only when cited (honors the citation fix); friendly gate-down error. Verified:
  launches headless (HTTP 200, no traceback) and the imported engine answers correctly end-to-end.
  Run: `streamlit run src/app.py`. **Next:** Phase 7 — aggregation route (the count bug).
- **2026-07-25 — Session 8 (Phase 7 — Aggregation route ✅):**
  Fixed the learner-found count bug ("how many employees?" → said 4, real 97). Added a third route
  `"aggregate"` alongside metadata/semantic. `llm_route()` now returns a **plan dict** (so it can express
  count-everyone vs count-within-a-filter); `retrieve_with_plan()` computes `count = len(store.get(where=…))`
  in **Python** (the LLM never counts — that was the bug); `generate._answer_count()` hands the LLM the exact
  number to phrase warmly (learner chose "Python counts, LLM phrases"). Scope = single counts (total +
  filtered); group-by tables deferred to Phase 8. Verified: total→97, Aditya→12, MPLS→8; listing & semantic
  unchanged; graceful fallback holds (LLM off → counts degrade to semantic, no crash). Docs: learnings §13
(+ §13.1 "follow one question through the code" — 4-step trace + the "aggregate = a `len()`'d metadata filter" insight).
  **Lesson: Python owns the number, the LLM owns the words.** **Next:** Phase 8 — experiments & polish.
- **2026-07-25 — Session 9 (Phase 8a — Query decomposition ✅):**
  Playground testing surfaced that COMPOUND questions ("count people under Aditya AND who is Vivek Sharma?")
  only got half-answered — `llm_route()` returns ONE plan, so one intent was always dropped. Fixed with
  query decomposition: `llm_decompose()` splits into standalone sub-questions, `llm_route_many()` routes
  EACH through the **existing** `llm_route()`/`resolve_value()` guardrail (reuse, not reinvent), the
  aggregate/metadata/semantic tail of `retrieve_with_plan()` was lifted into `_execute_plan()` and called
  per sub-plan by new `retrieve_multi()`, and `generate._stitch()` combines the already-correct sub-answers
  into one warm reply (no new facts, no recount). A simple question decomposes to length-1 → unchanged
  single path (byte-for-byte). Also: (a) hardened the grand-total escape hatch in `llm_route` (only count
  everyone when the question reads like a total); (b) tone tweak in `SYSTEM_PROMPT` — only suggest
  "check the spelling" when NO record matched, so a correct fuzzy match no longer nags. Clarified the
  "Vihaan fabrication": "Vihaan Sharma" is a REAL manager (14 reports) — the earlier "no records" was a
  flaky false-negative, not a hallucinated count; decomposition makes resolution deterministic. Verified:
  compound→both answered, total→97, Dhruv→22, vignesh→Vigneshwar B (no nag), fallback (use_llm=False)→
  rule-based no crash. Docs: learnings §14. **Lesson: decompose → route each through the same guardrail →
  stitch; reuse beats reinvent.** **Next:** Phase 8 — experiments & polish.
- **2026-07-25 — Session 9b (Phase 8a follow-up — answer granularity):**
  Playground finding: asking for ONE field ("who is the manager of Vignesh?", "which technology does
  Vivek know?") returned the WHOLE card (mgr + project + tech), plus a chatty "let me know!" after every
  bullet. Diagnosed as a **presentation** bug, not retrieval/tool-calling: `format_cards()` feeds the full
  card and the prompt never said "answer only the field asked" — a tool would return the same card, so tool
  calling fixes nothing (contrast the count bug, which WAS computation → Python). Fix = prompt-only: two
  `SYSTEM_PROMPT` rules — (1) match granularity (one-field question → one field; full card only for
  "who is/tell me about X"); (2) trim tails to one closing offer for the whole reply. Kept the legitimate
  "did you mean…?" on a real fuzzy name match. Verified with the exact 4-part question + a "who is X" control.
  Docs: learnings §14.1. **Lesson: computation bugs → move work to Python; presentation bugs → fix the prompt.**
  **Next:** Phase 8 — experiments & polish.
- **2026-07-26 — Session 10 (Phase 8b — Conversational memory ✅):**
  Playground finding: no memory — "I am vignesh" then "whom am I?" → "I don't have info about you".
  Root cause: `build_messages()` built a fresh [system, human] list each turn; nothing stored/replayed
  prior turns. Fix (in-session history, CLI): `build_messages`/`answer`/`_answer_one` gain an OPTIONAL
  `history` kwarg (default None → stateless, so `app.py` is unchanged); history is slotted between the
  system rules and the current turn. `main.py`'s `ask_loop` owns a `history` list, appends (q, a) after
  each successful answer, caps to the last 6 turns, and adds a `reset`/`clear` command. **Key gotcha:**
  the existing "use ONLY the records" grounding rule FOUGHT history — the model distrusted earlier turns
  and refused ("whom am I?" retrieves random cards). Reconciled the SYSTEM_PROMPT to "records AND facts
  established earlier in THIS conversation", with "not found" only when NEITHER has the answer. After the
  fix: "whom am I?" recalls Vigneshwar B; "who is his manager?" (after Pari) resolves the pronoun →
  Aditya Sharma; `reset` → forgets again. **Boundary documented:** history steers GENERATION, not
  RETRIEVAL (router still routes on the raw question) — follow-up-aware retrieval = query rewriting, the
  next step. Verified: memory + follow-up + reset work; metadata (22)/count (97) no regression; app.py
  stateless-unchanged. Docs: learnings §15. **Lesson: memory has two jobs — the LLM seeing history
  (done) vs. retrieval following references (query rewriting, next); and a grounding prompt must be told
  history is a trusted source.** **Next:** Phase 8 — experiments & polish (incl. query rewriting).
- **2026-07-29 — Session 11 (Phase 9 — Schema-agnostic RAG ✅):**
  Learner's ask: "let the user give ANY Excel (dob, salary, grade, notes, mentor…) and the RAG still
  works if I change the data." Problem: the domain was hardcoded in 3 places — `data.py` (4 named cols),
  `retrieval.py` (`FILTERABLE_FIELDS` + hand-written `FIELD_CUES` + fixed known-values), and the implicit
  "identity == name". Fix: infer the schema at load time and drive everything from it.
  **New module `src/schema.py`:** classifies each column into one of five ROLES — identity / categorical /
  numeric / date / freetext — using tunable thresholds (`CATEGORICAL_MAX_DISTINCT=40`, ratio 0.5,
  `PARSE_SUCCESS_RATIO=0.8`). Order matters: numeric before date (so 20000 isn't a timestamp), regex
  pre-filter so names aren't parsed as dates. `SchemaProfile` persists to `chroma_db/schema.json` (query
  process is separate from indexing). **`data.py`:** generic `row_to_card()` (prose from whatever columns
  exist, freetext appended) + generic scalar metadata (numeric→float, date→**epoch float** so Chroma
  `$gt`/`$lt` works, categorical→string); `load_employee_documents()` now returns `(docs, schema)`.
  **`index.py`:** persists + reloads the schema (`get_schema()`). **`retrieval.py` (the big one):**
  known-values + router PROMPT generated from the schema; **NEW `range` route**
  (`{route,field,op,value[,value2]}` via Chroma `$gt`/`$lt`); aggregate generalized count → avg/min/max/sum
  (Python owns the math); the `resolve_value` fuzzy-snap guardrail kept, plus numeric/date thresholds must
  parse or fall back to semantic; a generic (cue-free) rule-based fallback for when the gate is down.
  **`generate.py`:** `_answer_count` → `_answer_aggregate` (phrases avg/min/max/sum). **CLI + Streamlit:**
  show the DETECTED schema on startup; Streamlit gets a **file-uploader → reindex → chat** (the headline
  "it's dynamic" demo). Verified end-to-end: original sheet unchanged (reports-to → 22, total → 97,
  semantic + memory intact); a synthetic {salary, joining date, notes} sheet correctly routes "earns more
  than 2M" (range→3), "avg salary of M2" (aggregate→2,000,000), "joined after 2022" (date range→2),
  "highest salary" (max→3,000,000). Docs: learnings §16. **Lesson: to make a hybrid RAG schema-agnostic,
  the SEMANTIC half is already generic — the work is the metadata half: infer column roles from the data,
  persist the profile, and GENERATE the router prompt/known-values/routes from it so they can't drift.
  Chroma's scalar-only metadata is why dates become epoch floats.** **Next:** query rewriting
  (follow-up-aware retrieval) / Phase 8c BM25 (planned).
- **2026-08-21 — Session 12 (root entrypoint fix):** User tried `python3 main.py` from the repository
  root and hit `[Errno 2] No such file or directory` because the real CLI lived at `src/main.py` only.
  Added a tiny root-level launcher `main.py` that inserts `src/` on `sys.path` and delegates to the
  existing CLI, so `python main.py` and `python src/main.py` both work. Verified with the project
  virtualenv: `./.venv/bin/python main.py --help` and a full startup/quit cycle both succeed.

---

## ✅ Resolved Decisions
1. **LLM:** local Claude @ `http://localhost:8080/v1` (OpenAI-compatible). No key/cost.
2. **Embeddings:** local `all-MiniLM-L6-v2`.
3. **Size:** 97 employees — small, so we add metadata filtering (pure vector search alone is weak here).
4. **Excel:** provided; will copy into `data/employees.xlsx`.

## 🔌 Local Claude Gate @ localhost:8080 (VERIFIED 2026-07-24)
- **Re-check anytime:** `bash scripts/check_gate.sh` · full method in `docs/verify-gate.md`
- **Health:** `{"status":"ok","version":"0.7.0","backend":"copilot"}`
- **API:** OpenAI-compatible → `GET /v1/models`, `POST /v1/chat/completions`. No API key needed.
- **Chat test:** passed (round-trip confirmed).
- **Generation model (pick one):** `claude-sonnet-5` (balanced, default) or `claude-haiku-4.5` (fast).
  Opus not needed for grounded lookups.
- **Embeddings via gate:** currently not working in this environment.
  → **Current decision:** use local `sentence-transformers` with `all-MiniLM-L6-v2` as primary.
- **LangChain wiring:**
  - LLM: `ChatOpenAI(base_url="http://localhost:8080/v1", api_key="unused", model="claude-sonnet-5")`
  - Emb: `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")`

## 🌐 Environment Quirks — first model download only (corporate SSL proxy)
The network does **TLS inspection**, so `certifi`'s CA bundle fails to verify `huggingface.co`.
These matter **only** the first time `all-MiniLM-L6-v2` is downloaded on a machine. Once cached
(`~/.cache/huggingface/hub`), the model loads **fully offline** and none of this applies.

1. **CA bundle:** use the Mac's Keychain roots (includes the corporate root), saved to
   `certs/keychain-roots.pem`. Regenerate with:
   ```bash
   security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain > certs/keychain-roots.pem
   security find-certificate -a -p /Library/Keychains/System.keychain >> certs/keychain-roots.pem
   ```
   Then set `SSL_CERT_FILE=$PWD/certs/keychain-roots.pem` for the download.
2. **Disable Xet:** HF's `xet` chunked downloader corrupts through the proxy
   (`Byte range not sequential`). Set `HF_HUB_DISABLE_XET=1` to force the classic HTTP downloader.

One-time download command (already run in Session 1):
```bash
SSL_CERT_FILE=$PWD/certs/keychain-roots.pem HF_HUB_DISABLE_XET=1 \
  python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
  HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2').embed_query('warmup')"
```

**Streamlit file-watcher traceback (harmless):** `streamlit run src/app.py` logs a traceback about
`transformers.models.tvp … torchvision`. It's Streamlit's auto-reload watcher scanning imported modules and
tripping a lazy `transformers` vision import that needs `torchvision` (not installed). The app works fine — it
happens at watcher scan time, not on any query. Silence with `--server.fileWatcherType none` if desired.

---

## ▶️ How to Resume a Session
Tell Claude:
> "Read PROJECT.md in employee-rag and let's continue from the next pending phase."
