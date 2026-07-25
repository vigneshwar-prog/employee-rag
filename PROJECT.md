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

### Phase 5 — Interface ✅
- [x] CLI Q&A loop (`main.py`) — single entry point, opens store once,
      `--debug` flag reveals the routing decision, clean quit + gate-down error
- [ ] (Optional) Streamlit UI with a text box + sources panel

### Phase 6 — Improvements (learning extensions) ⏳
- [ ] **Aggregation route (count / how-many)** — learner-found bug: "how many employees in
      total?" fell to semantic → counted the 4 retrieved cards → answered "4" (real = 97).
      Semantic/metadata can't count. Fix: add a third route `"aggregate"` to `llm_route`
      (e.g. `{route:"aggregate", operation:"count", field:"manager"|null}`) + a Python branch
      that does `store.get(where=...)` and returns the real count. This IS lightweight
      tool-calling — same "LLM decides, Python executes" pattern already in `llm_route()`.
      Covers "how many employees", "how many under X", "how many on MPLS".
- [ ] (Later) Full framework-based tool/function calling for richer ops beyond count.
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
- Phase 5 — Interface: ✅ Done (CLI `main.py`; Streamlit still optional)
- Phase 6 — Improvements: ⏳ Pending

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

---

## ▶️ How to Resume a Session
Tell Claude:
> "Read PROJECT.md in employee-rag and let's continue from the next pending phase."
