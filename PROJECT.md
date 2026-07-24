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

### Phase 1 — Data ⏳  (source: 97 rows, cols = Employee Name, Manager, Project, Technology)
- [ ] Load with pandas via LangChain loader; inspect
- [ ] Handle 6 rows with missing Project/Technology → fill with "Unassigned"
- [ ] Build one **Document per employee**:
      - `page_content` = readable card, e.g.
        *"Ishan Sharma is managed by Dhruv Sharma, works on the Morgan Stanley project,
        specializing in R&S (Routing & Switching)."*
      - `metadata` = {name, manager, project, technology}  ← powers filtering

### Phase 2 — Embeddings & Indexing ⏳
- [ ] Embed cards with `all-MiniLM-L6-v2` (HuggingFaceEmbeddings)
- [ ] Persist to Chroma on disk (`./chroma_db`); store metadata alongside
- [ ] `index.py` re-run script (rebuild when Excel changes)

### Phase 3 — Hybrid Retrieval ⏳  ← core learning
- [ ] Semantic retriever: embed question, top-k similarity search
- [ ] Metadata filter: detect known managers/projects/techs in the question →
      Chroma `where={"manager": "Dhruv Sharma"}` for exact "list all" answers
- [ ] Debug mode: print retrieved employees before calling the LLM

### Phase 4 — Generation (Claude @ localhost:8080) ⏳
- [ ] Wire `ChatOpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")`
- [ ] Prompt template: system rules ("answer ONLY from context, say if unknown")
      + retrieved employee cards + question
- [ ] Return grounded answer **+ list of source employees** (anti-hallucination)
- [ ] Build the LangChain RAG chain (retriever → prompt → LLM → parser)

### Phase 5 — Interface ⏳
- [ ] CLI Q&A loop (`main.py`)
- [ ] (Optional) Streamlit UI with a text box + sources panel

### Phase 6 — Improvements (learning extensions) ⏳
- [ ] Compare pure-semantic vs hybrid on "list everyone on MPLS"
- [ ] Try different k; try a bigger embedding model
- [ ] Aggregations ("how many people per manager?") via metadata
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
- Phase 1 — Data: ⏳ Pending
- Phase 2 — Embeddings & Indexing: ⏳ Pending
- Phase 3 — Retrieval: ⏳ Pending
- Phase 4 — Generation: ⏳ Pending
- Phase 5 — Interface: ⏳ Pending
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
