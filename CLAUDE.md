# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read first
**`PROJECT.md` is the source of truth** for the plan, architecture, confirmed tech-stack decisions,
the Excel schema, the phase-by-phase progress tracker, and the session log. Read it at the start of
every session; at the end, update its **Progress Tracker** and **Session Log**. Do not duplicate that
content here — this file only covers what isn't obvious from PROJECT.md.

## What this is
A learning project: a hybrid RAG system that answers natural-language questions about ~97 team
members loaded from an Excel sheet (`data/employees.xlsx`, sheet "Employee Mapping", columns:
Employee Name, Manager, Project, Technology). As of the latest session the repo is **scaffolded but
has no code yet** — `src/` and `data/` are empty. Code goes in `src/` (planned: `index.py` to build
the vector store, `main.py` for the CLI Q&A loop).

## Core architectural decision (don't undo without reason)
The retriever is **hybrid**, not pure vector search: semantic similarity **plus** metadata filtering
on `manager` / `project` / `technology`. The data is small and categorical, so "list everyone on
MPLS" / "who reports to X" are exact filters (Chroma `where` clause), which pure embedding search
gets wrong. Each employee = one Chroma Document: `page_content` is a readable card, `metadata` holds
`{name, manager, project, technology}`. Fill missing Project/Technology with `"Unassigned"`.

## The local LLM gate (critical dependency)
Both generation and embeddings run against a local, OpenAI-compatible Claude gate at
`http://localhost:8080` — **no API key, no cost**. LangChain wiring:
- LLM: `ChatOpenAI(base_url="http://localhost:8080/v1", api_key="unused", model="claude-sonnet-5")`
- Embeddings: `OpenAIEmbeddings(base_url="http://localhost:8080/v1", api_key="unused", model="text-embedding-3-small")`

**Verify the gate before debugging any LLM/embedding failure** — a port can be open but not serving:
```bash
bash scripts/check_gate.sh    # 4 checks, ends with real inference using a unique sentinel
```
Full rationale and manual curl steps: `docs/verify-gate.md`. Override with env vars `GATE_BASE` /
`GATE_MODEL`. If offline, the documented fallback is local `sentence-transformers` (`all-MiniLM-L6-v2`).

## Run (once code exists)
```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```
