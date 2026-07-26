# Phase 8c — BM25 hybrid experiment (PLANNED, not yet implemented)

**Status:** planned & parked. Implement later, then move on.
**Decision recorded:** "Learning experiment + docs" scope (option A) — BM25 gated on the SEMANTIC
branch only, off by default; metadata/aggregate paths untouched.

---

## Is it even required? (the honest answer — keep this framing for interviews)

**Not required for this dataset — and that judgment IS the lesson.** BM25 is lexical (keyword) ranking;
it shines where embeddings miss the literal token (error codes, IDs, acronyms, rare names). But in
employee-rag the "keyword" queries ("who reports to Dhruv", "list everyone on MPLS", counts) are already
handled by the **metadata `where` filter** via the LLM router + `resolve_value()` guardrail — which is
*exact and complete* (all 12 MPLS people, not a top-k), i.e. **stronger than BM25** for categorical data.
Semantic search only runs as the fuzzy fallback, exactly where BM25 is weak and embeddings are right.
So the "keyword gap" BM25 usually bridges is, in this architecture, the metadata branch — already solved,
better.

**Why still build it:** it's a top-tier interview topic (hybrid search, RRF), and knowing *when NOT to
add it* is senior signal. Frame it as: *"categorical data made metadata filtering the better keyword path,
so BM25 wasn't needed in the flagship; I implemented it as a documented experiment and wrote up when
BM25+vector hybrid (RRF) IS the right call — a large free-text corpus like the RFC scenario."*

---

## The idea in one line
On the **semantic fallback only**, retrieve vector top-k *and* BM25 top-k, fuse with **Reciprocal Rank
Fusion (RRF)**. Off by default (`use_bm25=False`); metadata/aggregate paths untouched.

## Why RRF (interview-ready)
Vector scores (cosine distance) and BM25 scores (term-frequency sums) are on **incompatible scales** — you
can't add them. RRF fuses by **rank**, not score: `score(d) = Σ 1/(rrf_k + rank_i(d))` (rrf_k≈60). Robust,
tuning-free, the standard hybrid default.

---

## Files & changes (all additive, backwards-compatible)

**1. `requirements.txt`** — add `rank_bm25` (confirmed NOT currently installed).

**2. `src/index.py`** — `get_bm25()` helper: `store.get()` all docs, tokenize `page_content`, build a
`BM25Okapi` index once. 97 rows → in-memory per process is fine (note the cost in a comment).

**3. `src/retrieval.py`** — the core:
- `_tokenize(text)` — lowercase/split; comment that it's deliberately naive (real systems stem + drop
  stopwords).
- `_bm25_search(question, store, k)` — top-k docs by BM25.
- `_rrf_fuse(vector_docs, bm25_docs, k=4, rrf_k=60)` — fuse two ranked lists → one ranked list, deduped by
  employee name.
- Thread `use_bm25=False` through `retrieve_multi → retrieve_with_plan → _execute_plan`. **Only the
  SEMANTIC branch** in `_execute_plan` consults it: if `use_bm25`, fuse vector+BM25; else current behavior
  verbatim.
- `--debug` prints both candidate lists + the fused order (so you can SEE what BM25 pulled in that vectors
  missed — the demo moment).

**4. `src/main.py`** — a `bm25` toggle command (like `reset`) to flip it mid-session and compare answers
live. Default off.

**5. Docs (the real payoff):**
- `docs/learning-qa.html` — new **Lesson 007: "BM25, hybrid search & RRF — what, when, and why we didn't
  need it here."** What BM25 is, why raw-score fusion fails, how RRF works, and the honest judgment
  (categorical → metadata filter beats BM25; BM25+vector hybrid is for large free-text / the RFC scenario).
  Match existing card format + add to sidebar nav.
- `docs/learnings.html` — short section tagged as an **experiment** (badge), linking to the QA lesson, with
  the `bm25` toggle demo. Update nav + section balance.
- `PROJECT.md` — Progress Tracker (Phase 8c) + Session log entry.

## Backwards compatibility & safety
- Default `use_bm25=False` everywhere → **zero behavior change** unless opted in. `app.py` and existing
  callers untouched.
- BM25 never touches metadata/aggregate → exact-set answers (MPLS list, counts) stay exact.

## Verification
- `bm25` OFF → identical answers to today (no regression).
- `bm25` ON, "who keeps the network reliable?" → `--debug` shows vector list, BM25 list, fused order;
  answer still sensible.
- A keyword-y semantic query → BM25 surfaces an exact-token card vectors ranked lower — the "why hybrid"
  moment.
- Metadata/count queries → unchanged whether toggle is on or off.

## Scope
~1 new dep, ~60 lines in `retrieval.py`, one `index.py` helper, one CLI toggle, and the docs. **No changes
to `generate.py` or `app.py`** (they consume `retrieve_multi` → `{plan, docs}` only).
