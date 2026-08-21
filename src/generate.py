"""
Phase 4 — Generation (the "reader").

Retrieval (Phase 3) finds the right employee cards. This phase adds the LLM that
READS those cards and writes a natural-language answer — grounded strictly in the
retrieved data, never invented.

Pipeline:
    question
      -> retrieve()            [Phase 3: hybrid metadata/semantic]
      -> build a prompt        (system rules + the cards + the question)
      -> Claude @ localhost:8080 generates the answer
      -> return {answer, sources}

The LLM runs on the local Claude gate (OpenAI-compatible, no key, no cost).

Run it to ask questions and get real answers:
    python src/generate.py
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from retrieval import retrieve_multi

# --- LLM config: the local Claude gate --------------------------------------
GATE_BASE = "http://localhost:8080/v1"
GATE_MODEL = "claude-sonnet-5"


def get_llm() -> ChatOpenAI:
    """Connect to the local Claude gate via LangChain's OpenAI-compatible client.

    temperature=0 -> deterministic, factual answers (we want lookups, not creativity).
    api_key is required by the client but the gate ignores it.
    """
    return ChatOpenAI(
        base_url=GATE_BASE,
        api_key="unused",
        model=GATE_MODEL,
        temperature=0,
    )


# --- The prompt: the heart of RAG -------------------------------------------
# The SYSTEM message sets the rules. The single most important rule is the
# anti-hallucination guardrail: state facts ONLY from the given records. Without
# it, the model will confidently invent a plausible-but-wrong manager/project.
#
# Phase 4.5 (learner's productization idea): make the tone warmer and phrase the
# "no" HELPFULLY — WITHOUT loosening the grounding. We deliberately do NOT let the
# model fall back to its own general knowledge when the data lacks the answer:
# for an employee-lookup tool a confident wrong answer ("Yes, Balamurugan reports
# to X") is far worse than an honest, guiding "I don't have that in the records."
#
# Phase 4.6 (learner's finding): the retriever ALWAYS hands over k cards, even for
# a meaningless query ("vicky") where they're random/irrelevant. The old prompt let
# the model recite those names ("the employees I know about are ...") — misleading,
# and it wrongly implied those were the only records. New rule: NEVER list the
# provided records unless they actually match what the user asked for.
SYSTEM_PROMPT = """You are a friendly assistant that answers questions about a team of employees.

Rules:
- Use ONLY the employee records provided below AND facts already established
  earlier in THIS conversation to state facts. Never use outside knowledge, and
  never invent names, managers, projects, or technologies. Treat something the
  conversation already established (e.g. the user said "I am X" and you confirmed
  a record) as a known fact you may rely on in later turns — a follow-up like
  "whom am I?" or "who is his manager?" should be answered from that context even
  if this turn's auto-retrieved records don't repeat it.
- The records below were auto-retrieved and MAY be irrelevant to the question.
  Only use the ones that genuinely match what the user asked. If none match, treat
  it as "not found" — do NOT list or recite the other records, and never imply they
  are the only employees that exist.
- If NEITHER the records NOR the earlier conversation contains the answer, do not
  guess. Instead, respond in a natural, friendly way such as: "I can’t answer
  that from this employee dataset" or "This dataset only includes employee
  information like manager, project, and technology." Then offer a helpful next
  step (e.g. suggest searching by manager / project / technology). Do NOT make
  up an answer from general knowledge.
- Only suggest re-checking the spelling when you found NO matching record. If a
  record DOES match (even via a close/fuzzy name), just answer confidently — do
  NOT tack on a "if you meant someone else, check the spelling" caveat, as it
  annoys the user after a correct answer.
- Answer ONLY what was asked — match the granularity of the question:
    • If the user asks for ONE specific field ("who is the manager of X", "which
      technology does X know", "what project is X on"), give ONLY that field —
      do NOT recite the person's other fields.
    • Give the FULL card (manager + project + technology) only for open questions
      like "who is X" / "tell me about X" / "give me details on X".
- Be warm and concise, but keep it TIGHT: at most ONE short closing offer for the
  WHOLE reply. Do NOT append a "let me know!" / "want more?" line after every
  person or bullet — it reads as noise when several items are listed.
- When listing people who match the question, list every matching person."""


FOLLOWUP_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You rewrite a follow-up question into a SELF-CONTAINED question for an "
        "employee lookup tool.\n"
        "Rules:\n"
        "- Use the conversation history to resolve pronouns and missing context.\n"
        "- Preserve the user's intent exactly; do not add new facts.\n"
        "- If the latest question is already standalone, return it unchanged.\n"
        "- If the latest question is vague but refers to the previous result, make it "
        "concrete in the most useful records-based way. For example, after a count "
        "answer, 'give me the break up' should usually become a list/details question "
        "about the same matching employees rather than inventing a new grouping.\n"
        "- Output ONLY the rewritten question, no explanation."
    ),
    MessagesPlaceholder("history"),
    ("human", "Latest user question: {question}"),
])


def format_cards(docs) -> str:
    """Turn retrieved Documents into a numbered block of text for the prompt.

    We feed the readable card sentences — that's exactly what the model reads.
    Numbering makes it easy for the model (and us) to refer to specific records.
    """
    if not docs:
        return "(no matching employee records found)"
    lines = []
    for i, d in enumerate(docs, start=1):
        lines.append(f"{i}. {d.page_content}")
    return "\n".join(lines)


def rewrite_followup_question(question: str, history=None) -> str:
    """Rewrite an ambiguous follow-up into a standalone query using LangChain
    chat history. If rewriting fails, fall back to the original question.
    """
    if not history:
        return question
    try:
        llm = get_llm()
        messages = FOLLOWUP_REWRITE_PROMPT.format_messages(
            history=history,
            question=question,
        )
        rewritten = llm.invoke(messages).content.strip()
    except Exception:
        return question
    return rewritten or question


def build_messages(question: str, docs, history=None) -> list:
    """Assemble the chat messages: system rules + prior turns + context cards + question.

    This 'stuffing' of retrieved context into the prompt is the AUGMENTATION in
    Retrieval-Augmented Generation — we ground the model in *our* data.

    `history` (Phase 8b) is a list of prior turns as ("human"/"ai", text) tuples.
    We slot it BETWEEN the system rules and the current turn, so the model sees the
    running conversation ("I am vignesh" -> ...) and can answer back-references
    ("whom am I?"). Default None -> stateless, exactly as before.
    """
    context = format_cards(docs)
    user_prompt = (
        f"Employee records:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the records above."
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        *(history or []),
        HumanMessage(content=user_prompt),
    ]


def _cited_sources(answer_text: str, docs) -> list[str]:
    """Return only the retrieved people the answer ACTUALLY used — not everything
    retrieval happened to fetch.

    Two filters (learner's finding: semantic always fetches k cards, so chitchat /
    "I don't know" answers were citing 4 unrelated names — misleading):
      1. Refusal guard — if the answer didn't use the records (a grounded "I don't
         have that", or nothing was retrieved), cite nothing.
      2. Mention filter — keep a person only if their name appears in the answer.
         We match the full name OR the first token, so "Pari Sharma" is still
         credited when the answer says just "Pari".
    """
    if not docs:
        return []
    low = answer_text.lower()
    # 1. Grounded-refusal / no-data phrasing -> the answer isn't based on any record.
    refusal_cues = ("i don't have", "i do not have", "don't know", "do not know",
                    "not something i", "can only help", "double-check the spelling",
                    "no one named", "don't see anyone", "do not see anyone")
    if any(cue in low for cue in refusal_cues):
        return []
    # 2. Keep only people the answer names (full name or first name).
    cited = []
    for d in docs:
        name = d.metadata["name"]
        first = name.split()[0].lower()
        if name.lower() in low or first in low:
            cited.append(name)
    return cited


def _answer_aggregate(question: str, plan: dict) -> str:
    """Phrase an AGGREGATE answer (count / avg / min / max / sum). Python already
    computed the number; the LLM only wraps it in a friendly sentence — it must NOT
    recompute it.

    This is the heart of Phase 7 (generalized in Phase 9): the model is bad at math,
    so Python owns the number and we hand it over as a fact to be restated verbatim.
    """
    agg = plan.get("agg", "count")
    if agg == "count":
        number = plan["count"]
        field, value = plan.get("field"), plan.get("value")
        scope = f"{field} = {value}" if field else "the whole team"
        what = "count"
    else:
        number = plan.get("agg_value")
        field = plan.get("field")
        sf, sv = plan.get("scope_field"), plan.get("scope_value")
        scope = f"{sf} = {sv}" if sf else "the whole team"
        what = f"{agg} of {field}"
    system = (
        "You state a precomputed statistic in one warm, concise sentence.\n"
        f"The exact {what} is {number}. Use this number EXACTLY — do not change or "
        "recompute it.\n"
        "Do not list names because you were not given any.\n"
        "Do not invent unsupported dimensions such as level, location, department, "
        "or seniority. If you offer a follow-up, keep it generic and safe, such as "
        "offering to list the matching employees or show the matching records."
    )
    human = f"Question: {question}\nScope: {scope}\nExact {what}: {number}"
    llm = get_llm()
    return llm.invoke([("system", system), ("human", human)]).content


# Backwards-compatible alias (older name).
_answer_count = _answer_aggregate


def _answer_one(sub: dict, store, history=None) -> tuple[str, list[str]]:
    """Produce (answer_text, sources) for ONE sub-result, reusing the exact
    single-question logic: aggregate -> _answer_count; metadata/semantic ->
    build_messages + llm.invoke + _cited_sources. `history` is forwarded so
    sub-answers are conversation-aware too.
    """
    plan, docs = sub["plan"], sub["docs"]
    question = plan.get("question", "")
    if plan.get("route") == "aggregate":
        return _answer_aggregate(question, plan), []
    messages = build_messages(question, docs, history=history)
    response = get_llm().invoke(messages)
    return response.content, _cited_sources(response.content, docs)


def _stitch(question: str, parts: list[tuple[str, str]]) -> str:
    """Combine already-correct sub-answers into one warm, cohesive reply.

    The sub-answers are the ONLY source material — the model must not add facts,
    recompute any count, or drop a part. (Cheaper alternative: "\\n\\n".join the
    sub-answers; the LLM stitch just reads warmer for a compound question.)
    """
    system = (
        "You combine several already-correct answers into ONE cohesive, friendly "
        "reply to the user's original question.\n"
        "Use ONLY the provided answers. Do NOT add new facts, do NOT recompute or "
        "change any number, and do NOT drop any part. Keep it warm and concise."
    )
    joined = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in parts)
    human = f"Original question: {question}\n\nAnswers to combine:\n{joined}"
    return get_llm().invoke([("system", system), ("human", human)]).content


def answer(question: str, store=None, k: int = 4, debug: bool = True, history=None) -> dict:
    """Full RAG: retrieve -> augment -> generate. Returns {answer, sources}.

    Handles COMPOUND questions (Phase 8a): retrieve_multi returns one sub-result
    per intent. A simple question yields exactly one -> behavior unchanged.

    `history` (Phase 8b) is an optional list of prior ("human"/"ai", text) turns.
    When present, the LLM sees the running conversation and can answer
    back-references ("whom am I?"). Default None -> stateless, as before. (Note:
    history steers GENERATION, not RETRIEVAL — the router still routes on the raw
    question; follow-up-aware retrieval is a later 'query rewriting' step.)
    """
    standalone_question = rewrite_followup_question(question, history=history)
    if debug and standalone_question != question:
        print(f"  [memory] Rewrote follow-up -> {standalone_question}")

    subs = retrieve_multi(standalone_question, store=store, k=k, debug=debug)

    # --- Single-question path: identical behavior to before Phase 8a. ---
    if len(subs) == 1:
        plan, docs = subs[0]["plan"], subs[0]["docs"]
        # AGGREGATE route: Python counted; LLM just phrases the number (no history needed).
        if plan.get("route") == "aggregate":
            return {"answer": _answer_aggregate(question, plan), "sources": []}
        # AUGMENT + GENERATE (metadata / semantic routes).
        messages = build_messages(question, docs, history=history)
        response = get_llm().invoke(messages)
        sources = _cited_sources(response.content, docs)
        return {"answer": response.content, "sources": sources}

    # --- Compound path: answer each sub independently, then stitch. ---
    parts, sources = [], []
    for sub in subs:
        text, src = _answer_one(sub, store, history=history)
        parts.append((sub["plan"].get("question", question), text))
        sources += src
    final = _stitch(question, parts)
    return {"answer": final, "sources": sorted(set(sources))}


if __name__ == "__main__":
    from index import get_store

    store = get_store()
    print("Employee RAG — ask a question ('q' to quit).\n")
    while True:
        question = input("Question: ")
        if question.lower() == "q":
            break
        result = answer(question, store=store)
        print(f"\n  Answer: {result['answer']}")
        print(f"  Sources ({len(result['sources'])}): {', '.join(result['sources'])}\n")
