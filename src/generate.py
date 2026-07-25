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

from langchain_openai import ChatOpenAI

from retrieval import retrieve

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
# anti-hallucination guardrail: answer ONLY from the given records. Without it,
# the model will confidently invent a plausible-but-wrong manager/project.
SYSTEM_PROMPT = """You answer questions about a team of employees.

Rules:
- Use ONLY the employee records provided below. Do not use outside knowledge.
- If the records do not contain the answer, say: "I don't know based on the data."
- Be concise. When listing people, list every matching person.
- Do not invent names, managers, projects, or technologies."""


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


def build_messages(question: str, docs) -> list:
    """Assemble the chat messages: system rules + context cards + the question.

    This 'stuffing' of retrieved context into the prompt is the AUGMENTATION in
    Retrieval-Augmented Generation — we ground the model in *our* data.
    """
    context = format_cards(docs)
    user_prompt = (
        f"Employee records:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the records above."
    )
    return [
        ("system", SYSTEM_PROMPT),
        ("human", user_prompt),
    ]


def answer(question: str, store=None, k: int = 4, debug: bool = True) -> dict:
    """Full RAG: retrieve -> augment -> generate. Returns {answer, sources}."""
    # 1. RETRIEVE (Phase 3 decides metadata-filter vs semantic).
    docs = retrieve(question, store=store, k=k, debug=debug)

    # 2. AUGMENT + 3. GENERATE.
    messages = build_messages(question, docs)
    llm = get_llm()
    response = llm.invoke(messages)

    # sources = the exact records the answer is allowed to be based on.
    sources = [d.metadata["name"] for d in docs]
    return {"answer": response.content, "sources": sources}


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
