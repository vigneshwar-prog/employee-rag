"""
Phase 5 — The interface (the front door).

Everything under this — data, index, retrieval, generation — is now wired
together behind a single command:

    python src/main.py            # clean Q&A loop
    python src/main.py --debug    # also print the routing decision per question

Why a separate file when generate.py already has a loop? Separation of concerns:
generate.py is the ENGINE (retrieve -> augment -> generate). main.py is the UX
around it — open the store once, hide the debug noise by default, and fail
gracefully (empty input, Ctrl-C/Ctrl-D, a clear message if the gate is down)
instead of dumping a stack trace at the learner.
"""

import argparse
import sys

from langchain_core.chat_history import InMemoryChatMessageHistory

from index import get_store, get_schema
from generate import answer


BANNER = """\
Employee RAG — ask natural-language questions about the team.
Examples:
    who reports to Vignesh?
    list everyone on MPLS
    who keeps the network reliable?
Type 'q' (or Ctrl-C / Ctrl-D) to quit.  Type 'reset' to clear the conversation memory.
"""

# Keep the last N turns of history (N=6 turns = 12 messages). Enough for natural
# back-references ("whom am I?", "his manager?") without bloating the prompt.
HISTORY_TURNS = 6


def ask_loop(store, debug: bool) -> None:
    """Read a question, answer it, print answer + sources. Repeat until quit.

    Phase 8b — CONVERSATIONAL MEMORY: we keep a running `history` of prior
    (question, answer) turns and pass it to answer() each time, so the model can
    resolve back-references. Memory lives only for this session (lost on restart).
    """
    history = InMemoryChatMessageHistory()
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D / Ctrl-C — exit cleanly, no traceback.
            print("\nBye.")
            return

        if question.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            return
        if question.lower() in {"reset", "clear"}:
            history.clear()
            print("  [memory cleared]\n")
            continue
        if not question:
            continue  # empty line -> just re-prompt

        try:
            result = answer(question, store=store, debug=debug, history=history.messages)
        except Exception as exc:
            # The most likely cause is the local gate being down/unreachable.
            print(f"\n  [error] Could not answer: {exc}")
            print("  Is the local Claude gate up?  bash scripts/check_gate.sh\n")
            continue

        print(f"\n  Answer: {result['answer']}")
        srcs = result["sources"]
        # Only cite sources when the answer was actually built from records.
        # (Chitchat / "I don't know" answers cite nothing — no misleading names.)
        if srcs:
            print(f"  Sources ({len(srcs)}): {', '.join(srcs)}\n")
        else:
            print()

        # Record this turn AFTER a successful answer (so a gate error doesn't
        # poison history), then cap to the last HISTORY_TURNS turns.
        history.add_user_message(question)
        history.add_ai_message(result["answer"])
        del history.messages[: -2 * HISTORY_TURNS]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask natural-language questions about the team (hybrid RAG)."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="print the routing decision (METADATA vs SEMANTIC) for each question",
    )
    args = parser.parse_args()

    # Open the vector store ONCE and reuse it for every question (embedding the
    # model + connecting to Chroma is the slow part; do it a single time).
    try:
        store = get_store()
    except Exception as exc:
        print(f"[fatal] Could not open the vector store: {exc}")
        print("Did you build it first?  python src/index.py")
        sys.exit(1)

    print(BANNER)
    # Phase 9: show what the system INFERRED about the current sheet, so the user
    # SEES the detected columns/roles (great demo + debugging aid).
    schema = get_schema()
    if schema:
        print(schema.describe() + "\n")
    ask_loop(store, debug=args.debug)


if __name__ == "__main__":
    main()
