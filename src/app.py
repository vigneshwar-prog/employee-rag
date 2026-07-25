"""
Phase 6 — Streamlit UI (the browser front-end).

Same engine as the CLI (`generate.answer()`), now behind a web page: a text box,
the answer, and a sources panel. Every fix we built in Phases 3.5/4.5/4.6 carries
over for free, because we call the exact same `answer()` — the UI is only a new
*face* on the unchanged engine.

Run it:
    streamlit run src/app.py

KEY STREAMLIT CONCEPT — reruns:
Streamlit re-executes this whole script top-to-bottom on EVERY interaction
(typing, clicking Ask, ticking a box). Opening the vector store loads the MiniLM
embedding model, which is slow — so we must NOT do it on every rerun. The
@st.cache_resource decorator runs get_store() ONCE and hands back the same object
on later reruns. It's the web equivalent of main.py's "open the store once".
"""

import sys
from pathlib import Path

# generate.py uses bare imports (from index import ...), which resolve when `src`
# is on the path. `streamlit run src/app.py` doesn't add src automatically, so we do.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from index import get_store
from generate import answer


@st.cache_resource(show_spinner="Loading the employee index (one-time)…")
def load_store():
    """Open the Chroma store ONCE and reuse it across reruns.

    @st.cache_resource caches the returned object for the whole app session, so the
    slow embedding-model load happens a single time, not on every keystroke/click.
    """
    return get_store()


def main() -> None:
    st.set_page_config(page_title="Employee RAG", page_icon="🧑‍💼")

    st.title("🧑‍💼 Employee RAG")
    st.caption(
        "Ask natural-language questions about the team. "
        "The engine routes each question to an exact metadata filter or a semantic search."
    )

    store = load_store()

    # A checkbox mirrors the CLI's --debug flag: show the routing decision.
    debug = st.checkbox("Show routing decision (debug)", value=False)

    # st.form batches the input + button so the script only reruns the query on
    # submit (Enter or the button), not on every character typed.
    with st.form("ask"):
        question = st.text_input(
            "Your question",
            placeholder="e.g. who reports to Dhruv Sharma?   |   list everyone on MPLS",
        )
        submitted = st.form_submit_button("Ask")

    if not submitted:
        return
    if not question.strip():
        st.warning("Please type a question first.")
        return

    # Call the SAME engine the CLI uses. debug=True makes retrieve() print the route
    # to the server console; we also surface it in the UI below when the box is ticked.
    try:
        with st.spinner("Thinking…"):
            result = answer(question, store=store, debug=debug)
    except Exception as exc:
        st.error(
            f"Could not answer: {exc}\n\n"
            "Is the local Claude gate up?  `bash scripts/check_gate.sh`"
        )
        return

    st.markdown("### Answer")
    st.markdown(result["answer"])

    # Honor the citation fix: only show sources when the answer actually used some.
    sources = result["sources"]
    if sources:
        with st.expander(f"📇 Sources ({len(sources)})", expanded=True):
            for name in sources:
                st.markdown(f"- {name}")
    else:
        st.caption("No specific employee records were cited for this answer.")


if __name__ == "__main__":
    main()
