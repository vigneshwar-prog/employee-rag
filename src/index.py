"""
Phase 2 — Embeddings & indexing.

Takes the 97 employee Documents from Phase 1, turns each card into a vector
(embedding), and stores everything in a persistent Chroma vector database on disk.

Run it whenever the Excel changes, to (re)build the index:
    python src/index.py

Why a persistent store? Embedding 97 cards is cheap, but we don't want to redo it
on every question. Chroma writes the vectors + metadata to ./chroma_db once; later
phases just OPEN that directory and query it instantly.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from data import load_employee_documents

# --- Config -----------------------------------------------------------------
# Anchor the store to the PROJECT ROOT so the DB is always ./chroma_db there,
# regardless of the current working directory when you run the script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")
# One logical "table" of vectors inside that store.
COLLECTION_NAME = "employees"
# Local embedding model (Phase 0 decision: the gate doesn't serve embeddings).
# 384-dimensional vectors, runs on CPU, loads offline once cached.
EMBED_MODEL = "all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """The ONE place the embedding model is configured.

    Phase 3 (retrieval) must embed the *question* with the exact same model that
    embedded the *cards* — otherwise the vectors live in different spaces and
    similarity is meaningless. Importing this function keeps them in lockstep.
    """
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_index() -> Chroma:
    """Excel -> Documents -> embeddings -> persistent Chroma store.

    Uses each employee's name as a STABLE id, so re-running updates the same
    records instead of appending duplicates (Chroma upserts by id).
    """
    docs = load_employee_documents()
    ids = [doc.metadata["name"] for doc in docs]

    # Sanity check: names must be unique for ids to work as upsert keys.
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate employee names found; ids must be unique.")

    embeddings = get_embeddings()

    # from_documents: embed every card and write vectors + metadata to disk.
    # In langchain-chroma 1.x, passing persist_directory saves automatically.
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    return store


def get_store() -> Chroma:
    """OPEN the already-built store for querying (used by later phases).

    Does NOT re-embed anything — it just connects to ./chroma_db. Call build_index()
    first (once) to create it.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )


if __name__ == "__main__":
    print(f"Building index -> {PERSIST_DIR}  (model: {EMBED_MODEL})")
    store = build_index()

    count = store._collection.count()
    print(f"Indexed {count} employees.")

    # Prove it works: a purely SEMANTIC search (no metadata filter yet).
    # Note the query uses words that do NOT appear in the data on purpose.
    query = "who keeps the network reliable and monitored?"
    print(f"\nSemantic search test — query: {query!r}")
    for doc in store.similarity_search(query, k=3):
        print(f"  - {doc.metadata['name']:20s} | {doc.metadata['technology']}")
