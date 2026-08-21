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
import os

# The embedding model is cached locally (Phase 0). Force HuggingFace OFFLINE so it
# never phones home to check for updates — that network call fails behind the
# corporate SSL-inspecting proxy ("Cannot send a request..."). Must be set BEFORE
# importing huggingface libs, so it goes at the very top.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Belt-and-suspenders: a few HF code paths (e.g. the PEFT adapter probe) ignore the
# OFFLINE flag and still try the network. If they do, point them at the corporate
# CA bundle we saved in Phase 0 so TLS verification succeeds instead of crashing.
_CA = Path(__file__).resolve().parent.parent / "certs" / "keychain-roots.pem"
if _CA.exists():
    os.environ.setdefault("SSL_CERT_FILE", str(_CA))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_CA))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from data import load_employee_documents

# --- Config -----------------------------------------------------------------
# Anchor the store to the PROJECT ROOT so the DB is always ./chroma_db there,
# regardless of the current working directory when you run the script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")
# Where the inferred schema is persisted (Phase 9). Retrieval runs in a separate
# process from indexing, so it reloads the schema from here instead of re-inferring.
SCHEMA_PATH = PROJECT_ROOT / "chroma_db" / "schema.json"
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

    Uses each row's identity value as a STABLE id, so re-running updates the same
    records instead of appending duplicates (Chroma upserts by id).

    Phase 9: load_employee_documents() now ALSO returns the inferred SchemaProfile;
    we persist it (schema.json) so the query-time process can route against it.
    """
    docs, schema = load_employee_documents()
    ids = [doc.metadata["name"] for doc in docs]

    # Sanity check: identity values must be unique for ids to work as upsert keys.
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate identity values found; ids must be unique.")

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
    # Persist the inferred schema next to the vectors so retrieval can reload it.
    schema.save(SCHEMA_PATH)
    return store


def get_schema():
    """Load the persisted SchemaProfile (Phase 9). Returns None if the index was
    built before schema persistence existed (callers then infer/fallback)."""
    from schema import SchemaProfile
    return SchemaProfile.load(SCHEMA_PATH)


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
    print(f"Indexed {count} records.")
    schema = get_schema()
    if schema:
        print("\n" + schema.describe())

    # Prove it works: a purely SEMANTIC search (no metadata filter yet).
    # We print the full card + both scores so the ranking is not a black box:
    #   distance  -> raw L2 distance (question vs card vector). LOWER = closer.
    #   relevance -> LangChain's normalized 0..1 score.         HIGHER = better.
    while True:
        query = input("\nSemantic search test — enter a question (or 'q' to quit): ")
        if query.lower() == "q":
            break
        print(f"\nSemantic search test — query: {query!r}")

        scored = store.similarity_search_with_score(query, k=3)
        relevances = dict(
            (d.metadata["name"], r)
            for d, r in store.similarity_search_with_relevance_scores(query, k=3)
        )
        for rank, (doc, distance) in enumerate(scored, start=1):
            m = doc.metadata
            rel = relevances.get(m["name"], float("nan"))
            print(f"\n  #{rank}  distance={distance:.4f}  relevance={rel:.3f}")
            print(f"      name : {m['name']}")
            print(f"      meta : { {k: v for k, v in m.items() if k != 'name'} }")
            print(f"      card : {doc.page_content}")
