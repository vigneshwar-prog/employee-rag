"""
Phase 1 (+ Phase 9) — Data loading & transformation, now SCHEMA-AGNOSTIC.

Turns ANY Excel sheet into a list of LangChain `Document` objects:
  - one Document per row
  - page_content = a human-readable "card" built from WHATEVER columns exist (embedded)
  - metadata     = every filterable column (categorical/numeric/date) (powers filtering)

Why both? Our retriever is HYBRID:
  - the card's embedding answers fuzzy/semantic questions ("who knows automation?")
  - the metadata answers exact/range questions ("grade == A", "salary > 20 lakhs")

Phase 9 change: the four columns are NO LONGER hardcoded. We infer a SchemaProfile
(src/schema.py) from the sheet and drive card-building + metadata from it, so the
same code works on {name,manager,project,technology} OR {name,dob,salary,grade,notes}.

Run it to inspect what it produces:
    python src/data.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from schema import (SchemaProfile, infer_schema,
                    CATEGORICAL, NUMERIC, DATE, FREETEXT)

# --- Config: where the data lives -------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCEL_PATH = PROJECT_ROOT / "data" / "employees.xlsx"
# sheet_name=0 -> the FIRST sheet, whatever it's called (no longer hardcoded to
# "Employee Mapping"). Any uploaded workbook's first sheet just works.
DEFAULT_SHEET = 0

# Value used when a CATEGORICAL cell is missing (numerics/dates are left null, not faked).
UNASSIGNED = "Unassigned"


def load_dataframe(path: Path = EXCEL_PATH, sheet=DEFAULT_SHEET) -> pd.DataFrame:
    """Read ANY Excel sheet and do MINIMAL, schema-agnostic cleaning.

    We can't fill/strip by hardcoded column name anymore, so:
      1. read the first sheet
      2. strip whitespace on every *text* column (matters for exact filtering)
    Blank-filling is deferred to build_documents, which knows each column's ROLE
    (only categoricals get "Unassigned"; numbers/dates stay null).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Excel not found at {path.resolve()}. Copy it in before running."
        )

    df = pd.read_excel(path, sheet_name=sheet)

    # Strip whitespace on object/string columns generically.
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()
    return df


def _fmt_value(profile, raw) -> str:
    """Render a cell for the CARD text (human-readable prose)."""
    if pd.isna(raw):
        return UNASSIGNED
    if profile.role == DATE:
        # show a clean date, not a timestamp
        try:
            return pd.to_datetime(raw).date().isoformat()
        except Exception:
            return str(raw)
    return str(raw).strip()


def row_to_card(row: pd.Series, schema: SchemaProfile) -> str:
    """Build the readable sentence that gets EMBEDDED, from whatever columns exist.

    Kept prose-like on purpose (embedding models were trained on prose): we lead with
    the identity, then "Field: value" clauses for each attribute, and append any
    freetext (notes/bio) verbatim at the end so it's fully searchable semantically.
    """
    ident = schema.identity
    ident_val = _fmt_value(schema.columns[ident], row.get(ident))

    clauses, freetext_bits = [], []
    for col, prof in schema.columns.items():
        if col == ident:
            continue
        val = _fmt_value(prof, row.get(col))
        if prof.role == FREETEXT:
            if val and val != UNASSIGNED:
                freetext_bits.append(f"{col}: {val}")
        else:
            clauses.append(f"{col} is {val}")

    sentence = f"{ident_val} — " + "; ".join(clauses) + "." if clauses else f"{ident_val}."
    if freetext_bits:
        sentence += " " + ". ".join(freetext_bits) + "."
    return sentence


def _meta_value(profile, raw):
    """Render a cell for METADATA (must be a Chroma-safe SCALAR: str/int/float/bool).

    Dates are stored as an epoch FLOAT so Chroma's `$gt`/`$lt` range filters work.
    Numerics are stored as float. Categoricals as stripped strings. Missing ->
    "Unassigned" for categoricals, omitted for numeric/date (don't fake a number).
    """
    if profile.role == NUMERIC:
        num = pd.to_numeric(raw, errors="coerce")
        return None if pd.isna(num) else float(num)
    if profile.role == DATE:
        dt = pd.to_datetime(raw, errors="coerce")
        return None if pd.isna(dt) else float(dt.timestamp())
    # categorical (identity handled separately)
    if pd.isna(raw):
        return UNASSIGNED
    return str(raw).strip()


def build_documents(df: pd.DataFrame, schema: SchemaProfile | None = None) -> list[Document]:
    """Convert each row into one LangChain Document (card + scalar metadata),
    driven by the inferred schema. Freetext columns are embedded (in the card) but
    NOT put in metadata (keeps the filterable metadata lean and scalar)."""
    if schema is None:
        schema = infer_schema(df)

    documents: list[Document] = []
    for _, row in df.iterrows():
        ident = schema.identity
        meta = {"name": str(row.get(ident)).strip()}  # 'name' = the identity, kept as the stable key

        for col, prof in schema.columns.items():
            if col == ident or prof.role == FREETEXT:
                continue  # identity already in 'name'; freetext isn't filterable
            v = _meta_value(prof, row.get(col))
            if v is not None:
                meta[col] = v

        documents.append(Document(page_content=row_to_card(row, schema), metadata=meta))
    return documents


def load_employee_documents() -> tuple[list[Document], SchemaProfile]:
    """Public entry point: Excel -> cleaned DataFrame -> (Documents, SchemaProfile).

    NOTE (Phase 9): now returns the inferred schema alongside the docs, so index.py
    can persist it and retrieval.py can route against it. Callers updated accordingly.
    """
    df = load_dataframe()
    schema = infer_schema(df)
    return build_documents(df, schema), schema


if __name__ == "__main__":
    df = load_dataframe()
    schema = infer_schema(df)
    print(f"Loaded {len(df)} rows.\n")
    print(schema.describe())

    docs = build_documents(df, schema)
    print(f"\nBuilt {len(docs)} Documents. First one:\n")
    print("page_content:", docs[0].page_content)
    print("metadata:    ", docs[0].metadata)
