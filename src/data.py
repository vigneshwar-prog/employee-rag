"""
Phase 1 — Data loading & transformation.

Turns the Excel sheet into a list of LangChain `Document` objects:
  - one Document per employee
  - page_content = a human-readable "card" (this is what gets EMBEDDED)
  - metadata      = {name, manager, project, technology} (this powers exact FILTERING)

Why both? Our retriever is HYBRID:
  - the card's embedding answers fuzzy/semantic questions ("who knows networking automation?")
  - the metadata answers exact questions ("list everyone whose manager is Dhruv Sharma")

Run it directly to inspect what it produces:
    python src/data.py
"""

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

# --- Config: where the data lives and what the columns are called ---------
# Anchor paths to the PROJECT ROOT (the parent of this src/ file), so the
# script works no matter which directory you launch it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCEL_PATH = PROJECT_ROOT / "data" / "employees.xlsx"
SHEET_NAME = "Employee Mapping"

# The 4 columns in the sheet. Project/Technology can be blank -> we fill them.
COL_NAME = "Employee Name"
COL_MANAGER = "Manager"
COL_PROJECT = "Project"
COL_TECHNOLOGY = "Technology"

# Value used when Project/Technology is missing (e.g. managers have no project).
UNASSIGNED = "Unassigned"


def load_dataframe(path: Path = EXCEL_PATH, sheet: str = SHEET_NAME) -> pd.DataFrame:
    """Read the Excel sheet and clean it.

    Cleaning steps:
      1. read the sheet with pandas (openpyxl engine, chosen automatically for .xlsx)
      2. strip stray whitespace from text cells
      3. fill missing Project / Technology with "Unassigned"
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Excel not found at {path.resolve()}. "
            "Copy it in (see PROJECT.md Phase 0) before running."
        )

    df = pd.read_excel(path, sheet_name=sheet)

    # Normalize whitespace on every text column so "Dhruv Sharma " == "Dhruv Sharma".
    # This matters a lot for FILTERING later: a trailing space would break exact matches.
    for col in (COL_NAME, COL_MANAGER, COL_PROJECT, COL_TECHNOLOGY):
        df[col] = df[col].astype("string").str.strip()

    # Fill the blanks. Only Project & Technology have missing values (6 each).
    df[COL_PROJECT] = df[COL_PROJECT].fillna(UNASSIGNED)
    df[COL_TECHNOLOGY] = df[COL_TECHNOLOGY].fillna(UNASSIGNED)

    return df


def row_to_card(name: str, manager: str, project: str, technology: str) -> str:
    """Build the readable sentence that will be EMBEDDED for this employee.

    Kept as one natural sentence on purpose: embedding models were trained on
    prose, so a real sentence embeds better than "name=..., manager=..." key-value text.
    """
    return (
        f"{name} is managed by {manager}, "
        f"works on the {project} project, "
        f"specializing in {technology}."
    )


def build_documents(df: pd.DataFrame) -> list[Document]:
    """Convert each cleaned row into one LangChain Document (card + metadata)."""
    documents: list[Document] = []
    for _, row in df.iterrows():
        name = row[COL_NAME]
        manager = row[COL_MANAGER]
        project = row[COL_PROJECT]
        technology = row[COL_TECHNOLOGY]

        doc = Document(
            page_content=row_to_card(name, manager, project, technology),
            metadata={
                "name": name,
                "manager": manager,
                "project": project,
                "technology": technology,
            },
        )
        documents.append(doc)
    return documents


def load_employee_documents() -> list[Document]:
    """Public entry point: Excel -> cleaned DataFrame -> list[Document]."""
    df = load_dataframe()
    return build_documents(df)


if __name__ == "__main__":
    # Inspect what Phase 1 produces, without embedding anything yet.
    df = load_dataframe()
    print(f"Loaded {len(df)} rows.")
    print(f"Missing after fill -> Project: {int(df[COL_PROJECT].eq(UNASSIGNED).sum())} "
          f"marked Unassigned, Technology: {int(df[COL_TECHNOLOGY].eq(UNASSIGNED).sum())} marked Unassigned.")

    docs = build_documents(df)
    print(f"\nBuilt {len(docs)} Documents. First one:\n")
    print("page_content:", docs[0].page_content)
    print("metadata:    ", docs[0].metadata)

    print("\nAn 'Unassigned' example (a manager with no project):\n")
    print("page_content:", docs[95].page_content)
    print("metadata:    ", docs[95].metadata)
