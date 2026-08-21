"""
Phase 9 — Schema inference (the brain that makes the RAG schema-agnostic).

THE PROBLEM this solves
-----------------------
Phases 1-8 HARDCODED the domain: exactly four columns (name/manager/project/
technology) were baked into data.py and retrieval.py. That made the system accurate
but BRITTLE — drop in a sheet with {dob, salary, grade, mentor, notes, ...} and it
breaks, because nothing knows what those columns mean or how to search them.

THE FIX
-------
Stop hardcoding. Look at ANY DataFrame and, per column, DECIDE A ROLE from the data
itself. Everything downstream (the embedded card, the metadata dict, the router's
known-values, the LLM router prompt) is then generated FROM this inferred schema —
so it can never drift out of sync with the data, and it adapts to any sheet.

THE FIVE ROLES (each maps to a different retrieval strategy)
------------------------------------------------------------
    identity     the row's label ("who is X").         -> the name we cite
    categorical  few distinct text values (manager,    -> EXACT metadata filter
                 grade, dept, mentor).                     + COUNT/aggregate
    numeric      int/float column (salary, years_exp,  -> RANGE filter (>,<,between)
                 age).                                     + avg/min/max/sum
    date         parseable dates (dob, joining_date).   -> date-RANGE filter
    freetext     high-cardinality prose (notes, bio).   -> SEMANTIC only (embedded)

Run it to see what it infers from the current sheet:
    python src/schema.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


# --- Role names (string constants so callers don't hardcode literals) --------
IDENTITY = "identity"
CATEGORICAL = "categorical"
NUMERIC = "numeric"
DATE = "date"
FREETEXT = "freetext"

# --- Tunable thresholds (the ONLY "magic numbers"; documented so you can tune) -
# A text column is CATEGORICAL (worth exact-filtering on) when it has few enough
# distinct values. We test BOTH an absolute cap and a ratio, because "few" scales
# with table size: 30 distinct is categorical in 97 rows, not in 100 rows of 100.
CATEGORICAL_MAX_DISTINCT = 40      # absolute: at most this many distinct values
CATEGORICAL_MAX_RATIO = 0.5        # relative: distinct/rows below this
# A column counts as DATE/NUMERIC only if most non-null cells parse as such.
PARSE_SUCCESS_RATIO = 0.8


@dataclass
class ColumnProfile:
    """What we inferred about ONE column."""
    name: str
    role: str
    distinct: list[str] = field(default_factory=list)   # categorical: the values
    minimum: float | None = None                        # numeric/date: range
    maximum: float | None = None

    def __repr__(self) -> str:
        extra = ""
        if self.role == CATEGORICAL:
            extra = f" ({len(self.distinct)} values)"
        elif self.role in (NUMERIC, DATE):
            extra = f" [{self.minimum} .. {self.maximum}]"
        return f"<{self.name}: {self.role}{extra}>"


@dataclass
class SchemaProfile:
    """The inferred schema for the whole sheet — the object every other module reads."""
    columns: dict[str, ColumnProfile]
    identity: str                                        # the label column's name

    # --- Convenience views the retriever/data layer ask for ------------------
    def names_of(self, role: str) -> list[str]:
        return [c.name for c in self.columns.values() if c.role == role]

    @property
    def categorical(self) -> list[str]:
        return self.names_of(CATEGORICAL)

    @property
    def numeric(self) -> list[str]:
        return self.names_of(NUMERIC)

    @property
    def date(self) -> list[str]:
        return self.names_of(DATE)

    @property
    def freetext(self) -> list[str]:
        return self.names_of(FREETEXT)

    def distinct_values(self) -> dict[str, list[str]]:
        """{column: [values]} for every categorical column — the 'known vocabulary'
        the router validates against (the generalization of the old _known_values)."""
        return {c.name: c.distinct for c in self.columns.values() if c.role == CATEGORICAL}

    def describe(self) -> str:
        """A one-line-per-column human summary — shown at startup / on upload so the
        user SEES what the system understood (great demo + debugging aid)."""
        lines = [f"Detected schema ({len(self.columns)} columns, identity = {self.identity!r}):"]
        for c in self.columns.values():
            lines.append(f"  - {c.name:22s} -> {c.role}"
                         + (f"  e.g. {c.distinct[:4]}" if c.role == CATEGORICAL else "")
                         + (f"  range {c.minimum}..{c.maximum}" if c.role in (NUMERIC, DATE) else ""))
        return "\n".join(lines)

    # --- Persistence: retrieval runs in a separate process from indexing, so the
    #     inferred schema is saved to JSON at build time and reloaded at query time. -
    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "columns": {
                name: {"role": c.role, "distinct": c.distinct,
                       "minimum": c.minimum, "maximum": c.maximum}
                for name, c in self.columns.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SchemaProfile":
        cols = {
            name: ColumnProfile(name=name, role=cd["role"],
                                distinct=cd.get("distinct", []),
                                minimum=cd.get("minimum"), maximum=cd.get("maximum"))
            for name, cd in d["columns"].items()
        }
        return cls(columns=cols, identity=d["identity"])

    def save(self, path) -> None:
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "SchemaProfile | None":
        import json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return None
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


# ============================================================================
# INFERENCE — the actual detection logic
# ============================================================================

def _looks_numeric(s: pd.Series) -> bool:
    """True if most non-null cells parse as numbers (handles clean numeric columns,
    and simple stringy numbers like '5' or '20000')."""
    non_null = s.dropna()
    if non_null.empty:
        return False
    if pd.api.types.is_numeric_dtype(non_null):
        return True
    parsed = pd.to_numeric(non_null, errors="coerce")
    return parsed.notna().mean() >= PARSE_SUCCESS_RATIO


def _looks_date(s: pd.Series) -> bool:
    """True if most non-null cells parse as dates. We test AFTER numeric so a plain
    integer column (salary) isn't mistaken for a date."""
    non_null = s.dropna()
    if non_null.empty:
        return False
    if pd.api.types.is_datetime64_any_dtype(non_null):
        return True
    # Only attempt on object/string columns; avoid coercing pure ints to epoch dates.
    if pd.api.types.is_numeric_dtype(non_null):
        return False
    # Quick reject: if cells have no date-ish separators, don't even try (avoids
    # dateutil parsing names like "Ishan Sharma" one-by-one + its noisy warning).
    sample = non_null.astype(str)
    if not sample.str.contains(r"[/\-.]|\d{4}", regex=True).any():
        return False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", dayfirst=False)
    return parsed.notna().mean() >= PARSE_SUCCESS_RATIO


def _classify(name: str, s: pd.Series, n_rows: int, identity_taken: bool) -> ColumnProfile:
    """Decide ONE column's role. Order matters: identity -> numeric -> date ->
    categorical (low-cardinality text) -> freetext (everything else)."""
    non_null = s.dropna()

    # NUMERIC (before date, so 20000 isn't read as a timestamp)
    if _looks_numeric(s):
        nums = pd.to_numeric(non_null, errors="coerce").dropna()
        return ColumnProfile(name, NUMERIC,
                             minimum=float(nums.min()) if not nums.empty else None,
                             maximum=float(nums.max()) if not nums.empty else None)

    # DATE
    if _looks_date(s):
        dts = pd.to_datetime(non_null, errors="coerce").dropna()
        return ColumnProfile(name, DATE,
                             minimum=float(dts.min().timestamp()) if not dts.empty else None,
                             maximum=float(dts.max().timestamp()) if not dts.empty else None)

    # From here the column is text. Distinct count decides categorical vs freetext.
    distinct = sorted({str(v).strip() for v in non_null})
    ratio = (len(distinct) / n_rows) if n_rows else 1.0

    # IDENTITY: the first all-unique text column becomes the row label.
    if not identity_taken and len(distinct) == len(non_null) and len(distinct) == n_rows:
        return ColumnProfile(name, IDENTITY, distinct=[])

    # CATEGORICAL: few distinct values -> worth exact-filtering / counting on.
    if len(distinct) <= CATEGORICAL_MAX_DISTINCT and ratio <= CATEGORICAL_MAX_RATIO:
        # longest-first so "R&S (Routing & Switching)" is matched before a short token
        distinct = sorted(distinct, key=len, reverse=True)
        return ColumnProfile(name, CATEGORICAL, distinct=distinct)

    # FREETEXT: high-cardinality prose -> semantic only (not a filter).
    return ColumnProfile(name, FREETEXT)


def infer_schema(df: pd.DataFrame) -> SchemaProfile:
    """Introspect a DataFrame and return its SchemaProfile.

    Identity pick is deterministic: the FIRST all-unique text column (usually the
    name/ID in column 1). If none is all-unique, we fall back to the first column so
    there's always a label to cite.
    """
    n_rows = len(df)
    columns: dict[str, ColumnProfile] = {}
    identity_taken = False

    for col in df.columns:
        prof = _classify(str(col), df[col], n_rows, identity_taken)
        if prof.role == IDENTITY:
            identity_taken = True
        columns[str(col)] = prof

    # Fallback: no all-unique column found -> make the first column the identity.
    if not identity_taken:
        first = str(df.columns[0])
        columns[first] = ColumnProfile(first, IDENTITY, distinct=[])

    identity = next(c.name for c in columns.values() if c.role == IDENTITY)
    return SchemaProfile(columns=columns, identity=identity)


if __name__ == "__main__":
    # Show what we infer from the CURRENT sheet (should be: identity + 3 categoricals).
    from data import load_dataframe   # noqa: E402
    df = load_dataframe()
    schema = infer_schema(df)
    print(schema.describe())
    print("\ncategorical:", schema.categorical)
    print("numeric:    ", schema.numeric)
    print("date:       ", schema.date)
    print("freetext:   ", schema.freetext)
