# pyright: reportMissingImports=false
"""
CSV agent for `New Data Set from 1960.csv` using LlamaIndex PandasQueryEngine.

Loads the wide GDP-by-sector dataset, tidies it to long format
(Activity_Sector, Year, Value), and exposes a simple query loop.
"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from llama_index.core import Settings
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.openai import OpenAI

# Ensure project root is on the import path so `config` resolves when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config

# Default model can be overridden with OPENAI_MODEL env var
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

# Target dataset for this retriever
DATASET_PATH = (
    Path(__file__).resolve().parent / "dataset" / "New Data Set from 1960.csv"
)


def _ensure_openai_llm(model: str = DEFAULT_MODEL) -> None:
    """
    Configure LlamaIndex to use OpenAI LLM.
    """
    api_key = Config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment or Config.")
    Settings.llm = OpenAI(model=model, api_key=api_key)


def _load_dataframe(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and tidy the wide GDP dataset into long format.

    - Skip the first two title/blank rows (header starts at row index 2).
    - Strip column names, detect year columns, and melt to long form.
    - Coerce values to numeric.
    """
    path = csv_path or DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")

    # Header is on the third row (0-based index 2)
    df = pd.read_csv(path, header=2)

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]
    if "Activity Sector" in df.columns:
        df = df.rename(columns={"Activity Sector": "Activity_Sector"})

    # Identify year-like columns
    year_cols = []
    for col in df.columns:
        if col == "Activity_Sector":
            continue
        col_clean = col.strip().split()[0]  # handle "2010 2" -> "2010"
        if col_clean.isdigit():
            year_cols.append(col)

    # Melt to long format: Activity_Sector, Year, Value
    df_long = df.melt(
        id_vars=["Activity_Sector"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value",
    )

    # Normalize Year and Value
    df_long["Year"] = df_long["Year"].apply(lambda x: str(x).strip().split()[0])
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce").astype("Int64")
    df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")

    # Drop rows with no sector or year
    df_long = df_long[df_long["Activity_Sector"].notna() & df_long["Year"].notna()]

    return df_long


@lru_cache(maxsize=1)
def get_csv_query_engine(
    *,
    csv_path: Optional[Path] = None,
    verbose: bool = False,
    synthesize_response: bool = True,
) -> PandasQueryEngine:
    """
    Build and memoize a PandasQueryEngine over the GDP dataset.
    """
    _ensure_openai_llm()
    df = _load_dataframe(csv_path)
    return PandasQueryEngine(
        df=df,
        verbose=verbose,
        synthesize_response=synthesize_response,
    )


def query_csv(question: str, **engine_kwargs) -> str:
    """
    Convenience helper to run a single NL query against the CSV.
    """
    engine = get_csv_query_engine(**engine_kwargs)
    response = engine.query(question)
    return str(response)


__all__ = [
    "get_csv_query_engine",
    "query_csv",
]


if __name__ == "__main__":
    """
    Simple terminal loop for ad-hoc queries.

    Usage:
        python csv_agent/csv_retriver_from_1960.py
    Then type a question; press Enter on an empty line to quit.
    """
    print("CSV GDP agent ready. Ask a question (empty line to exit).")
    while True:
        try:
            user_q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_q:
            break
        try:
            answer = query_csv(user_q)
            print(answer)
        except Exception as exc:
            print(f"Error: {exc}")
