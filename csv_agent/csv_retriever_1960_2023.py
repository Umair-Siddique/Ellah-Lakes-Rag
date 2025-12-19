# pyright: reportMissingImports=false
"""
Lightweight CSV agent using LlamaIndex's PandasQueryEngine.

This mirrors the official example:
https://developers.llamaindex.ai/python/examples/query_engine/pandas_query_engine/
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
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# Choose a single CSV; adjust here if you want the other dataset.
DATASET_PATH = (
    Path(__file__).resolve().parent / "dataset" / "Data Set for 1960 to 2023.csv"
)


def _ensure_openai_llm(model: str = DEFAULT_MODEL) -> None:
    """
    Configure LlamaIndex to use OpenAI LLM.

    OPENAI_API_KEY must be set (config.py loads .env for local dev).
    """
    # Prefer explicitly provided key from Config; fallback to env if missing.
    api_key = Config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment or Config.")
    Settings.llm = OpenAI(model=model, api_key=api_key)


def _load_dataframe(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and tidy the CSV so rows have Year + Quarter + numeric columns.

    The raw file interleaves year rows (e.g., "1960") with quarter rows
    ("Q1", "Q2", ...). We forward-fill the year, keep only quarter rows,
    and coerce numeric columns.
    """
    path = csv_path or DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")

    # Header is on the second line, so use header=1 to skip the title row.
    df = pd.read_csv(path, header=1)

    # Separate Year and Quarter columns.
    raw_year = df["Year"].astype(str)
    df["Quarter"] = raw_year.where(raw_year.str.startswith("Q"))
    df["Year"] = raw_year.where(~raw_year.str.startswith("Q")).ffill()

    # Keep only quarter rows.
    df = df[df["Quarter"].notna()].copy()

    # Cast Year to int when possible.
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # Coerce numeric columns.
    for col in df.columns:
        if col in ("Year", "Quarter"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@lru_cache(maxsize=1)
def get_csv_query_engine(
    *,
    csv_path: Optional[Path] = None,
    verbose: bool = False,
    synthesize_response: bool = True,
) -> PandasQueryEngine:
    """
    Build and memoize a PandasQueryEngine over the chosen CSV.

    Returns a query engine that converts natural language to pandas code,
    executes it, and (optionally) synthesizes a natural language answer.
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

    Example:
        answer = query_csv("What is the earliest year in the dataset?")
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
        python csv_agent/csv_retriever.py
    Then type a question; press Enter on an empty line to quit.
    """
    print("CSV agent ready. Ask a question (empty line to exit).")
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