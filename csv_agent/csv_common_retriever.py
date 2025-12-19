# pyright: reportMissingImports=false
"""
Common router for CSV agents.

Decides whether to query the monetary aggregates CSV (1960–1980 quarterly)
or the GDP-by-sector CSV (1960+ annual) based on the user question, then
forwards the query to the appropriate PandasQueryEngine.
"""

import os
import re
import sys
from pathlib import Path
from typing import Literal

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

# Ensure project root is on the import path so `config` and sibling modules resolve.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config  # noqa: E402
from csv_agent.csv_retriever_1960_2023 import query_csv as query_monetary  # noqa: E402
from csv_agent.csv_retriver_from_1960 import query_csv as query_gdp  # noqa: E402


# Default model can be overridden with OPENAI_MODEL env var
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")


def _ensure_openai_llm(model: str = DEFAULT_MODEL) -> None:
    api_key = Config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment or Config.")
    Settings.llm = OpenAI(model=model, api_key=api_key)


def _classify_query(question: str) -> Literal["monetary", "gdp"]:
    """
    Lightweight rule-based router between the two CSVs.

    Monetary cues: quarters (Q1/Q2/Q3/Q4), money aggregates (M1/M2),
    credit, reserve money, PSDD, DMB, CIC, NFA/NCG.

    GDP cues: sector/industry words, GDP, agriculture, services, manufacturing,
    construction, wholesale/retail, transport, telecom, finance, insurance, real estate.
    """
    q = question.lower()

    monetary_terms = [
        "q1",
        "q2",
        "q3",
        "q4",
        "quarter",
        "m1",
        "m2",
        "nfa",
        "ncg",
        "psdd",
        "dmb",
        "cic",
        "reserve money",
        "broad money",
        "narrow money",
        "credit to",
    ]

    gdp_terms = [
        "gdp",
        "sector",
        "agriculture",
        "crop",
        "livestock",
        "forestry",
        "fishing",
        "industry",
        "manufacturing",
        "petroleum",
        "gas",
        "mining",
        "construction",
        "building",
        "wholesale",
        "retail",
        "trade",
        "services",
        "transport",
        "communication",
        "telecom",
        "finance",
        "insurance",
        "real estate",
        "business services",
        "hotel",
        "restaurant",
        "utilities",
    ]

    if any(term in q for term in monetary_terms):
        return "monetary"
    if any(term in q for term in gdp_terms):
        return "gdp"

    # Heuristic: presence of a quarter-like token (Q#) -> monetary
    if re.search(r"\bq[1-4]\b", q):
        return "monetary"

    # Fallback: prefer GDP (wider date range) if no strong signal
    return "gdp"


def query_common_with_source(question: str, **engine_kwargs) -> tuple[str, str]:
    """
    Route the question to the appropriate CSV and return the answer plus source tag.
    """
    _ensure_openai_llm()
    target = _classify_query(question)
    if target == "monetary":
        return query_monetary(question, **engine_kwargs), "monetary"
    return query_gdp(question, **engine_kwargs), "gdp"


def query_common(question: str, **engine_kwargs) -> str:
    answer, _ = query_common_with_source(question, **engine_kwargs)
    return answer


__all__ = ["query_common", "query_common_with_source"]


if __name__ == "__main__":
    """
    Terminal loop: routes each question to the appropriate CSV retriever.
    """
    print("CSV common router ready. Ask a question (empty line to exit).")
    while True:
        try:
            user_q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_q:
            break
        try:
            answer = query_common(user_q)
            print(answer)
        except Exception as exc:
            print(f"Error: {exc}")
