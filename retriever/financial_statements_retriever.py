from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pinecone import Pinecone

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config

NAMESPACE = "financial_statements"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMENSION = 3072

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FILTER_PROMPT = """You produce Pinecone metadata filters as pure JSON (no prose).

Allowed fields and types (must respect these):
- companyName: string (e.g., "ELLAH LAKES PLC")
- rcNumber: string (e.g., "RC299748", "RC34296")
- reportTitle: string (report title/description)
- consolidationLevel: string (consolidated, standalone, etc.)
- statementType: string (e.g., "financial position", "comprehensive income", "cash flow")
- financialPeriodLabel: string (e.g., "as at January 31, 2013", "for the year ended")
- periodStart: string YYYY-MM-DD (use $eq only; do NOT use $gt/$gte/$lt/$lte)
- periodEnd: string YYYY-MM-DD (use $eq only; do NOT use $gt/$gte/$lt/$lte)
- periodLengthMonths: number (3, 6, 9, 12 etc.)
- fiscalYear: number (e.g., 2013, 2020, 2024)
- auditedStatus: string (audited, unaudited)
- currency: string (e.g., "NGN", "USD")
- chunkPage: number (page number in the document)
- topics: list of strings (document topics/tags)
- keyFigures: list of strings (financial metrics in format "key:value")

Pinecone filter rules:
- Use equality on strings ({"field": "value"}) or operators like $eq/$in where useful.
- For list fields (topics, keyFigures), use $in to match any of the strings.
- For numbers (fiscalYear, periodLengthMonths, chunkPage), you may use $eq, $gte, $lte, $gt, $lt as needed.
- Do NOT apply range operators to date strings (periodStart, periodEnd); use exact match with $eq only.
- For fiscal year queries like "2023 statements" or "latest 2024 reports", use fiscalYear filter.
- For quarterly/semi-annual queries, use periodLengthMonths (3=quarterly, 6=semi-annual, 9=nine months, 12=annual).
- For audited vs unaudited, use auditedStatus filter.
- If no filter is helpful, return {}.

Return ONLY a JSON object compatible with Pinecone filtering.
"""


def _init_clients() -> tuple[OpenAI, Any]:
    if not Config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing; set it in your .env file.")
    if not Config.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is missing; set it in your .env file.")
    if not Config.INDEX_NAME:
        raise RuntimeError("INDEX_NAME is missing; set it in your .env file.")

    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    index = pc.Index(Config.INDEX_NAME)
    return client, index


def _embed_query(query: str, client: OpenAI) -> List[float]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
        dimensions=EMBED_DIMENSION,
    )
    return response.data[0].embedding


def _generate_filter(query: str, client: OpenAI, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Use LLM to generate Pinecone metadata filter from natural language query."""
    messages = [
        {"role": "system", "content": FILTER_PROMPT},
        {"role": "user", "content": f"User query: {query}\n\nReturn JSON filter:"},
    ]
    try:
        response = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=messages,
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        # Extract JSON object from markdown or plain text
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
        filt = json.loads(content)
        if not isinstance(filt, dict):
            return None
        logger.info("Generated filter: %s", filt)
        return filt
    except Exception as exc:
        logger.warning("Filter generation failed; continuing without filter. Error: %s", exc)
        return None


def _sanitize_filter(filt: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure Pinecone filter is valid: remove range ops on date strings; keep simple equality."""
    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for k, v in node.items():
                # Handle logical operators
                if k in ("$and", "$or") and isinstance(v, list):
                    out[k] = [clean(item) for item in v]
                    continue
                # If value is a dict with a single range op but value is a date string, flatten to equality
                if isinstance(v, dict) and len(v) == 1:
                    op, val = next(iter(v.items()))
                    # Detect date strings and convert range ops to equality
                    if op in ("$gte", "$lte", "$gt", "$lt") and isinstance(val, str) and "-" in val:
                        out[k] = val  # switch to equality match
                        logger.info("Sanitized date range filter on '%s' to equality: %s", k, val)
                        continue
                out[k] = clean(v)
            return out
        elif isinstance(node, list):
            return [clean(x) for x in node]
        else:
            return node

    return clean(filt)


def search_financial_statements(
    query: str,
    top_k: int = 5,
    filter_model: Optional[str] = None,
    manual_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Dense semantic search with optional LLM-generated or manual metadata filter.
    
    Args:
        query: Natural language search query
        top_k: Number of results to return
        filter_model: Optional LLM model for filter generation (default: gpt-4o-mini)
        manual_filter: Optional manual Pinecone filter dict (overrides LLM filter)
    
    Returns:
        List of match dictionaries with metadata
    """
    client, index = _init_clients()

    # Embed the query
    vector = _embed_query(query, client)
    
    # Determine filter: manual takes precedence over LLM-generated
    flt = manual_filter
    if flt is None:
        flt = _generate_filter(query, client, model=filter_model)
    
    if flt:
        flt = _sanitize_filter(flt)
        logger.info("Using filter: %s", flt)

    # Build search kwargs
    search_kwargs: Dict[str, Any] = {
        "namespace": NAMESPACE,
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True,
    }
    if flt:
        search_kwargs["filter"] = flt

    # Execute search with filter
    try:
        res = index.query(**search_kwargs)
        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
        logger.info("Found %s matches with filter", len(matches))
    except Exception as exc:
        logger.warning("Filtered query failed (%s). Retrying without filter.", exc)
        # Retry without filter on error
        res = index.query(
            namespace=NAMESPACE,
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])

    # If filter returned zero results, retry without filter to avoid empty results
    if not matches and flt:
        logger.info("No matches with filter. Retrying without filter...")
        res = index.query(
            namespace=NAMESPACE,
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
        logger.info("Found %s matches without filter", len(matches))

    return matches


def format_match(match: Dict[str, Any]) -> str:
    """Format a single match for display."""
    md = match.get("metadata", {}) or {}
    doc_id = match.get("id")
    score = match.get("score")
    
    company = md.get("companyName")
    rc_number = md.get("rcNumber")
    report_title = md.get("reportTitle")
    statement_type = md.get("statementType")
    period = md.get("financialPeriodLabel")
    fiscal_year = md.get("fiscalYear")
    audited = md.get("auditedStatus")
    currency = md.get("currency")
    summary = md.get("summary", "")
    
    # Truncate summary for display
    if summary and len(summary) > 300:
        summary = summary[:300] + "..."
    
    output = [
        f"- ID: {doc_id} | Score: {score:.4f}",
        f"  Company: {company} ({rc_number or 'N/A'})",
        f"  Report: {report_title}",
        f"  Statement Type: {statement_type or 'N/A'}",
        f"  Period: {period} | Fiscal Year: {fiscal_year or 'N/A'}",
        f"  Status: {audited or 'N/A'} | Currency: {currency or 'N/A'}",
    ]
    
    if summary:
        output.append(f"  Summary: {summary}")
    
    return "\n".join(output)


if __name__ == "__main__":
    print("=" * 70)
    print("Financial Statements Retriever")
    print("=" * 70)
    print("Enter natural language queries to search financial statements.")
    print("Examples:")
    print("  - Show me 2024 quarterly financial statements")
    print("  - What were the audited annual reports for 2023?")
    print("  - Find statements of financial position from 2022")
    print("  - Show me cash flow statements with negative operating cash")
    print("Empty line to exit.")
    print("=" * 70)
    
    while True:
        try:
            user_q = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not user_q:
            break
        
        results = search_financial_statements(user_q, top_k=5)
        
        if not results:
            print("No results found.")
            continue
        
        print(f"\nFound {len(results)} results:\n")
        for i, match in enumerate(results, 1):
            print(f"\n[Result {i}]")
            print(format_match(match))

