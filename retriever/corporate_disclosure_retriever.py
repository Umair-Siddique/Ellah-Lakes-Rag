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

NAMESPACE = "corporate_disclosure"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMENSION = 3072

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FILTER_PROMPT = """You produce Pinecone metadata filters as pure JSON (no prose).

Allowed fields and types (must respect these):
- doc_type: string (e.g., board_meeting_notice, agm_notice, director_appointment, regulatory_notice, financial_results)
- event_category: string (e.g., meeting, appointment, resignation, fundraise, acquisition, governance)
- meeting_type: string (board_meeting, agm, egm, court_meeting)
- year: number (e.g., 2020)
- disclosure_date: string YYYY-MM-DD (use $eq only; do NOT use $gt/$gte/$lt/$lte)
- meeting_date: string YYYY-MM-DD (use $eq only)
- meeting_time: string HH:MM (24h, use $eq only)
- transaction_type: string (e.g., private_placement, cross_deal)
- transaction_amount: number
- currency: string (e.g., NGN, USD)
- counterparties: list of strings
- regulatory_bodies: list of strings
- key_people_events: list of strings (names/roles/events)
- agenda_items: list of strings

Pinecone filter rules:
- Use equality on strings ({"field": "value"}) or operators like $eq/$in where useful.
- For list fields, use $in to match any of the strings.
- For numbers, you may use $eq, $gte, $lte as needed.
- Do NOT apply range operators to date strings; use exact match on dates.
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
        # Extract JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
        filt = json.loads(content)
        if not isinstance(filt, dict):
            return None
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
                if k in ("$and", "$or") and isinstance(v, list):
                    out[k] = [clean(item) for item in v]
                    continue
                # If v is a dict with a single range op but value is a date string, flatten to equality.
                if isinstance(v, dict) and len(v) == 1:
                    op, val = next(iter(v.items()))
                    if op in ("$gte", "$lte", "$gt", "$lt") and isinstance(val, str) and "-" in val:
                        out[k] = val  # switch to equality
                        continue
                out[k] = clean(v)
            return out
        elif isinstance(node, list):
            return [clean(x) for x in node]
        else:
            return node

    return clean(filt)


def search_corporate_disclosures(
    query: str,
    top_k: int = 5,
    filter_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Dense search with LLM-generated metadata filter."""
    client, index = _init_clients()

    vector = _embed_query(query, client)
    flt = _generate_filter(query, client, model=filter_model)
    if flt:
        flt = _sanitize_filter(flt)

    search_kwargs: Dict[str, Any] = {
        "namespace": NAMESPACE,
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True,
    }
    if flt:
        search_kwargs["filter"] = flt

    try:
        res = index.query(**search_kwargs)
        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    except Exception as exc:
        logger.warning("Filtered query failed (%s). Retrying without filter.", exc)
        res = index.query(
            namespace=NAMESPACE,
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])

    # If filter returned zero, retry without filter to avoid empty results.
    if not matches and flt:
        res = index.query(
            namespace=NAMESPACE,
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])

    return matches


def format_match(match: Dict[str, Any]) -> str:
    md = match.get("metadata", {}) or {}
    doc_id = match.get("id")
    score = match.get("score")
    title = md.get("title")
    doc_type = md.get("doc_type")
    disclosure_date = md.get("disclosure_date")
    summary = md.get("summary")
    return (
        f"- id: {doc_id} | score: {score:.4f}\n"
        f"  title: {title}\n"
        f"  doc_type: {doc_type} | disclosure_date: {disclosure_date}\n"
        f"  summary: {summary}"
    )


if __name__ == "__main__":
    print("Corporate disclosure retriever. Enter queries (empty line to exit).")
    while True:
        try:
            user_q = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_q:
            break
        results = search_corporate_disclosures(user_q, top_k=5)
        if not results:
            print("No results.")
            continue
        for match in results:
            print(format_match(match))

