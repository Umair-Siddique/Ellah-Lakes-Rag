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

NAMESPACE = "director_dealing"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMENSION = 3072

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FILTER_PROMPT = """You produce Pinecone metadata filters as pure JSON (no prose).

Allowed fields and types (must respect these):
- company_name: string
- insider_name: string
- position_status: string
- insider_type: string (Corporate | Individual)
- is_corporate_insider: boolean
- transaction_nature: string (e.g., Sale of shares, Purchase of shares)
- instrument_type: string (e.g., shares)
- isin: string
- currency: string (e.g., NGN, USD)
- aggregated_volume: number
- aggregated_price: number
- transaction_dates: list of strings YYYY-MM-DD (use exact match with $in)
- announcement_date: string YYYY-MM-DD (use $eq only)
- place_of_transaction: string
- announcement_location: string
- signatory_name: string
- signatory_designation: string
- year: number
- month: string (\"01\" .. \"12\")
- document_type: string (insider_dealing_notification)

Pinecone filter rules:
- Use equality on strings ({\"field\": \"value\"}) or $eq/$in where useful.
- For list fields, use $in to match any of the strings.
- For numbers, you may use $eq, $gte, $lte as needed.
- Do NOT apply range operators to date strings; use exact match on dates.
- If no filter helps, return {}.

Return ONLY a JSON object compatible with Pinecone filtering."""


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
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
        filt = json.loads(content)
        if isinstance(filt, dict):
            return filt
        return None
    except Exception as exc:
        logger.warning("Filter generation failed; continuing without filter. Error: %s", exc)
        return None


def _sanitize_filter(filt: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure Pinecone filter validity: avoid range ops on date strings; keep equality."""
    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for k, v in node.items():
                if k in ("$and", "$or") and isinstance(v, list):
                    out[k] = [clean(item) for item in v]
                    continue
                if isinstance(v, dict) and len(v) == 1:
                    op, val = next(iter(v.items()))
                    if op in ("$gte", "$lte", "$gt", "$lt") and isinstance(val, str) and "-" in val:
                        out[k] = val  # treat as equality
                        continue
                out[k] = clean(v)
            return out
        if isinstance(node, list):
            return [clean(x) for x in node]
        return node

    return clean(filt)


def search_director_dealings(
    query: str,
    top_k: int = 5,
    filter_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Dense search with optional LLM-generated metadata filter."""
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

    # If filtered search returns zero, retry without filter.
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
    insider = md.get("insider_name")
    nature = md.get("transaction_nature")
    announcement_date = md.get("announcement_date")
    summary = md.get("summary") or md.get("short_summary")
    return (
        f"- id: {doc_id} | score: {score:.4f}\n"
        f"  insider: {insider} | transaction: {nature} | announcement_date: {announcement_date}\n"
        f"  summary: {summary}"
    )


if __name__ == "__main__":
    print("Director dealing retriever. Enter queries (empty line to exit).")
    while True:
        try:
            user_q = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_q:
            break
        results = search_director_dealings(user_q, top_k=5)
        if not results:
            print("No results.")
            continue
        for match in results:
            print(format_match(match))

