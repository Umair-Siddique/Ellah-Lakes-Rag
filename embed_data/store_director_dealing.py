from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from openai import OpenAI
from pinecone import Pinecone

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config

DATA_DIR = ROOT_DIR / "processed" / "Director_Dealings"
NAMESPACE = "director_dealing"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMENSION = 3072

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _chunk(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _load_documents() -> List[Dict[str, Any]]:
    """Load all processed director dealing JSON payloads."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Processed directory not found: {DATA_DIR}")

    documents: List[Dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            documents.append(json.load(handle))
    logger.info("Loaded %s processed director dealing file(s)", len(documents))
    return documents


def _build_text_to_embed(payload: Dict[str, Any]) -> str:
    """Compose focused text for embedding."""
    parts: List[str] = []

    def add(label: str, value: Any) -> None:
        if value not in (None, "", []):
            parts.append(f"{label}: {value}")

    add("Company", payload.get("company_name"))
    add("Insider name", payload.get("insider_name"))
    add("Position/status", payload.get("position_status"))
    add("Insider type", payload.get("insider_type"))
    add("Transaction nature", payload.get("transaction_nature"))
    add("Instrument type", payload.get("instrument_type"))
    add("ISIN", payload.get("isin"))

    add("Price per share", "; ".join([str(x) for x in payload.get("price_per_share") or []]))
    add("Volume", "; ".join([str(x) for x in payload.get("volume") or []]))
    add("Aggregated volume", payload.get("aggregated_volume"))
    add("Aggregated price", payload.get("aggregated_price"))
    add("Currency", payload.get("currency"))

    add("Transaction dates", "; ".join(payload.get("transaction_dates") or []))
    add("Announcement date", payload.get("announcement_date"))
    add("Place of transaction", payload.get("place_of_transaction") or payload.get("announcement_location"))

    # Detailed summary carries the richest context.
    add("Detailed summary", payload.get("detailed_summary") or payload.get("short_summary"))

    return "\n".join(parts)


def _build_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare Pinecone-compliant metadata."""
    scalar_keys = [
        "company_name",
        "rc_number",
        "legal_entity_identifier",
        "insider_name",
        "position_status",
        "insider_type",
        "is_corporate_insider",
        "transaction_nature",
        "instrument_type",
        "isin",
        "aggregated_volume",
        "aggregated_price",
        "currency",
        "announcement_date",
        "announcement_location",
        "place_of_transaction",
        "signatory_name",
        "signatory_designation",
        "year",
        "month",
        "document_type",
        "pdf_source",
    ]

    list_keys = [
        "price_per_share",
        "volume",
        "transaction_dates",
    ]

    metadata: Dict[str, Any] = {}

    for key in scalar_keys:
        value = payload.get(key)
        if value not in (None, "", []):
            metadata[key] = value

    for key in list_keys:
        value = payload.get(key)
        if value:
            # Pinecone list metadata must be list of strings.
            metadata[key] = [str(item) for item in value]

    # Store summaries and the exact embedded text for downstream LLM use.
    if payload.get("short_summary"):
        metadata["short_summary"] = payload["short_summary"]
    if payload.get("detailed_summary"):
        metadata["summary"] = payload["detailed_summary"]

    return metadata


def _get_embedding(texts: List[str], client: OpenAI) -> List[List[float]]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIMENSION,
    )
    return [item.embedding for item in response.data]


def _init_index() -> Any:
    if not Config.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is missing; set it in your .env file.")
    if not Config.INDEX_NAME:
        raise RuntimeError("INDEX_NAME is missing; set it in your .env file.")

    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    index_list = pc.list_indexes()
    index_names = set(index_list) if isinstance(index_list, list) else set(index_list.names())
    if Config.INDEX_NAME not in index_names:
        raise RuntimeError(
            f"Pinecone index '{Config.INDEX_NAME}' not found. Create it first before running this script."
        )
    return pc.Index(Config.INDEX_NAME)


def upsert_director_dealings(batch_size: int = 50) -> None:
    documents = _load_documents()
    if not documents:
        logger.warning("No director dealing documents found to embed.")
        return

    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    index = _init_index()

    for batch in _chunk(documents, batch_size):
        texts = [_build_text_to_embed(doc) for doc in batch]
        embeddings = _get_embedding(texts, client)

        vectors = []
        for doc, embedding, text in zip(batch, embeddings, texts):
            doc_id = Path(doc.get("pdf_source", "")).stem or doc.get("insider_name") or "director_dealing"
            metadata = _build_metadata(doc)
            metadata["content"] = text
            vectors.append(
                {
                    "id": doc_id,
                    "values": embedding,
                    "metadata": metadata,
                }
            )

        index.upsert(vectors=vectors, namespace=NAMESPACE)
        logger.info(
            "Upserted %s vectors into index '%s' (namespace='%s')",
            len(vectors),
            Config.INDEX_NAME,
            NAMESPACE,
        )

    logger.info("Completed upserting %s director dealing record(s).", len(documents))


if __name__ == "__main__":
    upsert_director_dealings()

