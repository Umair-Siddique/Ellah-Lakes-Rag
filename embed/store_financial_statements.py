from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from openai import OpenAI
from pinecone import Pinecone

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config

DATA_DIR = ROOT_DIR / "processed" / "Financials_Statements"
NAMESPACE = "financial_statements"
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
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Processed directory not found: {DATA_DIR}")
    docs: List[Dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            docs.append(json.load(handle))
    logger.info("Loaded %s processed financial statement file(s)", len(docs))
    return docs


def _build_text_to_embed(payload: Dict[str, Any]) -> str:
    """Compose a dense, retrieval-friendly text block."""
    parts: List[str] = []

    def add(label: str, value: Any) -> None:
        if value not in (None, "", []):
            parts.append(f"{label}: {value}")

    add("Company", payload.get("companyName"))
    add("Report title", payload.get("reportTitle"))
    add("Statement type", payload.get("statementType"))
    add("Financial period", payload.get("financialPeriodLabel"))
    add("Period start", payload.get("periodStart"))
    add("Period end", payload.get("periodEnd"))
    add("Fiscal year", payload.get("fiscalYear"))
    add("Audited status", payload.get("auditedStatus"))
    add("Currency", payload.get("currency"))

    # Key figures (flatten to text)
    key_figures = payload.get("keyFigures") or {}
    for k, v in key_figures.items():
        add(k, v)

    # Detailed summary is the richest signal; include it last (strip markdown noise)
    summary = payload.get("detailed_summary")
    if summary:
        cleaned = _strip_markdown(summary)
        add("Summary", cleaned)

    return "\n".join(parts)


def _strip_markdown(text: str) -> str:
    """Remove lightweight markdown artifacts (bullets, emphasis, headers)."""
    cleaned = re.sub(r"(?m)^\s*[-*+]\s+", "", text)  # drop bullet markers
    cleaned = re.sub(r"[#*_`]+", "", cleaned)  # drop emphasis/headings/backticks
    return cleaned.strip()


def _slugify(value: str) -> str:
    """Generate a lowercase, dash-delimited token suitable for an ID."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _make_document_id(payload: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Build a stable, unique document ID so multiple statements don't overwrite
    each other inside the same namespace.
    """
    candidates = [
        payload.get("rcNumber"),
        payload.get("companyName"),
        payload.get("statementType"),
        payload.get("financialPeriodLabel"),
        payload.get("fiscalYear"),
        payload.get("periodEnd"),
    ]

    slug_parts = [_slugify(str(part)) for part in candidates if part]
    if slug_parts:
        return "-".join(slug_parts), False

    # Fall back to the source file name if present.
    source_file = payload.get("source_file")
    if source_file:
        return Path(source_file).stem, False

    # Absolute fallback to avoid collisions.
    return f"financial-statement-{uuid.uuid4().hex}", True


def _build_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Collect structured metadata to persist alongside the vector."""
    scalar_keys = [
        "companyName",
        "rcNumber",
        "reportTitle",
        "consolidationLevel",
        "statementType",
        "financialPeriodLabel",
        "periodStart",
        "periodEnd",
        "periodLengthMonths",
        "fiscalYear",
        "auditedStatus",
        "currency",
        "chunkPage",
    ]

    metadata: Dict[str, Any] = {}
    for key in scalar_keys:
        value = payload.get(key)
        if value not in (None, "", []):
            metadata[key] = value

    topics = payload.get("topics")
    if topics:
        metadata["topics"] = [str(item) for item in topics]

    # Preserve key figures as list of strings to satisfy Pinecone metadata rules.
    if payload.get("keyFigures"):
        metadata["keyFigures"] = [f"{k}:{v}" for k, v in payload["keyFigures"].items()]

    # Add cleaned summary in metadata for UI convenience (markdown removed)
    if payload.get("detailed_summary"):
        metadata["summary"] = _strip_markdown(payload["detailed_summary"])

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
    index_names = (
        set(index_list)
        if isinstance(index_list, list)
        else set(index_list.names())
    )
    if Config.INDEX_NAME not in index_names:
        raise RuntimeError(
            f"Pinecone index '{Config.INDEX_NAME}' not found. Create it first before running this script."
        )
    return pc.Index(Config.INDEX_NAME)


def upsert_financial_statements(batch_size: int = 50) -> None:
    documents = _load_documents()
    if not documents:
        logger.warning("No financial statement documents found to embed.")
        return

    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    index = _init_index()

    for batch in _chunk(documents, batch_size):
        texts = [_build_text_to_embed(doc) for doc in batch]
        embeddings = _get_embedding(texts, client)

        vectors = []
        batch_ids: List[str] = []
        uuid_ids: List[str] = []
        for doc, embedding, text in zip(batch, embeddings, texts):
            doc_id, is_uuid = _make_document_id(doc)
            metadata = _build_metadata(doc)
            metadata["content"] = text  # store the exact text sent to the embed model
            vectors.append(
                {
                    "id": doc_id,
                    "values": embedding,
                    "metadata": metadata,
                }
            )
            batch_ids.append(doc_id)
            if is_uuid:
                uuid_ids.append(doc_id)

        logger.info(
            "Upserting %s vectors into namespace '%s' with ids=%s",
            len(vectors),
            NAMESPACE,
            batch_ids,
        )
        if uuid_ids:
            logger.info("UUID-based ids in this batch: %s", uuid_ids)
        index.upsert(vectors=vectors, namespace=NAMESPACE)
        logger.info(
            "Upserted %s vectors into index '%s' (namespace='%s')",
            len(vectors),
            Config.INDEX_NAME,
            NAMESPACE,
        )

    # Wait a moment for Pinecone to update stats (eventual consistency)
    logger.info("Waiting for Pinecone to update index stats...")
    time.sleep(3)
    
    stats = index.describe_index_stats()
    logger.info("Full index stats: %s", stats)
    
    # Handle different possible response structures
    namespaces = {}
    if hasattr(stats, 'namespaces'):
        namespaces = stats.namespaces or {}
    elif isinstance(stats, dict):
        namespaces = stats.get("namespaces", {})
    
    ns_stats = namespaces.get(NAMESPACE, {})
    vector_count = ns_stats.get("vector_count") if isinstance(ns_stats, dict) else getattr(ns_stats, "vector_count", None)
    
    logger.info(
        "Completed upserting %s record(s). Namespace '%s' now has %s vector(s).",
        len(documents),
        NAMESPACE,
        vector_count or "unknown",
    )
    
    # Verification: try to fetch one of the vectors to confirm it's actually stored
    if documents:
        first_doc_id, _ = _make_document_id(documents[0])
        try:
            fetch_result = index.fetch(ids=[first_doc_id], namespace=NAMESPACE)
            if fetch_result and fetch_result.vectors:
                logger.info("✓ Verification successful: Retrieved vector '%s' from namespace '%s'", first_doc_id, NAMESPACE)
            else:
                logger.warning("⚠ Verification failed: Could not retrieve vector '%s' from namespace '%s'", first_doc_id, NAMESPACE)
        except Exception as e:
            logger.error("Error during verification: %s", e)


if __name__ == "__main__":
    upsert_financial_statements()

