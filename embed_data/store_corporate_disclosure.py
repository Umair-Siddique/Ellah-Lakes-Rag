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

DATA_DIR = ROOT_DIR / "processed" / "Corporate_Disclosures"
NAMESPACE = "corporate_disclosure"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMENSION = 3072

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _chunk(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield sequence chunks of a given size."""
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _load_documents() -> List[Dict[str, Any]]:
    """Load all processed corporate disclosure JSON payloads."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Processed directory not found: {DATA_DIR}")

    documents: List[Dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            documents.append(json.load(handle))
    logger.info("Loaded %s processed corporate disclosure file(s)", len(documents))
    return documents


def _format_people_events(events: List[Dict[str, Any]]) -> str:
    """Flatten key_people_events into a readable sentence."""
    formatted = []
    for person in events:
        name = person.get("name")
        role = person.get("role")
        event = person.get("event")
        effective_date = person.get("effective_date")
        bits = [bit for bit in [name, role, event, effective_date] if bit]
        if bits:
            formatted.append(", ".join(bits))
    return "; ".join(formatted)


def _build_text_to_embed(payload: Dict[str, Any]) -> str:
    """Compose a focused text blob for embedding."""
    parts: List[str] = []

    def add(label: str, value: Any) -> None:
        if value:
            parts.append(f"{label}: {value}")

    add("Title", payload.get("title"))
    add("Document type", payload.get("doc_type"))

    counterparties = payload.get("counterparties") or []
    add("Counterparties", "; ".join(counterparties))

    regulators = payload.get("regulatory_bodies") or []
    add("Regulatory bodies", "; ".join(regulators))

    people = payload.get("key_people_events") or []
    if people:
        add("Key people events", _format_people_events(people))

    # The detailed summary is the highest-signal chunk for retrieval.
    add("Detailed summary", payload.get("detailed_summary"))

    return "\n".join(parts)


def _build_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Collect structured metadata to persist alongside the vector."""
    scalar_keys = [
        "doc_id",
        "source_file",
        "title",
        "doc_type",
        "disclosure_date",
        "year",
        "company_name",
        "rc_number",
        "head_office_address",
        "website",
        "meeting_type",
        "meeting_date",
        "meeting_time",
        "meeting_location",
        "register_closure_start",
        "register_closure_end",
        "closed_period_start",
        "closed_period_end",
        "company_secretary",
        "signatory_name",
        "event_category",
        "transaction_type",
        "transaction_amount",
        "currency",
        "citation_file",
    ]
    list_keys = [
        "agenda_items",
        "directors_as_listed",
        "counterparties",
        "regulatory_bodies",
        "regulatory_references",
        "pages",
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
            if isinstance(value, list):
                metadata[key] = [str(item) for item in value]
            else:
                metadata[key] = [str(value)]

    # Convert key_people_events (list of dicts) to a list of readable strings for Pinecone.
    kpe = payload.get("key_people_events") or []
    if kpe:
        metadata["key_people_events"] = [
            ", ".join([bit for bit in [
                person.get("name"),
                person.get("role"),
                person.get("event"),
                person.get("effective_date"),
            ] if bit])
            for person in kpe
        ]

    # Keep the detailed summary handy for UI display without re-generating embeddings.
    if payload.get("detailed_summary"):
        metadata["summary"] = payload["detailed_summary"]

    return metadata


def _get_embedding(texts: List[str], client: OpenAI) -> List[List[float]]:
    """Batch embed a list of texts."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIMENSION,
    )
    return [item.embedding for item in response.data]


def _init_index() -> Any:
    """Connect to the Pinecone index and validate existence."""
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
            f"Pinecone index '{Config.INDEX_NAME}' not found. "
            "Create it first before running this script."
        )
    return pc.Index(Config.INDEX_NAME)


def upsert_corporate_disclosures(batch_size: int = 50) -> None:
    """Embed corporate disclosure JSON files and upsert into Pinecone."""
    documents = _load_documents()
    if not documents:
        logger.warning("No documents found to embed.")
        return

    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    index = _init_index()

    for batch in _chunk(documents, batch_size):
        texts = [_build_text_to_embed(doc) for doc in batch]
        embeddings = _get_embedding(texts, client)

        vectors = []
        for doc, embedding, text in zip(batch, embeddings, texts):
            doc_id = doc.get("doc_id") or Path(doc.get("source_file", "")).stem
            metadata = _build_metadata(doc)
            metadata["content"] = text  # store the exact text sent to the embed model
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

    logger.info("Completed upserting %s corporate disclosure record(s).", len(documents))


if __name__ == "__main__":
    upsert_corporate_disclosures()

