"""
Process plain-text financial statement files into structured JSON plus a
dense-friendly summary for downstream RAG / similarity search.

Input directory:
  dataset/Financial_Statement_Text/*.txt

Output directory:
  processed/Financials_Statements/<stem>_page_1.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config  # noqa: E402  # import after sys.path mutation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_DIR = ROOT_DIR / "dataset" / "Financials_Statements"
TEXT_DATASET_DIR = ROOT_DIR / "dataset" / "Financial_Statement_Text"
PROCESSED_DIR = ROOT_DIR / "processed"
OUTPUT_DIR = PROCESSED_DIR / "Financials_Statements"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Base schema requested by the user.
BASE_RECORD: Dict[str, Any] = {
    "companyName": "",
    "rcNumber": None,
    "reportTitle": None,
    "consolidationLevel": None,
    "statementType": None,
    "financialPeriodLabel": None,
    "periodStart": None,
    "periodEnd": None,
    "periodLengthMonths": None,
    "fiscalYear": None,
    "auditedStatus": None,
    "currency": "NGN",
    "chunkPage": 1,
    "topics": [],
    "keyFigures": {
        "totalAssets": None,
        "totalLiabilities": None,
        "totalEquity": None,
        "revenue": None,
        "costOfSales": None,
        "grossProfit": None,
        "operatingProfitLoss": None,
        "netProfit": None,
        "eps": None,
        "cashAndCashEquivalents": None,
        "cashAtBank": None,
        "biologicalAssets": None,
        "propertyPlantEquipment": None,
        "borrowings": None,
        "retainedEarnings": None,
        "shareCapital": None,
        "sharePremium": None,
        "revaluationSurplus": None,
    },
    "detailed_summary": ".",
}

SYSTEM_PROMPT = """You are a meticulous financial-statement extraction assistant.

Goal: Given the raw text of a financial statement page, produce a JSON object that matches EXACTLY the schema below. Use null where a field is missing or unclear. Do NOT invent values.

SCHEMA:
{
  "companyName": "String or null",
  "rcNumber": "String or null",
  "reportTitle": "String or null",
  "consolidationLevel": "String or null",
  "statementType": "String or null",
  "financialPeriodLabel": "String or null",
  "periodStart": "YYYY-MM-DD or null",
  "periodEnd": "YYYY-MM-DD or null",
  "periodLengthMonths": Number or null,
  "fiscalYear": Number or null,
  "auditedStatus": "audited" | "unaudited" | null,
  "currency": "NGN",
  "keyFigures": {
    "totalAssets": Number or null,
    "totalLiabilities": Number or null,
    "totalEquity": Number or null,
    "revenue": Number or null,
    "costOfSales": Number or null,
    "grossProfit": Number or null,
    "operatingProfitLoss": Number or null,
    "netProfit": Number or null,
    "eps": Number or null,
    "cashAndCashEquivalents": Number or null,
    "cashAtBank": Number or null,
    "biologicalAssets": Number or null,
    "propertyPlantEquipment": Number or null,
    "borrowings": Number or null,
    "retainedEarnings": Number or null,
    "shareCapital": Number or null,
    "sharePremium": Number or null,
    "revaluationSurplus": Number or null
  }
}

Rules:
- Preserve numbers as they appear (strip commas, handle parentheses as negatives).
- If a figure is shown twice, pick the clearest main figure.
- If the page has no numeric figures, leave them null.
- Output ONLY valid JSON matching the schema. No extra fields or text.
"""

USER_PROMPT_TEMPLATE = """Extract the financial metadata JSON from the text below.

Return ONLY the JSON object. If a field is missing, use null.

TEXT:

{page_text}
"""

SUMMARY_PROMPT = """You are an expert summarizer for financial statements.

Given the extracted page text, write a concise but complete summary capturing ALL financial figures, periods, statement types, and important line items (assets, liabilities, equity, revenue, expenses, profit/loss, cash flow notes). Preserve numbers and wording where possible.

The summary should be dense and information-rich so it is useful for similarity search. Avoid generic filler. Mention dates/periods, currency, and any notable totals.

TEXT:

{page_text}
"""


def call_model_with_retry(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    max_retries: int = 3,
    expect_json: bool = True,
) -> str:
    """Invoke the OpenAI model with basic exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0,
            }
            if expect_json:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Received empty response content from model.")
            return content.strip()
        except Exception as exc:  # pragma: no cover - best-effort logging
            last_error = exc
            sleep_for = 2 ** (attempt - 1)
            logger.warning(
                "Model call failed on attempt %s/%s: %s. Retrying in %ss...",
                attempt,
                max_retries,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    raise RuntimeError("Exhausted OpenAI retries") from last_error


def generate_summary(text: str, client: OpenAI, model: str) -> str:
    """Generate a dense, information-rich summary for similarity search."""
    summary_prompt = SUMMARY_PROMPT.replace("{page_text}", text)
    messages = [
        {"role": "system", "content": "You are an expert summarizer of financial statements."},
        {"role": "user", "content": summary_prompt},
    ]
    return call_model_with_retry(client, messages, model, expect_json=False)


# Simple numeric parsing to normalize comma/parenthesis formatted values.
NUMERIC_CLEAN_RE = re.compile(r"[^\d\.\-]")


def parse_numeric(value: str) -> float | None:
    """Convert numeric-looking strings to float; returns None if parsing fails."""
    if not value:
        return None
    cleaned = value.strip()
    is_negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.replace("(", "").replace(")", "")
    cleaned = NUMERIC_CLEAN_RE.sub("", cleaned.replace(",", ""))
    try:
        number = float(cleaned)
        return -number if is_negative else number
    except ValueError:
        return None


KEY_FIGURE_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "totalAssets": ("total assets",),
    "totalLiabilities": ("total liabilities",),
    "totalEquity": ("total equity", "shareholders' funds", "shareholders funds"),
    "revenue": ("revenue", "turnover"),
    "costOfSales": ("cost of sales", "cost of goods sold"),
    "grossProfit": ("gross profit",),
    "operatingProfitLoss": ("operating profit", "operating loss"),
    "netProfit": ("profit for the year", "profit after tax", "loss after tax"),
    "eps": ("earnings per share", "eps"),
    "cashAndCashEquivalents": ("cash and cash equivalents",),
    "cashAtBank": ("cash at bank",),
    "biologicalAssets": ("biological assets",),
    "propertyPlantEquipment": ("property, plant and equipment", "ppe"),
    "borrowings": ("borrowings", "loans and borrowings"),
    "retainedEarnings": ("retained earnings",),
    "shareCapital": ("share capital", "issued share capital"),
    "sharePremium": ("share premium",),
    "revaluationSurplus": ("revaluation surplus",),
}


def extract_key_figures(text: str, kv_pairs: List[Tuple[str, str]] | None = None) -> Dict[str, Any]:
    """Best-effort numeric extraction for common financial statement line items."""
    found: Dict[str, Any] = {k: None for k in BASE_RECORD["keyFigures"].keys()}
    working_text = text.lower()

    # Regex scan through text content
    for key, phrases in KEY_FIGURE_PATTERNS.items():
        for phrase in phrases:
            match = re.search(rf"{re.escape(phrase)}\s*[:\-]?\s*([\(\)\d,\.]+)", working_text)
            if match:
                candidate = parse_numeric(match.group(1))
                if candidate is not None:
                    found[key] = candidate
                    break

    # Allow structured kv_pairs (label, value) to override if provided
    if kv_pairs:
        for label, value in kv_pairs:
            label_l = label.lower()
            for key, phrases in KEY_FIGURE_PATTERNS.items():
                if any(p in label_l for p in phrases):
                    parsed = parse_numeric(value)
                    if parsed is not None:
                        found[key] = parsed

    return found


def derive_company_name(text: str, source_file: str) -> str:
    """Try to pull a company name; fall back to filename stem."""
    match = re.search(r"([A-Z][A-Za-z\&\s]+?)(?:\s+Plc|\s+PLC)", text)
    if match:
        return match.group(1).strip() + " Plc"
    return Path(source_file).stem.replace("_", " ")


def extract_metadata_with_llm(
    text: str, client: OpenAI, model: str
) -> Dict[str, Any]:
    """Call OpenAI to extract metadata from the page text."""
    # Log the extracted text being sent to the LLM
    logger.info("Extracted Text passed to LLM:\n%s\n--- End of Text ---", text)

    user_prompt = USER_PROMPT_TEMPLATE.replace("{page_text}", text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        content = call_model_with_retry(client, messages, model)
        # Log LLM response for debugging as requested
        logger.info("LLM Response for text segment:\n%s", content)
        return json.loads(content)
    except Exception as e:
        logger.error("LLM extraction failed: %s", e)
        return {}


def build_page_metadata(
    text: str,
    kv_pairs: List[Tuple[str, str]] | None,
    page_number: int,
    source_file: str,
    client: OpenAI | None = None,
    model: str | None = None,
) -> Dict[str, Any]:
    """Populate the schema for a single page."""
    record = deepcopy(BASE_RECORD)
    record["chunkPage"] = page_number
    record["companyName"] = derive_company_name(text, source_file)
    record["detailed_summary"] = text.strip() if text.strip() else "."
    
    # If client is provided, use LLM for better extraction
    if client and model and len(text.strip()) > 10:
        llm_data = extract_metadata_with_llm(text, client, model)
        # Merge LLM data into record
        for key in [
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
        ]:
            if llm_data.get(key) is not None:
                record[key] = llm_data[key]

        if llm_data.get("keyFigures"):
            for k, v in llm_data["keyFigures"].items():
                if v is not None:
                    record["keyFigures"][k] = v

        # Generate dense-friendly summary
        summary = generate_summary(text, client=client, model=model)
        record["detailed_summary"] = summary
    else:
        # Fallback to regex and use raw text as summary
        record["keyFigures"] = extract_key_figures(text, kv_pairs or [])
        record["detailed_summary"] = text.strip() if text.strip() else "."

    return record


def save_page_record(record: Dict[str, Any], stem: str, page_number: int) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{stem}_page_{page_number}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    return out_path


def get_text_files() -> List[Path]:
    """Return all .txt files from the text dataset directory."""
    if not TEXT_DATASET_DIR.exists():
        return []
    
    txt_files = list(TEXT_DATASET_DIR.glob("*.txt"))
    return sorted(txt_files)


def process_text_file(txt_path: Path, client: OpenAI | None, model: str) -> None:
    """Process a single text file and create JSON output."""
    logger.info("Processing text file: %s", txt_path.name)
    
    # Read the text file
    try:
        with txt_path.open('r', encoding='utf-8') as f:
            text = f.read().strip()
    except Exception as e:
        logger.error("Failed to read %s: %s", txt_path.name, e)
        return
    
    if not text:
        logger.warning("Empty text file: %s", txt_path.name)
        return
    
    # Build metadata using LLM
    record = build_page_metadata(
        text=text,
        kv_pairs=[],
        page_number=1,
        source_file=str(txt_path),
        client=client,
        model=model,
    )
    
    # Save as JSON
    output_path = save_page_record(record, txt_path.stem, page_number=1)
    logger.info("Saved: %s", output_path)


def run(limit: int | None = None, use_text_files: bool = False) -> None:
    """Entry point for processing text files."""
    if not Config.OPENAI_API_KEY:
         logger.warning("OPENAI_API_KEY not found. LLM extraction will be skipped.")
         client = None
    else:
         client = OpenAI(api_key=Config.OPENAI_API_KEY)
         
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process text files
    text_files = get_text_files()
    if not text_files:
        logger.warning("No .txt files found in %s", TEXT_DATASET_DIR)
        return
    
    # Build set of already-processed stems to avoid duplicates
    existing = {p.stem.rsplit("_page_", 1)[0] for p in OUTPUT_DIR.glob("*.json")}
    
    logger.info("Found %d text file(s) to process", len(text_files))
    processed_count = 0
    skipped_count = 0
    for txt_path in text_files:
        stem = txt_path.stem
        if stem in existing:
            logger.info("Skipping %s (already processed)", txt_path.name)
            skipped_count += 1
            continue
        process_text_file(txt_path, client=client, model=DEFAULT_MODEL)
        processed_count += 1
    
    logger.info(
        "Text file processing complete. %d processed, %d skipped. Outputs in %s",
        processed_count,
        skipped_count,
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    logger.info("Processing text files from %s", TEXT_DATASET_DIR)
    run(use_text_files=True)

