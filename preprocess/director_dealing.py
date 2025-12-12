"""
Pipeline for extracting structured insider dealing metadata + summaries
from Ellah Lakes director-dealing PDFs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config

try:
    from pypdf import PdfReader  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    raise ImportError("pypdf is required. Install with `pip install pypdf`.") from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import pytesseract
    from pdf2image import convert_from_path
    # Configure Tesseract path for Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning(
        "OCR libraries not available. Install with: pip install pytesseract pdf2image pillow"
    )

DATASET_DIR = ROOT_DIR / "dataset" / "Director_Dealings"
PROCESSED_DIR = ROOT_DIR / "processed"
OUTPUT_DIR = PROCESSED_DIR / "Director_Dealings"
MAX_FILES: int | None = None  # None means process all PDFs
SYSTEM_PROMPT = """You are an expert extraction engine for insider share-dealing (director dealing) regulatory notifications.

You read highly structured "Notification of Share Dealing by Insiders" documents for Ellah Lakes Plc and your ONLY task is to produce ACCURATE, NORMALISED METADATA in a fixed JSON schema for a Retrieval-Augmented Generation (RAG) system.

These notifications follow a table-like template and contain:
- details of the insider (name, position/status),
- issuer details (Ellah Lakes Plc, LEI),
- transaction details (instrument, ISIN, nature of the transaction, prices and volumes, aggregated figures),
- dates and locations (transaction dates, place of transaction, announcement location and date),
- signature (company secretary or law firm).

Your rules for THOROUGH and ACCURATE extraction:

1. READ THE ENTIRE DOCUMENT carefully - scan every section for metadata.
2. Use ONLY the fields defined in the schema provided by the user.
3. If a field is missing or not clearly stated, set it to null - do NOT guess.
4. BE AGGRESSIVE in extraction - if you see a value, extract it.

Number Normalization (for easy filtering and querying):
5. Remove ALL commas from numbers: "10,000,000" → 10000000
6. Remove ALL currency symbols: "₦4.25", "N 4.25" → 4.25
7. Keep decimals intact: "4.27" stays as 4.27
8. Convert to pure numeric values for all amounts and volumes

Date Normalization (CRITICAL for filtering):
9. ALL dates MUST be ISO format: YYYY-MM-DD
   - "21 September 2023" → "2023-09-21"
   - "10 August 2022" → "2022-08-10"
   - "21/09/2023" → "2023-09-21"
10. Extract year as 4-digit number (2023, 2022)
11. Extract month as 2-digit string ("01", "09", "12")

Array Alignment:
12. If multiple lines of prices/volumes exist, maintain alignment:
    - price_per_share[i] corresponds to volume[i]
    - Each may correspond to transaction_dates[i] if per-line dates exist

Entity Classification:
13. insider_type:
    - "Corporate" when the insider is a company (e.g., "CBO CAPITAL PARTNERS LTD", "TELLURIA LIMITED")
    - "Individual" when the insider is a natural person (e.g., "Mr. Chuka Mordi", "Mrs. Patricia Ellah")
14. is_corporate_insider: boolean true if "Corporate", false if "Individual"

Mandatory Summaries:
15. short_summary (MANDATORY): 1 clear sentence with who, what action (buy/sell), volume, price, and date
    Example: "CBO Capital Partners Limited sold 15,000,000 shares at ₦4.27 per share on 21 September 2023."

16. Output MUST be a single valid JSON object matching the schema exactly. No comments, no extra text."""

USER_PROMPT_TEMPLATE = """You are a high-accuracy extraction engine for Ellah Lakes Plc insider share-dealing notifications.

Your goal is to produce:
1. Clean, normalised metadata in JSON format
2. A short one-sentence summary (MANDATORY)

CRITICAL REQUIREMENTS:
- Extract EVERY piece of information from the document
- Normalize ALL dates to YYYY-MM-DD format for easy filtering
- Normalize ALL numbers (remove commas and currency symbols)
- short_summary is MANDATORY and must be a complete sentence

Return ONLY a JSON object with the exact fields shown below.
Do NOT add any extra fields or comments.

METADATA SCHEMA (OUTPUT STRICT JSON):

{
  "company_name": null,
  "rc_number": null,
  "legal_entity_identifier": null,
  "insider_name": null,
  "position_status": null,
  "insider_type": null,
  "is_corporate_insider": null,
  "transaction_nature": null,
  "instrument_type": null,
  "isin": null,
  "price_per_share": [],
  "volume": [],
  "currency": null,
  "aggregated_volume": null,
  "aggregated_price": null,
  "transaction_dates": [],
  "announcement_date": null,
  "announcement_location": null,
  "place_of_transaction": null,
  "signatory_name": null,
  "signatory_designation": null,
  "year": null,
  "month": null,
  "document_type": "insider_dealing_notification",
  "pdf_source": null,
  "short_summary": null
}

Field-specific instructions (BE THOROUGH - extract every detail you find):

Company Information:
- company_name: exact name of the issuer (e.g., "Ellah Lakes Plc")
- rc_number: numeric RC number only as string (e.g., "RC 34296" → "34296", "RC 299748" → "299748")
- legal_entity_identifier: numeric LEI (e.g., "7219037")

Insider Information:
- insider_name: full name of the director/insider from section 1 (e.g., "CBO CAPITAL PARTNERS LIMITED", "Mr. Chuka Mordi")
- position_status: exact text from "Position/status" field (e.g., "Substantial Shareholder", "CEO/ Executive Director", "Managing Director")
- insider_type: "Corporate" if the insider is a company/organization, "Individual" if a person
- is_corporate_insider: boolean - true if insider_type is "Corporate", false if "Individual"

Transaction Details:
- transaction_nature: exact text from "Nature of the transaction" (e.g., "Sale of Shares", "Purchase of Shares", "Disposal of Shares")
- instrument_type: type of security (e.g., "shares", "equity", "ordinary shares")
- isin: security identification code exactly as shown (e.g., "NGELLAHLAKE8")

Price and Volume (normalize all numbers):
- price_per_share: array of numeric prices (remove commas, currency symbols)
  Examples: ["4.25", "4.30"] or [4.25, 4.30]
  From "₦4.25" → 4.25, "N 4.30" → 4.30
- volume: array of numeric volumes corresponding to each price (remove commas)
  Examples: [10000000, 5000000]
  From "10,000,000" → 10000000
- currency: standard 3-letter code (e.g., "NGN" for Naira, "USD" for US Dollars)
- aggregated_volume: total volume as number (remove commas)
  From "15,000,000" → 15000000
- aggregated_price: weighted average price or total price as number
  From "₦4.27" → 4.27

Dates and Locations (normalize all dates to YYYY-MM-DD):
- transaction_dates: array of ISO dates for each trade
  Examples: ["2023-09-21", "2023-09-22"]
  From "21 September 2023" → "2023-09-21"
  From "21/09/2023" → "2023-09-21"
- announcement_date: ISO date from "Location and Date of Announcement"
  Examples: "21 September 2023" → "2023-09-21", "10 August 2022" → "2022-08-10"
- announcement_location: city name (e.g., "Lagos", "Benin City", "Port Harcourt")
- place_of_transaction: city or exchange name (e.g., "Lagos", "Nigerian Exchange Limited")

Signatory:
- signatory_name: full name from "Name of Signatory" (e.g., "Mrs. Oluwabusayo Iretioluwa Awoyo", "Michael E. Ellah, Esq.")
- signatory_designation: role (e.g., "COMPANY SECRETARY", "Head of Finance", "Managing Director")

Metadata:
- year: 4-digit year from announcement_date (e.g., 2023, 2022)
- month: 2-digit month from announcement_date (e.g., "01", "09", "12")
- document_type: always "insider_dealing_notification"
- pdf_source: file path from the input

Summary (MANDATORY):
- short_summary: 1 concise sentence capturing the core event
  Example: "CBO Capital Partners Limited sold 15,000,000 shares of Ellah Lakes Plc at an average price of ₦4.27 on 21 September 2023."

CRITICAL EXTRACTION REMINDERS:
1. Read the ENTIRE document text carefully - metadata can appear anywhere
2. ALL dates → YYYY-MM-DD format (e.g., "21 September 2023" → "2023-09-21")
3. ALL numbers → remove commas and currency symbols (e.g., "10,000,000" → 10000000, "₦4.25" → 4.25)
4. Extract insider_name, position_status, transaction_nature exactly as written
5. Extract ALL prices and volumes into arrays
6. short_summary is MANDATORY - write a clear 1-sentence summary
7. If a field is not in the document, use null - do not guess

INPUT TEXT (PDF extraction starts below):

<<<PDF_TEXT>>>
"""

DEFAULT_METADATA_TEMPLATE: Dict[str, Any] = {
    "company_name": None,
    "rc_number": None,
    "legal_entity_identifier": None,
    "insider_name": None,
    "position_status": None,
    "insider_type": None,
    "is_corporate_insider": None,
    "transaction_nature": None,
    "instrument_type": None,
    "isin": None,
    "price_per_share": [],
    "volume": [],
    "currency": None,
    "aggregated_volume": None,
    "aggregated_price": None,
    "transaction_dates": [],
    "announcement_date": None,
    "announcement_location": None,
    "place_of_transaction": None,
    "signatory_name": None,
    "signatory_designation": None,
    "year": None,
    "month": None,
    "document_type": "insider_dealing_notification",
    "pdf_source": None,
    "short_summary": None,
}

DETAILED_SUMMARY_PROMPT = """You are an expert analyst creating a comprehensive, detailed summary of an insider share-dealing notification document.

Read the document text below and produce a thorough summary that includes EVERY significant piece of information.

Your summary MUST include:

1. **Insider Identification:**
   - Full name of the insider/director
   - Position/status (e.g., Substantial Shareholder, Managing Director)
   - Whether they are a corporate entity or individual person

2. **Transaction Details:**
   - Nature of the transaction (Sale of Shares, Purchase of Shares, etc.)
   - Instrument type and ISIN code
   - ALL prices mentioned (individual prices and aggregated/weighted average price)
   - ALL volumes mentioned (individual trades and total aggregated volume)
   - Currency used (e.g., Nigerian Naira - NGN)
   - Preserve exact figures: "10,000,000 shares at ₦4.25 per share"

3. **Dates and Locations:**
   - Transaction date(s) - list all dates if multiple transactions
   - Announcement date and location
   - Place of transaction (e.g., Nigerian Exchange Limited, Lagos)

4. **Issuer Information:**
   - Company name (Ellah Lakes Plc)
   - RC number if mentioned
   - Legal Entity Identifier (LEI) if present

5. **Signatory and Authorization:**
   - Name of the person who signed the notification
   - Their designation/role (e.g., Company Secretary, Head of Finance)

6. **Regulatory Context:**
   - Any references to regulatory bodies (NGX, SEC, etc.)
   - Any rule references or compliance notes

Format your summary in clear paragraphs organized by these sections. Include ALL numbers, dates, and names exactly as they appear in the document.

EXCLUDE only generic contact information (phone numbers, emails, office addresses).

DOCUMENT TEXT:

<<<PDF_TEXT>>>

Provide the detailed summary below:
"""


def get_pdf_files(limit: int | None = MAX_FILES) -> List[Path]:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    # Get all PDFs and remove duplicates based on lowercase filename
    all_pdfs = list(DATASET_DIR.glob("*.pdf")) + list(DATASET_DIR.glob("*.PDF"))
    
    # Remove duplicates: keep only one version if same name exists in different cases
    seen = set()
    pdfs = []
    for pdf in sorted(all_pdfs, key=lambda p: p.name.lower()):
        name_lower = pdf.name.lower()
        if name_lower not in seen:
            seen.add(name_lower)
            pdfs.append(pdf)
    
    if limit is None:
        selected = pdfs
    else:
        selected = pdfs[:limit]
    logger.info("Found %s unique director-dealing PDFs; processing %s", len(pdfs), len(selected))
    return selected


def extract_text_with_ocr(pdf_path: Path) -> str:
    """Extract text from PDF using OCR for image-based pages."""
    if not OCR_AVAILABLE:
        logger.error("OCR libraries not available. Cannot extract text from image-based PDF.")
        return ""
    
    try:
        logger.info("Using OCR to extract text from %s", pdf_path.name)
        images = convert_from_path(str(pdf_path))
        text_chunks: List[str] = []
        
        for idx, image in enumerate(images, start=1):
            logger.info("OCR processing page %d/%d of %s", idx, len(images), pdf_path.name)
            page_text = pytesseract.image_to_string(image)
            page_text = page_text.replace("\x00", " ").strip()
            if page_text:
                text_chunks.append(f"[PAGE {idx}]\n{page_text}")
        
        extracted_text = "\n\n".join(text_chunks).strip()
        logger.info("OCR extracted text from %d pages of %s", len(text_chunks), pdf_path.name)
        return extracted_text
    except Exception as e:
        logger.error("OCR extraction failed for %s: %s", pdf_path.name, str(e))
        return ""


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full text from a PDF, using OCR if needed.
    
    First tries standard text extraction. If insufficient text is found,
    falls back to OCR extraction for image-based PDFs.
    """
    text_chunks: List[str] = []
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    pages_with_text = 0
    
    # Try standard text extraction first
    for idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = page_text.replace("\x00", " ").strip()
        if page_text:
            text_chunks.append(f"[PAGE {idx}]\n{page_text}")
            pages_with_text += 1
    
    extracted_text = "\n\n".join(text_chunks).strip()
    
    # Check if we got enough text (at least 30% of pages have text and total text > 100 chars)
    text_coverage = (pages_with_text / total_pages) if total_pages > 0 else 0
    sufficient_text = text_coverage >= 0.3 and len(extracted_text) > 100
    
    if sufficient_text:
        logger.info(
            "Extracted text from %d/%d pages of %s (%.1f%%) - Standard extraction",
            pages_with_text,
            total_pages,
            pdf_path.name,
            text_coverage * 100
        )
        return extracted_text
    else:
        # Insufficient text, try OCR
        logger.info(
            "Insufficient text from standard extraction (%d/%d pages, %.1f%%). Attempting OCR...",
            pages_with_text,
            total_pages,
            text_coverage * 100
        )
        ocr_text = extract_text_with_ocr(pdf_path)
        
        # Use whichever extraction yielded more text
        if len(ocr_text) > len(extracted_text):
            return ocr_text
        elif extracted_text:
            return extracted_text
        else:
            return ocr_text


def build_user_prompt(doc_text: str, source_file: str) -> str:
    payload = f"PDF source: {source_file}\n\n{doc_text}"
    return USER_PROMPT_TEMPLATE.replace("<<<PDF_TEXT>>>", payload)


def call_model_with_retry(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    max_retries: int = 3,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Empty response from OpenAI model.")
            return content.strip()
        except Exception as exc:  # pragma: no cover
            last_error = exc
            sleep_for = 2 ** (attempt - 1)
            logger.warning(
                "Model call failed (%s/%s): %s. Retrying in %ss...",
                attempt,
                max_retries,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    raise RuntimeError("Failed to obtain response from OpenAI") from last_error


def parse_json_output(raw_content: str) -> Dict[str, Any]:
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw_content[start : end + 1])
        raise


def normalize_metadata(metadata: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    normalized = deepcopy(DEFAULT_METADATA_TEMPLATE)
    normalized["pdf_source"] = source_file
    for key, value in metadata.items():
        if key in normalized and value is not None:
            normalized[key] = value
    if not isinstance(normalized["price_per_share"], list):
        normalized["price_per_share"] = [normalized["price_per_share"]]
    if not isinstance(normalized["volume"], list):
        normalized["volume"] = [normalized["volume"]]
    if not isinstance(normalized["transaction_dates"], list):
        normalized["transaction_dates"] = [normalized["transaction_dates"]]
    return normalized


def generate_detailed_summary(
    text: str,
    client: OpenAI,
    model: str,
) -> str:
    """Call OpenAI to generate a comprehensive detailed summary of the document."""
    user_prompt = DETAILED_SUMMARY_PROMPT.replace("<<<PDF_TEXT>>>", text)
    messages = [
        {"role": "system", "content": "You are an expert analyst creating comprehensive document summaries."},
        {"role": "user", "content": user_prompt},
    ]
    detailed_summary = call_model_with_retry(client, messages, model=model)
    return detailed_summary


def generate_metadata(
    text: str,
    source_file: str,
    client: OpenAI,
    model: str,
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(text, source_file)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw_content = call_model_with_retry(client, messages, model=model)
    metadata = parse_json_output(raw_content)
    normalized = normalize_metadata(metadata, source_file)
    
    # Generate detailed summary separately
    logger.info("Generating detailed summary for %s", source_file)
    detailed_summary = generate_detailed_summary(text, client, model)
    normalized["detailed_summary"] = detailed_summary
    
    return normalized


def save_result(result: Dict[str, Any], stem: str) -> None:
    """Persist individual JSON file for each document."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_file = OUTPUT_DIR / f"{stem}.json"
    with per_file.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def run(limit: int | None = MAX_FILES, model: str | None = None) -> List[Dict[str, Any]]:
    if not Config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    skipped_count = 0
    pdfs = get_pdf_files(limit=limit)
    total = len(pdfs)
    existing = {p.stem for p in OUTPUT_DIR.glob("*.json")}
    
    for idx, pdf_path in enumerate(pdfs, start=1):
        if pdf_path.stem in existing:
            logger.info("Skipping %s - already processed", pdf_path.name)
            continue

        logger.info("Processing %s (%d/%d)", pdf_path.name, idx, total)
        text = extract_text_from_pdf(pdf_path)
        if not text:
            logger.warning(
                "Skipping %s - no extractable text (likely image-based PDF requiring OCR)",
                pdf_path.name,
            )
            skipped_count += 1
            continue
        
        try:
            metadata = generate_metadata(
                text=text,
                source_file=str(pdf_path),
                client=client,
                model=resolved_model,
            )
            save_result(metadata, pdf_path.stem)
            results.append(metadata)
            logger.info("✓ Successfully processed %s", pdf_path.name)
        except Exception as e:
            logger.error("Failed to process %s: %s", pdf_path.name, str(e))
            skipped_count += 1
            continue

    logger.info(
        "Processing complete: %s documents processed, %s skipped. Files saved to %s",
        len(results),
        skipped_count,
        OUTPUT_DIR,
    )
    return results


if __name__ == "__main__":
    run()

