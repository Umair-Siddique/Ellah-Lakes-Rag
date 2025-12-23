"""
Pipeline for extracting text from Ellah Lakes corporate disclosure PDFs,
calling OpenAI to build structured metadata + summaries, and persisting
the outputs for downstream RAG ingestion.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

try:
    from pypdf import PdfReader  # type: ignore[import]
    from pypdf.errors import DependencyError as PdfDependencyError  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - ensures clearer error messaging
    raise ImportError(
        "pypdf is required for PDF text extraction. Run `pip install pypdf`."
    ) from exc

try:
    import pytesseract
    from pdf2image import convert_from_path
    # Configure Tesseract path for Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if not OCR_AVAILABLE:
    logger.warning(
        "OCR libraries not available. Install with: pip install pytesseract pdf2image pillow"
    )

DATASET_DIR = ROOT_DIR / "dataset" / "Corporate_Disclosures"
PROCESSED_DIR = ROOT_DIR / "processed"
OUTPUT_DIR = PROCESSED_DIR / "Corporate_Disclosures"
MAX_FILES = None  # None means process all PDFs
SYSTEM_PROMPT = """You are an expert corporate-disclosure analyst AI.

You read corporate disclosure documents for Ellah Lakes Plc (a Nigerian agribusiness company) and produce ACCURATE, STRUCTURED METADATA for a Retrieval-Augmented Generation (RAG) system.

The documents include, for example:
- Board Meeting Notices
- AGM / EGM Notices
- Director / Executive Appointments and Resignations
- Corporate Actions (fundraise, cross deal, acquisition, offtake agreements)
- Regulatory / Compliance Notices
- Press releases about governance or capital markets updates

Your tasks:
1. Correctly understand the document's purpose and classify its type.
2. Extract ONLY the metadata fields defined in the schema provided by the user.
3. Use `null` for any field that is not present or cannot be determined.
4. Do NOT hallucinate or infer missing values.
   - If an amount, date, or name is not explicitly stated, leave the field as null.
5. Preserve all numbers exactly as written in the document (including decimals).
6. Extract ALL relevant people, dates, organizations, and regulators, but only into the specified fields.
7. BE THOROUGH: Carefully scan the entire document text to find all metadata fields.
8. Normalize all dates to YYYY-MM-DD format for easy filtering.
9. Extract ALL director names, people, counterparties, and regulatory bodies mentioned.
10. Output MUST be STRICT, VALID JSON that matches the schema. No comments, no extra text."""

USER_PROMPT_TEMPLATE = """Below is the text extracted from a corporate disclosure PDF related to Ellah Lakes Plc.

Please extract metadata according to the schema below. 
If any field does not exist in the document, set it to null.
Do NOT add extra fields.

METADATA SCHEMA (OUTPUT STRICT JSON):

{{
  "doc_id": null,
  "source_file": null,
  "title": null,
  "doc_type": null,
  "disclosure_date": null,
  "year": null,
  "company_name": "Ellah Lakes Plc",
  "rc_number": null,
  "head_office_address": null,
  "website": null,
  "meeting_type": null,
  "meeting_date": null,
  "meeting_time": null,
  "meeting_location": null,
  "agenda_items": [],
  "register_closure_start": null,
  "register_closure_end": null,
  "closed_period_start": null,
  "closed_period_end": null,
  "directors_as_listed": [],
  "key_people_events": [],
  "company_secretary": null,
  "signatory_name": null,
  "event_category": null,
  "transaction_type": null,
  "transaction_amount": null,
  "currency": null,
  "counterparties": [],
  "regulatory_bodies": [],
  "regulatory_references": [],
  "pages": [],
  "citation_file": null
}}

Field notes and normalisation rules (BE THOROUGH - extract everything you find):

- "title": short natural-language title summarising the disclosure (max ~20 words).
  Example: "Notice of Board Meeting to Consider Financing Options"

- "doc_type": choose the most specific type from these examples:
  - "board_meeting_notice" - notice of upcoming board meeting
  - "agm_notice" - annual general meeting notice
  - "egm_notice" - extraordinary general meeting notice
  - "director_resignation" - announcement of director resignation
  - "director_appointment" - announcement of director/executive appointment
  - "fundraise_update" - capital raising, equity placement
  - "corporate_action" - corporate actions, register closure
  - "regulatory_notice" - compliance, closed period notices
  - "acquisition" - acquisition or disposal announcements
  - "financial_results" - financial results announcements

- "disclosure_date": normalize to YYYY-MM-DD (e.g., "Dated this 13th day of November 2019" → "2019-11-13", "Lagos, 8 April 2024" → "2024-04-08")
- "year": 4-digit year from disclosure_date (e.g., 2019, 2024)
- "rc_number": numeric part only as string (e.g., "RC 34296" → "34296")

Meeting fields (if this is a meeting notice):
- "meeting_type": "board_meeting", "agm", "egm", "court_meeting"
- "meeting_date": YYYY-MM-DD format
- "meeting_time": 24-hour HH:MM format (e.g., "10:00 a.m." → "10:00", "3:00 p.m." → "15:00")
- "meeting_location": full address or venue name
- "agenda_items": array of strings, each agenda item

Closure periods:
- "register_closure_start": YYYY-MM-DD (first day register closes)
- "register_closure_end": YYYY-MM-DD (last day register closes)
- "closed_period_start": YYYY-MM-DD (start of closed period)
- "closed_period_end": YYYY-MM-DD (end of closed period)

People and roles:
- "directors_as_listed": ALL director names from the header/signature section, in order as listed
  Example: ["Mr. Chuka Mordi", "Gen. Zamari Lekwot (rtd)", "Mrs. Patricia Ireju Ellah"]
- "key_people_events": for EACH person mentioned in the document body, create object:
  {
    "name": "Full Name",
    "event": "appointment" | "resignation" | "continuing" | "announcement_subject",
    "role": "Managing Director" | "Chief Financial Officer" | "Director" | etc.,
    "effective_date": "YYYY-MM-DD" or null
  }
- "company_secretary": name or law firm (e.g., "OAKE Legal", "Michael E. Ellah, Esq.")
- "signatory_name": person who signed the document

Transaction/Event details:
- "event_category": high-level category - "fundraise", "free_float_update", "acquisition", "offtake_agreement", "financing", "governance", "meeting", "appointment", "resignation"
- "transaction_type": specific type - "private_placement", "cross_deal", "rights_issue", "board_resolution"
- "transaction_amount": numeric only (e.g., "₦1,500,000.00" → 1500000.00, "NGN 2.5 million" → 2500000)
- "currency": standard codes (NGN, USD, GBP, EUR)

Parties and regulators (extract ALL mentions):
- "counterparties": ALL companies, investors, shareholders, creditors, banks mentioned
  Example: ["CBO Capital Partners Limited", "Zenith Bank Plc", "Chief J.W Ellah & Sons Co"]
- "regulatory_bodies": ALL regulators mentioned
  Example: ["Nigerian Exchange Limited", "Securities and Exchange Commission", "The Nigerian Stock Exchange"]
- "regulatory_references": rules/circulars cited
  Example: ["Rule 17.15", "NSE Rulebook 2015"]

Metadata:
- "pages": extract page numbers from "[PAGE 1]" markers as integers [1, 2, 3]
- "citation_file": short filename for UI citation
- "head_office_address": if mentioned
- "website": if mentioned

DOCUMENT TEXT:

\"\"\"

{{PDF_TEXT_HERE}}

\"\"\"

IMPORTANT EXTRACTION GUIDELINES:
- Read the ENTIRE document text carefully - metadata can appear anywhere in the document.
- Be AGGRESSIVE in extraction - if you see a field value, extract it.
- ALL DATES must be normalized to YYYY-MM-DD format for easy filtering and querying.
  Examples: "8 April 2024" → "2024-04-08", "Dated this 13th day of November 2019" → "2019-11-13"
- ALL TIMES must be normalized to 24-hour HH:MM format.
  Examples: "10:00 a.m." → "10:00", "3:00 p.m." → "15:00"
- ALL AMOUNTS must be extracted as pure numeric values without currency symbols or formatting.
  Examples: "₦1,500,000.00" → 1500000.00, "NGN 2.5 million" → 2500000
- Currency codes must be standardized (NGN, USD, GBP, etc.).
- RC numbers must contain only the numeric portion as a string.
  Examples: "RC 34296" → "34296"
- directors_as_listed: Extract ALL director names in the order they appear in the document header.
- key_people_events: For each person mentioned in the body, create an entry with their name, event type, role, and effective date.
- counterparties: Extract ALL companies, investors, shareholders, banks, creditors mentioned.
- regulatory_bodies: Extract ALL regulators (Nigerian Exchange Limited, SEC, NSE, etc.).
- agenda_items: List all agenda items if this is a meeting notice.
- pages: Extract page numbers from markers like "[PAGE 1]" as integers [1, 2, 3].
- If a value is unclear or ambiguous, set the field to null rather than guessing.
- Only return valid JSON that exactly matches the schema.
"""

DETAILED_SUMMARY_PROMPT = """You are an expert analyst tasked with creating a comprehensive, detailed summary of a corporate disclosure document.

Read the document text provided below and produce a thorough summary that includes EVERY significant piece of information from the document.

Your summary should:
1. Be written in a clean, well-organized format with clear sections or paragraphs
2. Include ALL important details such as:
   - Purpose and nature of the disclosure
   - All key dates, amounts, and numerical data
   - Names of all people mentioned and their roles
   - All companies, organizations, and regulatory bodies referenced
   - Specific transactions, events, or decisions described
   - All agenda items, resolutions, or action points
   - Regulatory requirements or compliance matters
   - Any deadlines, closure periods, or important timeframes
3. EXCLUDE only generic contact information (phone numbers, email addresses, standard office addresses)
4. Preserve the exact wording of important terms, amounts, and dates
5. Organize the information logically by topic or chronology
6. Be thorough but concise - do not add interpretations or opinions

DOCUMENT TEXT:

\"\"\"

{{PDF_TEXT_HERE}}

\"\"\"

Provide the detailed summary below:
"""


def get_pdf_files(limit: int | None = MAX_FILES) -> List[Path]:
    """Return the first N PDF paths sorted alphabetically (case-insensitive)."""
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
        limited = pdfs
    else:
        limited = pdfs[:limit]
    logger.info("Found %s unique PDF(s); processing %s", len(pdfs), len(limited))
    return limited


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
    try:
        reader = PdfReader(str(pdf_path))
    except PdfDependencyError as exc:
        logger.error(
            "Cannot read %s: missing cryptography dependency (install cryptography>=3.1). Error: %s",
            pdf_path.name,
            exc,
        )
        return ""
    except Exception as exc:
        logger.error("Failed to read %s: %s", pdf_path.name, exc)
        return ""
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


def build_user_prompt(doc_text: str, source_file: str, doc_id: str) -> str:
    """Inject the PDF text and provenance into the user prompt template."""
    payload = (
        f"Source file: {source_file}\n"
        f"Doc ID: {doc_id}\n\n"
        f"{doc_text}"
    )
    return USER_PROMPT_TEMPLATE.replace("{{PDF_TEXT_HERE}}", payload)


def call_model_with_retry(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    max_retries: int = 3,
) -> str:
    """Invoke the OpenAI model with basic exponential backoff."""
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


def parse_json_output(raw_content: str) -> Dict[str, Any]:
    """Parse the JSON returned by the model, attempting auto-repair if needed."""
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw_content[start : end + 1]
            return json.loads(candidate)
        raise


def generate_detailed_summary(
    text: str,
    client: OpenAI,
    model: str,
) -> str:
    """Call OpenAI to generate a comprehensive detailed summary of the document."""
    user_prompt = DETAILED_SUMMARY_PROMPT.replace("{{PDF_TEXT_HERE}}", text)
    messages = [
        {"role": "system", "content": "You are an expert analyst creating comprehensive document summaries."},
        {"role": "user", "content": user_prompt},
    ]
    detailed_summary = call_model_with_retry(client, messages, model=model)
    return detailed_summary


def generate_metadata_for_pdf(
    text: str,
    source_file: str,
    doc_id: str,
    client: OpenAI,
    model: str,
) -> Dict[str, Any]:
    """Call OpenAI to build metadata + summary JSON for a single document."""
    user_prompt = build_user_prompt(text, source_file, doc_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw_content = call_model_with_retry(client, messages, model=model)
    metadata = parse_json_output(raw_content)

    # Enforce provenance fields even if the model omits them.
    metadata.setdefault("doc_id", doc_id)
    metadata.setdefault("source_file", source_file)
    
    # Generate detailed summary separately
    logger.info("Generating detailed summary for %s", doc_id)
    detailed_summary = generate_detailed_summary(text, client, model)
    metadata["detailed_summary"] = detailed_summary
    
    return metadata


def save_result(result: Dict[str, Any], doc_id: str) -> None:
    """Persist individual JSON file for each document."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_file = OUTPUT_DIR / f"{doc_id}.json"
    with per_file.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def run(limit: int | None = MAX_FILES, model: str | None = None) -> List[Dict[str, Any]]:
    """Entry point for preprocessing the first N PDFs."""
    if not Config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing; set it in your .env file.")

    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    skipped_count = 0
    pdfs = get_pdf_files(limit=limit)
    total = len(pdfs)

    # Skip PDFs already processed (by doc_id/filename stem).
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
            metadata = generate_metadata_for_pdf(
                text=text,
                source_file=str(pdf_path),
                doc_id=pdf_path.stem,
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
        OUTPUT_DIR
    )
    return results


if __name__ == "__main__":
    run()

