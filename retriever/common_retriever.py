from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator

from openai import OpenAI
import cohere

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config
from retriever.financial_statements_retriever import (
    search_financial_statements,
    format_match as format_financial_match,
)
from retriever.corporate_disclosure_retriever import (
    search_corporate_disclosures,
    format_match as format_corporate_match,
)
from retriever.director_dealing_retriever import (
    search_director_dealings,
    format_match as format_director_match,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """You are a query classification system for a financial RAG (Retrieval-Augmented Generation) system.

Your task is to analyze the user's query and determine which category it belongs to:

1. **financial_statements**: Queries about:
   - Financial statements (balance sheet, income statement, cash flow, statement of changes in equity)
   - Statement of financial position, comprehensive income
   - Financial ratios, metrics, KPIs
   - Quarterly, semi-annual, annual reports
   - Audited or unaudited financial data
   - Fiscal years, reporting periods
   - Revenue, profit, assets, liabilities, equity, cash flow
   - Financial performance, financial condition

2. **corporate_disclosure**: Queries about:
   - Board meetings, AGM (Annual General Meeting), EGM (Extraordinary General Meeting)
   - Corporate announcements, notices
   - Director appointments, resignations
   - Acquisitions, mergers, fundraising
   - Regulatory notices and compliance
   - Governance matters
   - Corporate actions and events
   - Meeting agendas, resolutions

3. **director_dealing**: Queries about:
   - Insider trading, insider dealing
   - Directors buying or selling shares
   - Share transactions by company insiders
   - Corporate insiders' stock transactions
   - Notification of dealings by persons discharging managerial responsibilities (PDMRs)
   - Insider ownership changes

Analyze the query and return a JSON object with:
- "category": one of ["financial_statements", "corporate_disclosure", "director_dealing"]
- "confidence": a number between 0 and 1 indicating your confidence
- "reasoning": a brief explanation of why you chose this category

If the query is ambiguous or could belong to multiple categories, choose the most relevant one and indicate lower confidence.

Return ONLY valid JSON, no other text."""


def _get_cohere_client():
    """
    Return a Cohere ClientV2 for v2 API endpoints.
    """
    if not Config.COHERE_API_KEY:
        return None
    try:
        # Use Cohere Python SDK v5+ (v2 API)
        return cohere.ClientV2(api_key=Config.COHERE_API_KEY)
    except AttributeError:
        logger.error("Cohere ClientV2 not available. Please upgrade: pip install --upgrade cohere")
        return None


def _classify_query(query: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Classify the query to determine which retriever to use.
    
    Args:
        query: User's natural language query
        model: LLM model to use for classification
    
    Returns:
        Dictionary with 'category', 'confidence', and 'reasoning'
    """
    if not Config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing; set it in your .env file.")
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    messages = [
        {"role": "system", "content": CLASSIFICATION_PROMPT},
        {"role": "user", "content": f"Query: {query}\n\nClassify this query:"},
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        
        # Extract JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
        
        result = json.loads(content)
        
        # Validate result
        if not isinstance(result, dict):
            raise ValueError("Classification result is not a dictionary")
        
        category = result.get("category")
        if category not in ["financial_statements", "corporate_disclosure", "director_dealing"]:
            logger.warning("Invalid category '%s', defaulting to financial_statements", category)
            category = "financial_statements"
            result["category"] = category
        
        logger.info(
            "Query classified as '%s' with confidence %.2f: %s",
            category,
            result.get("confidence", 0.0),
            result.get("reasoning", "No reasoning provided"),
        )
        
        return result
        
    except Exception as exc:
        logger.error("Classification failed: %s. Defaulting to financial_statements.", exc)
        return {
            "category": "financial_statements",
            "confidence": 0.5,
            "reasoning": f"Classification failed: {exc}",
        }


def _rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_n: Optional[int] = None,
    model: str = "rerank-english-v3.0",
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Cohere's v2 rerank API.
    
    Args:
        query: The user's query
        documents: List of document matches from vector search
        top_n: Number of top documents to return after reranking (None = all)
        model: Cohere rerank model to use (e.g., "rerank-english-v3.0", "rerank-multilingual-v3.0")
    
    Returns:
        Reranked list of documents with updated scores
    """
    if not Config.COHERE_API_KEY:
        logger.warning("COHERE_API_KEY is missing; skipping reranking.")
        return documents
    
    if not documents:
        return documents
    
    co = _get_cohere_client()
    if co is None:
        logger.warning("Cohere client unavailable; skipping reranking.")
        return documents

    try:
        # Prepare documents for reranking - use summary or text content
        docs_for_rerank = []
        for doc in documents:
            metadata = doc.get("metadata", {}) or {}
            # Try to get the most relevant text content
            text = metadata.get("summary") or metadata.get("text") or metadata.get("content") or ""
            if text:
                docs_for_rerank.append(text)
            else:
                # Fallback to all metadata as text
                docs_for_rerank.append(str(metadata))
        
        if not docs_for_rerank:
            logger.warning("No text content found in documents for reranking")
            return documents
        
        # Limit to 1000 documents as per Cohere recommendation
        if len(docs_for_rerank) > 1000:
            logger.warning("Too many documents (%d), limiting to 1000 for reranking", len(docs_for_rerank))
            docs_for_rerank = docs_for_rerank[:1000]
            documents = documents[:1000]
        
        # Call Cohere v2 rerank API
        logger.info("Reranking %d documents with Cohere model '%s'...", len(docs_for_rerank), model)
        
        # Determine top_n - if not specified, return all documents
        if top_n is None:
            top_n_val = len(docs_for_rerank)
        else:
            top_n_val = min(top_n, len(docs_for_rerank))
        
        # Make the v2 API call according to official docs
        rerank_response = co.rerank(
            model=model,
            query=query,
            documents=docs_for_rerank,
            top_n=top_n_val,
        )
        
        # Reorder documents based on rerank results
        reranked_docs = []
        
        # Access results from Cohere v2 API response
        if hasattr(rerank_response, 'results'):
            results = rerank_response.results
        elif isinstance(rerank_response, dict):
            results = rerank_response.get('results', [])
        else:
            logger.warning("Unexpected rerank response format: %s", type(rerank_response))
            return documents
        
        if not results:
            logger.warning("No results returned from reranking")
            return documents
        
        for result in results:
            # Get index and relevance score from v2 API response
            if hasattr(result, 'index'):
                original_idx = result.index
                relevance = result.relevance_score
            elif isinstance(result, dict):
                original_idx = result.get('index')
                relevance = result.get('relevance_score', 0)
            else:
                logger.warning("Unexpected result format: %s", type(result))
                continue
            
            if original_idx is not None and 0 <= original_idx < len(documents):
                original_doc = documents[original_idx]
                if original_doc is None:
                    logger.warning("Document at index %d is None, skipping", original_idx)
                    continue
                
                # Handle both dict and Pinecone ScoredVector objects
                if isinstance(original_doc, dict):
                    reranked_doc = original_doc.copy()
                elif hasattr(original_doc, 'to_dict'):
                    # Pinecone ScoredVector has to_dict() method
                    reranked_doc = original_doc.to_dict()
                elif hasattr(original_doc, '__dict__'):
                    # Fallback: convert object attributes to dict
                    reranked_doc = {
                        'id': getattr(original_doc, 'id', None),
                        'score': getattr(original_doc, 'score', 0),
                        'metadata': getattr(original_doc, 'metadata', {}),
                        'values': getattr(original_doc, 'values', []),
                    }
                else:
                    logger.warning("Document at index %d is not convertible: %s", original_idx, type(original_doc))
                    continue
                
                # Update the score with rerank score
                reranked_doc["score"] = relevance
                reranked_doc["rerank_score"] = relevance
                reranked_docs.append(reranked_doc)
        
        if reranked_docs:
            logger.info(
                "Reranking complete. Returned %d/%d documents. Top score: %.4f",
                len(reranked_docs),
                len(documents),
                reranked_docs[0]["score"]
            )
            return reranked_docs
        else:
            logger.warning("No documents were reranked successfully, returning originals")
            return documents
        
    except Exception as exc:
        logger.error("Reranking failed: %s. Returning original documents.", exc, exc_info=True)
        return documents


def _generate_streaming_response(
    query: str,
    documents: List[Dict[str, Any]],
    category: str,
    model: str = "command-a-03-2025",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Generator[str, None, None]:
    """
    Generate a streaming response using Cohere's chat API with retrieved documents.
    
    Args:
        query: The user's query
        documents: List of retrieved and reranked documents
        category: The category of documents (for context)
        model: Cohere chat model to use (default: command-a-03-2025)
        chat_history: Optional list of previous messages for context (last 5 messages)
    
    Yields:
        Chunks of the generated response
    """
    if not Config.COHERE_API_KEY:
        yield "Error: COHERE_API_KEY is missing; cannot generate response."
        return
    
    if not documents:
        yield "No relevant documents were found to answer your query."
        return
    
    co = _get_cohere_client()
    if co is None:
        yield "Error: Cohere client could not be initialized."
        return

    try:
        # Log how many documents we're using for generation
        logger.info("Generating response using %d documents", len(documents))
        
        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get("metadata", {}) or {}
            text = metadata.get("summary") or metadata.get("text") or metadata.get("content") or ""
            
            # Add document metadata for context
            doc_info = f"Document {i}:"
            if category == "financial_statements":
                company = metadata.get("companyName", "Unknown")
                report = metadata.get("reportTitle", "Unknown")
                period = metadata.get("financialPeriodLabel", "Unknown")
                doc_info += f"\nCompany: {company}\nReport: {report}\nPeriod: {period}"
            elif category == "corporate_disclosure":
                title = metadata.get("title", "Unknown")
                doc_type = metadata.get("doc_type", "Unknown")
                date = metadata.get("disclosure_date", "Unknown")
                doc_info += f"\nTitle: {title}\nType: {doc_type}\nDate: {date}"
            elif category == "director_dealing":
                insider = metadata.get("insider_name", "Unknown")
                transaction = metadata.get("transaction_nature", "Unknown")
                date = metadata.get("announcement_date", "Unknown")
                doc_info += f"\nInsider: {insider}\nTransaction: {transaction}\nDate: {date}"
            
            if text:
                context_parts.append(f"{doc_info}\nContent: {text}\n")
        
        context = "\n---\n".join(context_parts)
        
        # Create the preamble with context
        preamble = f"""You are a helpful financial assistant analyzing {category.replace('_', ' ')} documents.
Use the following retrieved documents to answer the user's question accurately and comprehensively.
If the documents don't contain enough information to answer the question, say so.

Retrieved Documents:
{context}
"""
        
        # Prepare chat history for Cohere v2 format
        cohere_messages = []
        if chat_history:
            for msg in chat_history:
                role = msg.get("role")
                content = msg.get("content", "")
                # Convert to Cohere v2 format
                if role == "user":
                    cohere_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    cohere_messages.append({"role": "assistant", "content": content})
        
        # Add the current user query with context prepended
        # Since v2 API chat_stream doesn't accept 'system' parameter,
        # we include the preamble/context in the user message
        query_with_context = f"{preamble}\n\nUser Question: {query}"
        cohere_messages.append({"role": "user", "content": query_with_context})
        
        # Stream the response using v2 API
        logger.info("Generating streaming response with Cohere v2 (with %d history messages)...", len(cohere_messages) - 1)
        
        # For v2 API (cohere>=5.0.0), use chat_stream() method
        stream = co.chat_stream(
            model=model,
            messages=cohere_messages,
            temperature=0.3,
        )
        
        for event in stream:
            # V2 API streaming events - handle different event types
            if hasattr(event, 'type'):
                event_type = event.type
                
                # Content delta event contains the actual text chunks
                if event_type == "content-delta":
                    if hasattr(event, 'delta') and hasattr(event.delta, 'message'):
                        if hasattr(event.delta.message, 'content') and hasattr(event.delta.message.content, 'text'):
                            yield event.delta.message.content.text
                
                # Message end event signals completion
                elif event_type == "message-end":
                    logger.info("Response generation complete")
                    break
            
            # Fallback for simpler event structures
            elif hasattr(event, 'delta'):
                if hasattr(event.delta, 'text'):
                    yield event.delta.text
            elif hasattr(event, 'text'):
                yield event.text
        
    except Exception as exc:
        logger.error("Response generation failed: %s", exc)
        yield f"\n\nError generating response: {exc}"


def search(
    query: str,
    top_k: int = 5,
    classification_model: str = "gpt-4o-mini",
    filter_model: Optional[str] = None,
    force_category: Optional[str] = None,
    enable_rerank: bool = True,
    rerank_top_n: Optional[int] = None,
    rerank_model: str = "rerank-english-v3.0",
) -> Dict[str, Any]:
    """
    Intelligent search that routes queries to the appropriate specialized retriever.
    
    Args:
        query: Natural language search query
        top_k: Number of results to return from initial search
        classification_model: LLM model for query classification (default: gpt-4o-mini)
        filter_model: Optional LLM model for filter generation in retrievers
        force_category: Optional category to skip classification and force a specific retriever
                       ("financial_statements", "corporate_disclosure", or "director_dealing")
        enable_rerank: Whether to rerank results using Cohere (default: False)
        rerank_top_n: Number of top documents after reranking (None = all)
        rerank_model: Cohere rerank model to use (default: rerank-english-v3.0)
    
    Returns:
        Dictionary with:
        - category: The detected or forced category
        - classification: Classification details (if not forced)
        - results: List of search results (reranked if enabled)
        - formatted_results: List of formatted result strings
    """
    # Determine category
    if force_category:
        if force_category not in ["financial_statements", "corporate_disclosure", "director_dealing"]:
            raise ValueError(f"Invalid force_category: {force_category}")
        category = force_category
        classification = {
            "category": category,
            "confidence": 1.0,
            "reasoning": "Category was manually forced",
        }
        logger.info("Using forced category: %s", category)
    else:
        classification = _classify_query(query, model=classification_model)
        category = classification["category"]
    
    # Route to appropriate retriever
    results = []
    formatter = None
    
    if category == "financial_statements":
        logger.info("Routing to financial_statements_retriever")
        results = search_financial_statements(query, top_k=top_k, filter_model=filter_model)
        formatter = format_financial_match
    
    elif category == "corporate_disclosure":
        logger.info("Routing to corporate_disclosure_retriever")
        results = search_corporate_disclosures(query, top_k=top_k, filter_model=filter_model)
        formatter = format_corporate_match
    
    elif category == "director_dealing":
        logger.info("Routing to director_dealing_retriever")
        results = search_director_dealings(query, top_k=top_k, filter_model=filter_model)
        formatter = format_director_match
    
    # Rerank if enabled
    if enable_rerank and results:
        logger.info("Reranking enabled: will rerank %d documents to top %s", len(results), rerank_top_n or "all")
        results = _rerank_documents(query, results, top_n=rerank_top_n, model=rerank_model)
        logger.info("After reranking: %d documents returned", len(results))
    
    # Format results
    formatted_results = []
    if results and formatter:
        formatted_results = [formatter(match) for match in results]
    
    return {
        "category": category,
        "classification": classification,
        "results": results,
        "formatted_results": formatted_results,
    }


def search_and_generate(
    query: str,
    top_k: int = 5,
    classification_model: str = "gpt-4o-mini",
    filter_model: Optional[str] = None,
    force_category: Optional[str] = None,
    rerank_top_n: Optional[int] = None,
    rerank_model: str = "rerank-english-v3.0",
    chat_model: str = "command-a-03-2025",
    chat_history: Optional[List[Dict[str, str]]] = None,
    enable_rerank: bool = True,
) -> Dict[str, Any]:
    """
    Complete RAG pipeline: search, rerank, and generate streaming response.
    
    Args:
        query: Natural language search query
        top_k: Number of results to retrieve initially
        classification_model: LLM model for query classification
        filter_model: Optional LLM model for filter generation
        force_category: Optional category override
        rerank_top_n: Number of top documents after reranking (None = use top_k)
        rerank_model: Cohere rerank model (default: rerank-english-v3.0)
        chat_model: Cohere chat model for response generation (default: command-a-03-2025)
        chat_history: Optional list of previous messages for context (last 5 messages)
    
    Returns:
        Dictionary with:
        - category: Document category
        - classification: Classification details
        - results: Reranked search results
        - formatted_results: Formatted result strings
        - response_stream: Generator for streaming response
    """
    # Search (enable rerank only if Cohere is configured and enabled)
    do_rerank = enable_rerank and bool(Config.COHERE_API_KEY)
    search_result = search(
        query=query,
        top_k=top_k,
        classification_model=classification_model,
        filter_model=filter_model,
        force_category=force_category,
        enable_rerank=do_rerank,
        rerank_top_n=rerank_top_n,
        rerank_model=rerank_model,
    )
    
    # Generate streaming response
    response_stream = _generate_streaming_response(
        query=query,
        documents=search_result["results"],
        category=search_result["category"],
        model=chat_model,
        chat_history=chat_history,
    )
    
    return {
        "category": search_result["category"],
        "classification": search_result["classification"],
        "results": search_result["results"],
        "formatted_results": search_result["formatted_results"],
        "response_stream": response_stream,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Financial RAG: Search → Rerank → Generate")
    print("=" * 70)
    print("The system will automatically:")
    print("1. Classify your query to the right document type")
    print("2. Retrieve relevant documents from Pinecone")
    print("3. Rerank them using Cohere for better relevance")
    print("4. Generate a streaming response using the top documents")
    print("\nExamples:")
    print("  - As at 31 January 2013, what were Ellah Lakes PLC's total assets?")
    print("  - What board meetings were scheduled in 2023?")
    print("  - Which directors sold shares last year?")
    print("\nPress Ctrl+C or enter empty line to exit")
    print("=" * 70)
    
    while True:
        try:
            user_q = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not user_q:
            print("Exiting...")
            break
        
        # Always use full pipeline: search, rerank, and generate
        result = search_and_generate(user_q, top_k=10, rerank_top_n=5)
        
        print(f"\n[Category: {result['category']}]")
        print(f"[Confidence: {result['classification']['confidence']:.2f}]")
        
        if not result['results']:
            print("\nNo relevant documents found.")
            continue
        
        print(f"[Using {len(result['results'])} reranked documents]\n")
        print("=" * 70)
        print("ANSWER:")
        print("=" * 70)
        
        # Stream the response
        for chunk in result['response_stream']:
            print(chunk, end='', flush=True)
        
        print("\n" + "=" * 70)
        
        # Optionally show source documents
        show_sources = input("\nShow source documents? (y/n) [n]: ").strip().lower()
        if show_sources == 'y':
            print("\n" + "=" * 70)
            print("SOURCE DOCUMENTS:")
            print("=" * 70)
            for i, formatted in enumerate(result['formatted_results'], 1):
                print(f"\n[Document {i}]")
                print(formatted)

