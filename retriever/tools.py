"""
LangChain tools for Financial RAG system.

This module defines tools that the LLM can call to search different document namespaces:
- financial_statements: Financial reports, balance sheets, income statements
- corporate_disclosure: Board meetings, AGM, corporate announcements
- director_dealing: Insider trading, director share transactions

Each tool wraps the existing retriever functionality without changing it.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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


def _convert_pinecone_to_dict(obj):
    """Convert Pinecone ScoredVector objects to dictionaries."""
    if isinstance(obj, dict):
        return obj
    elif hasattr(obj, 'to_dict'):
        # Pinecone ScoredVector has to_dict() method
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        # Fallback: convert object attributes to dict
        return {
            'id': getattr(obj, 'id', None),
            'score': getattr(obj, 'score', 0),
            'metadata': getattr(obj, 'metadata', {}),
            'values': getattr(obj, 'values', []),
        }
    else:
        return str(obj)


@tool
def search_financial_statements_tool(
    query: str,
    top_k: int = 10,
) -> str:
    """Search financial statements and reports.
    
    Use this tool when users ask about:
    - Financial statements (balance sheet, income statement, cash flow, statement of changes in equity)
    - Statement of financial position, comprehensive income
    - Financial ratios, metrics, KPIs
    - Quarterly, semi-annual, annual reports
    - Audited or unaudited financial data
    - Fiscal years, reporting periods
    - Revenue, profit, assets, liabilities, equity, cash flow
    - Financial performance, financial condition
    
    Args:
        query: The user's question about financial statements
        top_k: Number of results to retrieve (default: 10)
    
    Returns:
        JSON string containing search results with metadata
    """
    logger.info("Tool called: search_financial_statements_tool with query='%s', top_k=%d", query, top_k)
    
    try:
        results = search_financial_statements(query, top_k=top_k)
        
        # Convert Pinecone objects to dictionaries
        results_dicts = [_convert_pinecone_to_dict(r) for r in results]
        
        # Return results as JSON for the LLM to process
        return json.dumps({
            "category": "financial_statements",
            "num_results": len(results_dicts),
            "results": results_dicts,
        }, default=str)
    
    except Exception as e:
        logger.error("Error in search_financial_statements_tool: %s", e)
        return json.dumps({
            "category": "financial_statements",
            "error": str(e),
            "num_results": 0,
            "results": [],
        })


@tool
def search_corporate_disclosures_tool(
    query: str,
    top_k: int = 10,
) -> str:
    """Search corporate disclosures and announcements.
    
    Use this tool when users ask about:
    - Board meetings, AGM (Annual General Meeting), EGM (Extraordinary General Meeting)
    - Corporate announcements, notices
    - Director appointments, resignations
    - Acquisitions, mergers, fundraising
    - Regulatory notices and compliance
    - Governance matters
    - Corporate actions and events
    - Meeting agendas, resolutions
    - Closed periods, trading windows
    
    Args:
        query: The user's question about corporate disclosures
        top_k: Number of results to retrieve (default: 10)
    
    Returns:
        JSON string containing search results with metadata
    """
    logger.info("Tool called: search_corporate_disclosures_tool with query='%s', top_k=%d", query, top_k)
    
    try:
        results = search_corporate_disclosures(query, top_k=top_k)
        
        # Convert Pinecone objects to dictionaries
        results_dicts = [_convert_pinecone_to_dict(r) for r in results]
        
        # Return results as JSON for the LLM to process
        return json.dumps({
            "category": "corporate_disclosure",
            "num_results": len(results_dicts),
            "results": results_dicts,
        }, default=str)
    
    except Exception as e:
        logger.error("Error in search_corporate_disclosures_tool: %s", e)
        return json.dumps({
            "category": "corporate_disclosure",
            "error": str(e),
            "num_results": 0,
            "results": [],
        })


@tool
def search_director_dealings_tool(
    query: str,
    top_k: int = 10,
) -> str:
    """Search director dealings and insider transactions.
    
    Use this tool when users ask about:
    - Insider trading, insider dealing
    - Directors buying or selling shares
    - Share transactions by company insiders
    - Corporate insiders' stock transactions
    - Notification of dealings by persons discharging managerial responsibilities (PDMRs)
    - Insider ownership changes
    - Director share purchases or sales
    
    Args:
        query: The user's question about director dealings
        top_k: Number of results to retrieve (default: 10)
    
    Returns:
        JSON string containing search results with metadata
    """
    logger.info("Tool called: search_director_dealings_tool with query='%s', top_k=%d", query, top_k)
    
    try:
        results = search_director_dealings(query, top_k=top_k)
        
        # Convert Pinecone objects to dictionaries
        results_dicts = [_convert_pinecone_to_dict(r) for r in results]
        
        # Return results as JSON for the LLM to process
        return json.dumps({
            "category": "director_dealing",
            "num_results": len(results_dicts),
            "results": results_dicts,
        }, default=str)
    
    except Exception as e:
        logger.error("Error in search_director_dealings_tool: %s", e)
        return json.dumps({
            "category": "director_dealing",
            "error": str(e),
            "num_results": 0,
            "results": [],
        })


# Export all tools as a list for easy registration
ALL_TOOLS = [
    search_financial_statements_tool,
    search_corporate_disclosures_tool,
    search_director_dealings_tool,
]


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get OpenAI-compatible tool definitions for all retriever tools.
    
    Returns:
        List of tool definitions in OpenAI format
    """
    tools = []
    
    for tool_func in ALL_TOOLS:
        # Get the tool's schema
        tool_schema = {
            "type": "function",
            "function": {
                "name": tool_func.name,
                "description": tool_func.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        # Add parameters from the tool's args_schema if available
        if hasattr(tool_func, 'args_schema') and tool_func.args_schema:
            schema = tool_func.args_schema.model_json_schema()
            tool_schema["function"]["parameters"]["properties"] = schema.get("properties", {})
            tool_schema["function"]["parameters"]["required"] = schema.get("required", [])
        
        tools.append(tool_schema)
    
    return tools


def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """
    Execute a tool by name with given arguments.
    
    Args:
        tool_name: Name of the tool to execute
        tool_args: Dictionary of arguments to pass to the tool
    
    Returns:
        Tool execution result as string
    """
    tool_map = {tool.name: tool for tool in ALL_TOOLS}
    
    if tool_name not in tool_map:
        logger.error("Unknown tool: %s", tool_name)
        return json.dumps({
            "error": f"Unknown tool: {tool_name}",
            "available_tools": list(tool_map.keys())
        })
    
    tool_func = tool_map[tool_name]
    
    try:
        result = tool_func.invoke(tool_args)
        return result
    except Exception as e:
        logger.error("Error executing tool %s: %s", tool_name, e)
        return json.dumps({
            "error": f"Tool execution failed: {str(e)}"
        })


if __name__ == "__main__":
    """Test the tools"""
    print("=" * 70)
    print("Financial RAG Tools Test")
    print("=" * 70)
    
    # Test 1: Financial statements tool
    print("\n1. Testing search_financial_statements_tool...")
    result = search_financial_statements_tool.invoke({
        "query": "What were the total assets in 2013?",
        "top_k": 3
    })
    data = json.loads(result)
    print(f"   Category: {data['category']}")
    print(f"   Results found: {data['num_results']}")
    
    # Test 2: Corporate disclosures tool
    print("\n2. Testing search_corporate_disclosures_tool...")
    result = search_corporate_disclosures_tool.invoke({
        "query": "Board meetings in 2023",
        "top_k": 3
    })
    data = json.loads(result)
    print(f"   Category: {data['category']}")
    print(f"   Results found: {data['num_results']}")
    
    # Test 3: Director dealings tool
    print("\n3. Testing search_director_dealings_tool...")
    result = search_director_dealings_tool.invoke({
        "query": "Which directors sold shares?",
        "top_k": 3
    })
    data = json.loads(result)
    print(f"   Category: {data['category']}")
    print(f"   Results found: {data['num_results']}")
    
    # Test 4: Get tool definitions
    print("\n4. Testing get_tool_definitions...")
    tools = get_tool_definitions()
    print(f"   Total tools: {len(tools)}")
    for tool in tools:
        print(f"   - {tool['function']['name']}")
    
    print("\n" + "=" * 70)
    print("✓ All tool tests completed!")
    print("=" * 70)
