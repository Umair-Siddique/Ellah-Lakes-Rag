"""
LangChain tools for CSV agents.

This module defines tools that the LLM can call to query different CSV datasets:
- monetary_aggregates: Quarterly monetary data (1960-1980)
- gdp_by_sector: Annual GDP data by sector (1960+)

Each tool wraps the existing CSV retriever functionality without changing it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from csv_agent.csv_retriever_1960_2023 import query_csv as query_monetary
from csv_agent.csv_retriver_from_1960 import query_csv as query_gdp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@tool
def query_monetary_aggregates_tool(query: str) -> str:
    """Query monetary aggregates and credit data from 1960-1980 (quarterly).
    
    Use this tool when users ask about:
    - Money supply (M1, M2, broad money, narrow money)
    - Quarters (Q1, Q2, Q3, Q4) or quarterly data
    - Reserve money, credit aggregates
    - Net Foreign Assets (NFA), Net Claims on Government (NCG)
    - Private Sector Deposit Demand (PSDD)
    - Deposit Money Banks (DMB)
    - Currency in Circulation (CIC)
    - Monetary policy metrics
    - Banking sector liquidity
    - Credit to government or private sector
    
    Data coverage: 1960-1980, quarterly frequency
    
    Args:
        query: The user's question about monetary data
    
    Returns:
        Answer based on the monetary aggregates CSV data
    """
    logger.info("Tool called: query_monetary_aggregates_tool with query='%s'", query)
    
    try:
        answer = query_monetary(query)
        return answer
    
    except Exception as e:
        logger.error("Error in query_monetary_aggregates_tool: %s", e)
        return f"Error querying monetary aggregates: {str(e)}"


@tool
def query_gdp_by_sector_tool(query: str) -> str:
    """Query GDP data by economic sector from 1960 onwards (annual).
    
    Use this tool when users ask about:
    - GDP (Gross Domestic Product) or economic output
    - Economic sectors or industries
    - Agriculture (crop production, livestock, forestry, fishing)
    - Industry (manufacturing, mining, petroleum, gas, quarrying)
    - Construction and building
    - Services sector
    - Wholesale and retail trade
    - Transport and communication/telecom
    - Finance, insurance, and real estate
    - Business services
    - Hotels and restaurants
    - Utilities and public services
    - Sector contributions to GDP
    - Economic growth by sector
    - Annual economic data
    
    Data coverage: 1960 onwards, annual frequency
    
    Args:
        query: The user's question about GDP or sector data
    
    Returns:
        Answer based on the GDP by sector CSV data
    """
    logger.info("Tool called: query_gdp_by_sector_tool with query='%s'", query)
    
    try:
        answer = query_gdp(query)
        return answer
    
    except Exception as e:
        logger.error("Error in query_gdp_by_sector_tool: %s", e)
        return f"Error querying GDP by sector: {str(e)}"


# Export all tools as a list for easy registration
ALL_CSV_TOOLS = [
    query_monetary_aggregates_tool,
    query_gdp_by_sector_tool,
]


def get_csv_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get OpenAI-compatible tool definitions for all CSV tools.
    
    Returns:
        List of tool definitions in OpenAI format
    """
    tools = []
    
    for tool_func in ALL_CSV_TOOLS:
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


def execute_csv_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """
    Execute a CSV tool by name with given arguments.
    
    Args:
        tool_name: Name of the tool to execute
        tool_args: Dictionary of arguments to pass to the tool
    
    Returns:
        Tool execution result as string
    """
    tool_map = {tool.name: tool for tool in ALL_CSV_TOOLS}
    
    if tool_name not in tool_map:
        logger.error("Unknown CSV tool: %s", tool_name)
        return f"Error: Unknown tool '{tool_name}'. Available tools: {list(tool_map.keys())}"
    
    tool_func = tool_map[tool_name]
    
    try:
        result = tool_func.invoke(tool_args)
        return result
    except Exception as e:
        logger.error("Error executing CSV tool %s: %s", tool_name, e)
        return f"Error executing tool: {str(e)}"


if __name__ == "__main__":
    """Test the CSV tools"""
    print("=" * 70)
    print("CSV Tools Test")
    print("=" * 70)
    
    # Test 1: Monetary aggregates tool
    print("\n1. Testing query_monetary_aggregates_tool...")
    result = query_monetary_aggregates_tool.invoke({
        "query": "What was the M2 money supply in Q2 1970?"
    })
    print(f"   Answer: {result[:100]}...")
    
    # Test 2: GDP by sector tool
    print("\n2. Testing query_gdp_by_sector_tool...")
    result = query_gdp_by_sector_tool.invoke({
        "query": "What was the GDP contribution from agriculture in 1980?"
    })
    print(f"   Answer: {result[:100]}...")
    
    # Test 3: Get tool definitions
    print("\n3. Testing get_csv_tool_definitions...")
    tools = get_csv_tool_definitions()
    print(f"   Total tools: {len(tools)}")
    for tool in tools:
        print(f"   - {tool['function']['name']}")
    
    print("\n" + "=" * 70)
    print("✓ All CSV tool tests completed!")
    print("=" * 70)
