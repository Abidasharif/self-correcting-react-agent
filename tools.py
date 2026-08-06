from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, Optional

# --- Tool Input Schemas ---

class WebSearchInput(BaseModel):
    """Schema for web search queries."""
    query: str = Field(..., description="The search query string to execute.")
    num_results: int = Field(default=3, description="Number of search results to retrieve.")

class APIFetchInput(BaseModel):
    """Schema for API endpoint data fetching."""
    endpoint: str = Field(..., description="The API target endpoint (e.g., '/api/v1/data').")

class PythonExecInput(BaseModel):
    """Schema for Python code execution."""
    code: str = Field(..., description="Valid Python code snippet to execute. Must define a variable 'result'.")

class JSONTransformInput(BaseModel):
    """Schema for transforming raw data into structured JSON."""
    raw_data: str = Field(..., description="Raw text or stringified data to transform.")
    target_key: str = Field(..., description="Key to extract from the parsed content.")


import random
import time

# --- Tool 1: Flakey API Fetcher (Intentionally Broken Tool) ---
def flakey_api_call(endpoint: str) -> Dict[str, Any]:
    """
    Fetches data from an endpoint but fails ~40% of the time 
    with a 503 error to test agent recovery.
    """
    # Simulate API latency
    time.sleep(0.5)
    
    # Simulate intermittent HTTP 503 error
    if random.random() < 0.4:
        raise ConnectionError("HTTP 503 Service Unavailable: Intermittent Gateway Timeout")
    
    # Successful mock response
    return {
        "status": 200,
        "endpoint": endpoint,
        "data": {
            "metrics": [12.5, 45.0, 78.2],
            "status_flag": "ACTIVE"
        }
    }

# --- Tool 2: Python Code Evaluator ---
def python_evaluator(code: str) -> Dict[str, Any]:
    """
    Safely executes Python code within an isolated local scope.
    Returns structured results or catches runtime exceptions.
    """
    local_scope = {}
    try:
        # Standard execution within local context
        exec(code, {}, local_scope)
        
        # Check if the code defined an explicit output variable
        output = local_scope.get("result", "Execution completed without returning a explicit 'result' variable.")
        return {"status": "success", "output": output}
        
    except Exception as e:
        # Catch syntax/runtime errors and pass them back gracefully to the agent
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }

# --- Tool 3: Web Search Mock/Real ---
def web_search(query: str, num_results: int = 3) -> Dict[str, Any]:
    """
    Simulates a web search query. Returns empty results if query is too vague.
    """
    if len(query.strip()) < 3:
        return {"status": "warning", "output": [], "message": "Query too short to yield results."}
        
    return {
        "status": "success",
        "results": [
            {"title": f"Result for {query}", "snippet": f"Data extracted for {query} analysis."}
            for _ in range(num_results)
        ]
    }
TOOL_REGISTRY = {
    "web_search": (web_search, WebSearchInput),
    "flakey_api_call": (flakey_api_call, APIFetchInput),
    "python_evaluator": (python_evaluator, PythonExecInput)
}

def execute_tool_safely(tool_name: str, raw_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates LLM tool arguments using Pydantic schemas and executes the tool.
    Handles schema validation errors and runtime execution crashes gracefully.
    """
    if tool_name not in TOOL_REGISTRY:
        return {
            "status": "error", 
            "error": f"Tool '{tool_name}' does not exist. Available tools: {list(TOOL_REGISTRY.keys())}"
        }

    tool_func, schema_class = TOOL_REGISTRY[tool_name]

    # 1. Pydantic Schema Validation Step
    try:
        validated_args = schema_class(**raw_args)
    except ValidationError as val_err:
        return {
            "status": "error",
            "error": f"Schema Validation Error: Bad arguments provided for tool '{tool_name}'.",
            "details": val_err.errors()
        }

    # 2. Tool Execution Step
    try:
        result = tool_func(**validated_args.model_dump())
        return result
    except Exception as exec_err:
        return {
            "status": "error",
            "error": f"Runtime Exception during tool execution: {str(exec_err)}"
        }

if __name__ == "__main__":
    # Test 1: Invalid Arguments (Schema Validation Failure)
    res1 = execute_tool_safely("web_search", {"wrong_key": "test"})
    print("Test 1 (Schema Error):", res1)

    # Test 2: Broken Code Execution (Python Error)
    res2 = execute_tool_safely("python_evaluator", {"code": "x = 1 / 0"})
    print("Test 2 (Code Error):", res2)

    # Test 3: Intermittent Flakey Tool Failure
    print("Test 3 (Flakey API Calls):")
    for i in range(3):
        res3 = execute_tool_safely("flakey_api_call", {"endpoint": "/data"})
        print(f" Run {i+1}:", res3)