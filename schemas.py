from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal

class AgentAction(BaseModel):
    thought: str = Field(..., description="Explicit reasoning step based on current Working Memory state.")
    tool: str = Field(..., description="Tool to execute: 'web_search', 'flakey_api_call', or 'python_evaluator'.")
    args: Dict[str, Any] = Field(..., description="Key-value arguments matching the tool's schema.")

class FinalAnswer(BaseModel):
    thought: str = Field(..., description="Reasoning explaining why the goal is complete.")
    final_answer: str = Field(..., description="The complete final deliverable or answer.")

class AgentStepResponse(BaseModel):
    response_type: Literal["ACTION", "FINAL_ANSWER"] = Field(..., description="Specify if taking an action or delivering final answer.")
    action: Optional[AgentAction] = Field(None, description="Required if response_type is 'ACTION'.")
    final_answer: Optional[FinalAnswer] = Field(None, description="Required if response_type is 'FINAL_ANSWER'.")