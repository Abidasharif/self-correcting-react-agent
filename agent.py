import json
import os
import re
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from openai import OpenAI

# 1. Fix module resolution path (Project Root)
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 2. Local module imports
from src.evaluator import FailureMode, RecoveryEngine, StepEvaluator
from src.memory import WorkingMemory
from src.observability import CLIViewer, HTMLViewerGenerator, ObservabilityLogger
from src.schemas import AgentStepResponse
from src.tools import execute_tool_safely

SYSTEM_PROMPT = """You are an autonomous AI Agent built on the ReAct pattern.
Achieve the user's goal by calling tools step-by-step.

{working_memory_context}

AVAILABLE TOOLS:
1. web_search(query: str, num_results: int) -> Search web for information.
2. flakey_api_call(endpoint: str) -> Fetch endpoint data (may timeout intermittently).
3. python_evaluator(code: str) -> Execute Python code.

RESPONSE FORMAT RULES:
You MUST respond strictly with valid JSON following one of these two structures:

To call a tool:
{{
  "response_type": "ACTION",
  "action": {{
    "thought": "<Your step-by-step reasoning>",
    "tool": "<tool_name>",
    "args": {{ "param": "value" }}
  }}
}}

To finish:
{{
  "response_type": "FINAL_ANSWER",
  "final_answer": {{
    "thought": "<Reasoning why goal is met>",
    "final_answer": "<Final result deliverable>"
  }}
}}
"""


class SelfCorrectingAgent:

    def __init__(
        self,
        goal: str,
        max_steps: int = 10,
        recovery_budget: int = 2,
        client: Optional[OpenAI] = None,
    ):
        self.goal = goal
        self.max_steps = max_steps
        self.memory = WorkingMemory(original_goal=goal)
        self.evaluator = StepEvaluator()
        self.recovery = RecoveryEngine(max_subtask_budget=recovery_budget)
        self.client = client
        self.execution_logs = []
        self.logger = ObservabilityLogger()

    def log_event(self, event_type: str, details: Dict[str, Any]):
        """Logs structured events for observability."""
        self.execution_logs.append(
            {
                "step": len(self.execution_logs) + 1,
                "type": event_type,
                "details": details,
            }
        )

    def parse_llm_response(self, text: str) -> Dict[str, Any]:
        """Parses LLM response using Pydantic JSON validation with fallback."""
        clean_text = text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:]
        clean_text = clean_text.strip()

        # Try Pydantic JSON Parsing
        try:
            data = json.loads(clean_text)
            parsed = AgentStepResponse(**data)

            if parsed.response_type == "FINAL_ANSWER" and parsed.final_answer:
                return {
                    "type": "FINAL_ANSWER",
                    "thought": parsed.final_answer.thought,
                    "answer": parsed.final_answer.final_answer,
                }

            if parsed.response_type == "ACTION" and parsed.action:
                return {
                    "type": "ACTION",
                    "thought": parsed.action.thought,
                    "tool": parsed.action.tool,
                    "args": parsed.action.args,
                }
        except Exception:
            pass

        # Fallback to Text/Regex ReAct Parsing
        if "Final Answer:" in text:
            final_part = text.split("Final Answer:")[1].strip()
            return {"type": "FINAL_ANSWER", "answer": final_part}

        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL
        )
        action_match = re.search(
            r"Action:\s*(.*?)(?=\nAction Input:|$)", text, re.DOTALL
        )
        input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)

        if thought_match and action_match and input_match:
            try:
                action_args = json.loads(input_match.group(1).strip())
                return {
                    "type": "ACTION",
                    "thought": thought_match.group(1).strip(),
                    "tool": action_match.group(1).strip(),
                    "args": action_args,
                }
            except json.JSONDecodeError as e:
                return {
                    "type": "PARSING_ERROR",
                    "error": f"Invalid JSON in Action Input: {str(e)}",
                }

        return {
            "type": "PARSING_ERROR",
            "error": "LLM response did not adhere to required JSON or ReAct structure.",
        }

    def _call_llm_api(self, prompt: str) -> str:
        """Invokes the LLM API using direct client or fallback simulation."""
        if self.client:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content

        # Fallback simulation trace for testing offline without API key:
        if len(self.execution_logs) == 0:
            return json.dumps(
                {
                    "response_type": "ACTION",
                    "action": {
                        "thought": "I need to fetch user data.",
                        "tool": "flakey_api_call",
                        "args": {"endpoint": "/api/users"},
                    },
                }
            )
        else:
            return json.dumps(
                {
                    "response_type": "FINAL_ANSWER",
                    "final_answer": {
                        "thought": "Processing complete.",
                        "final_answer": "Extracted user statistics successfully.",
                    },
                }
            )

    def run(self) -> str:
        """Executes the autonomous loop with self-correction."""
        step_count = 0
        final_result = (
            "Goal execution reached maximum allowed steps without completion."
        )

        while step_count < self.max_steps:
            step_count += 1
            print(f"\n--- [AGENT STEP {step_count}/{self.max_steps}] ---")

            # 1. Format System Prompt with Memory Context
            memory_context = self.memory.to_context_prompt()
            prompt = SYSTEM_PROMPT.format(working_memory_context=memory_context)

            # 2. Call LLM
            llm_text = self._call_llm_api(prompt)

            # 3. Parse Response
            parsed = self.parse_llm_response(llm_text)

            if parsed["type"] == "FINAL_ANSWER":
                print(f"[FINAL ANSWER RECEIVED]: {parsed['answer']}")
                self.log_event("FINAL_ANSWER", {"output": parsed["answer"]})
                final_result = parsed["answer"]
                break

            if parsed["type"] == "PARSING_ERROR":
                error_msg = parsed["error"]
                print(f"[FORMAT ERROR]: {error_msg}")
                self.log_event("PARSING_ERROR", {"error": error_msg})
                correction_prompt = (
                    f"CRITICAL FORMAT ERROR: {error_msg}\n"
                    "Please output valid JSON matching the schema format."
                )
                self.memory.add_observation(
                    key=f"step_{step_count}_format_error",
                    value=correction_prompt,
                )
                continue

            # 4. Extract Thought and Action
            thought = parsed["thought"]
            tool_name = parsed["tool"]
            tool_args = parsed["args"]

            print(f"[REASONING TRACE]: {thought}")
            print(f"[ACTION]: {tool_name} with args {tool_args}")
            self.log_event(
                "REACT_TRACE",
                {"thought": thought, "tool": tool_name, "args": tool_args},
            )

            # 5. Safe Tool Execution
            tool_result = execute_tool_safely(tool_name, tool_args)
            print(f"[TOOL RESULT]: {tool_result}")

            # 6. Failure Detection
            failure_mode, reason = self.evaluator.evaluate(
                goal=self.goal,
                subtask=thought,
                tool_name=tool_name,
                tool_result=tool_result,
                memory=self.memory,
            )

            # 7. Self-Correction & Recovery Branching
            if failure_mode != FailureMode.NONE:
                print(f"[FAILURE DETECTED]: {failure_mode.value} -> {reason}")
                recovery_plan = self.recovery.handle_failure(
                    failure_mode=failure_mode,
                    reason=reason,
                    subtask=thought,
                    memory=self.memory,
                )
                print(
                    f"[RECOVERY STRATEGY]: {recovery_plan.get('message', '')}"
                )
                self.log_event(
                    "SELF_CORRECTION",
                    {
                        "failure_mode": failure_mode.value,
                        "reason": reason,
                        "recovery_plan": recovery_plan,
                    },
                )
            else:
                self.memory.mark_completed(subtask=thought, result=tool_result)
                self.memory.add_observation(
                    key=f"step_{step_count}_output", value=tool_result
                )
                self.log_event("STEP_SUCCESS", {"result": tool_result})

        # Render Log Trace
        log_path = self.logger.save_run_log(
            run_id="run_001",
            goal=self.goal,
            execution_trace=self.execution_logs,
            final_status=(
                "SUCCESS"
                if any(x["type"] == "FINAL_ANSWER" for x in self.execution_logs)
                else "FAILED"
            ),
        )

        with open(log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)
            CLIViewer.render(log_data)
            HTMLViewerGenerator.generate_html(log_data)

        return final_result


if __name__ == "__main__":
    api_key = os.getenv("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")

    if api_key:
        client = OpenAI(api_key=api_key)
        agent = SelfCorrectingAgent(
            goal="Scrape user data and analyze results", client=client
        )
    else:
        print("[NOTICE] Running agent in simulated fallback mode...")
        agent = SelfCorrectingAgent(
            goal="Test agent self-correction mechanics"
        )

    result = agent.run()
    print(f"\nExecution Result: {result}")