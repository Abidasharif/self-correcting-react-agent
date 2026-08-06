from enum import Enum
from typing import Dict, Any, Tuple, Optional
import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Now import your modules
from src.memory import WorkingMemory

class FailureMode(Enum):
    NONE = "NONE"
    TOOL_FAILURE = "TOOL_FAILURE"             # Hard execution crashes, API HTTP 5xx, schema errors
    RESULT_INCONSISTENCY = "RESULT_INCONSISTENCY" # Tool executed OK, but returned empty or bad output
    GOAL_DRIFT = "GOAL_DRIFT"                 # Execution is repeating or diverging from original goal

class StepEvaluator:
    """
    Evaluates step execution results against working memory and goal context.
    """
    @staticmethod
    def evaluate(
        goal: str,
        subtask: str,
        tool_name: str,
        tool_result: Dict[str, Any],
        memory: WorkingMemory
    ) -> Tuple[FailureMode, str]:
        """
        Analyzes tool output and detects failures.
        Returns a tuple of (FailureMode, Diagnostic Reason).
        """
        # 1. Check for Tool Failure (Execution exception or non-200 HTTP codes)
        if tool_result.get("status") == "error":
            error_msg = tool_result.get("error", "Unknown tool runtime failure")
            return FailureMode.TOOL_FAILURE, f"Tool '{tool_name}' failed execution: {error_msg}"

        # 2. Check for Result Inconsistency (Executed without crashing, but returned bad payload)
        output = tool_result.get("output") or tool_result.get("results")
        if output is None or output == [] or output == "":
            return (
                FailureMode.RESULT_INCONSISTENCY,
                f"Action '{subtask}' succeeded technically, but returned empty output."
            )

        # 3. Check for Goal Drift (Agent stuck in a loop without populating new facts)
        recent_subtasks = [task["subtask"] for task in memory.completed_subtasks[-3:]]
        if len(recent_subtasks) >= 3 and len(set(recent_subtasks)) == 1:
            return (
                FailureMode.GOAL_DRIFT,
                f"Agent is executing redundant subtask '{subtask}' repeatedly."
            )

        return FailureMode.NONE, "Action executed cleanly and produced valid observations."

class RecoveryEngine:
    """
    Applies recovery strategies based on failure modes and enforces replanning budgets.
    """
    def __init__(self, max_subtask_budget: int = 2):
        self.max_subtask_budget = max_subtask_budget
        self.subtask_attempts: Dict[str, int] = {}

    def get_attempt_count(self, subtask: str) -> int:
        return self.subtask_attempts.get(subtask, 0)

    def record_attempt(self, subtask: str) -> None:
        self.subtask_attempts[subtask] = self.get_attempt_count(subtask) + 1

    def handle_failure(
        self,
        failure_mode: FailureMode,
        reason: str,
        subtask: str,
        memory: WorkingMemory
    ) -> Dict[str, Any]:
        """
        Determines recovery plan or flags subtask as unresolvable if budget is exceeded.
        """
        self.record_attempt(subtask)
        attempts = self.get_attempt_count(subtask)

        # Check budget limit
        if attempts > self.max_subtask_budget:
            memory.mark_unresolvable(
                subtask=subtask,
                reason=f"Exceeded recovery budget ({self.max_subtask_budget} attempts). Last error: {reason}"
            )
            return {
                "action": "ABANDON_SUBTASK",
                "message": f"Budget limit reached for '{subtask}'. Marked unresolvable and proceeding.",
                "attempts": attempts
            }

        # Targeted Recovery Strategies
        if failure_mode == FailureMode.TOOL_FAILURE:
            return {
                "action": "RETRY_WITH_FALLBACK",
                "message": f"Tool failure detected. Attempt {attempts}/{self.max_subtask_budget}. Fallback or alternate tool required.",
                "attempts": attempts
            }

        elif failure_mode == FailureMode.RESULT_INCONSISTENCY:
            return {
                "action": "MODIFY_QUERY_PARAMS",
                "message": f"Inconsistent result. Attempt {attempts}/{self.max_subtask_budget}. Broadening search or parameter criteria.",
                "attempts": attempts
            }

        elif failure_mode == FailureMode.GOAL_DRIFT:
            return {
                "action": "REPLAN_PIPELINE",
                "message": f"Goal drift detected. Forcing pipeline replan from Working Memory state.",
                "attempts": attempts
            }

        return {"action": "CONTINUE", "message": "No issues detected.", "attempts": attempts}

if __name__ == "__main__":
    memory = WorkingMemory(original_goal="Scrape user profile data")
    evaluator = StepEvaluator()
    recovery = RecoveryEngine(max_subtask_budget=2)

    # Test 1: Simulated Tool Failure (HTTP 503 from flakey_api_call)
    failed_tool_res = {"status": "error", "error": "HTTP 503 Service Unavailable"}
    mode, reason = evaluator.evaluate("Goal", "Fetch Profile", "flakey_api_call", failed_tool_res, memory)
    rec_plan = recovery.handle_failure(mode, reason, "Fetch Profile", memory)

    print("Test 1 Failure Mode:", mode.value)
    print("Test 1 Recovery Action:", rec_plan)

    # Test 2: Simulating Exceeded Budget Limit
    rec_plan_2 = recovery.handle_failure(mode, reason, "Fetch Profile", memory)
    rec_plan_3 = recovery.handle_failure(mode, reason, "Fetch Profile", memory)
    print("\nTest 2 Exceeded Budget Recovery:", rec_plan_3)
    print("Working Memory State:", memory.unresolvable_subtasks)