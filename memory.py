from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json

@dataclass
class WorkingMemory:
    """
    Explicit stateful working memory for the agent.
    Tracks goal progression, accumulated knowledge, and subtask status.
    """
    original_goal: str
    current_plan: List[str] = field(default_factory=list)
    completed_subtasks: List[Dict[str, Any]] = field(default_factory=list)
    observations: Dict[str, Any] = field(default_factory=dict)
    unresolvable_subtasks: List[Dict[str, str]] = field(default_factory=list)

    def add_observation(self, key: str, value: Any) -> None:
        """Stores a key insight or extracted fact."""
        self.observations[key] = value

    def mark_completed(self, subtask: str, result: Any) -> None:
        """Records a successfully completed subtask."""
        self.completed_subtasks.append({
            "subtask": subtask,
            "result": result
        })
        if subtask in self.current_plan:
            self.current_plan.remove(subtask)

    def mark_unresolvable(self, subtask: str, reason: str) -> None:
        """Flags a subtask as unresolvable when budget is exceeded."""
        self.unresolvable_subtasks.append({
            "subtask": subtask,
            "reason": reason
        })
        if subtask in self.current_plan:
            self.current_plan.remove(subtask)

    def to_context_prompt(self) -> str:
        """
        Formats working memory into a clean, structured string 
        to inject into the LLM system prompt.
        """
        context = {
            "Original Goal": self.original_goal,
            "Remaining Plan": self.current_plan,
            "Completed Subtasks": [item["subtask"] for item in self.completed_subtasks],
            "Unresolvable/Skipped Subtasks": self.unresolvable_subtasks,
            "Discovered Facts & Findings": self.observations
        }
        return f"=== CURRENT WORKING MEMORY STATE ===\n{json.dumps(context, indent=2)}\n===================================="

if __name__ == "__main__":
    # Initialize Memory
    memory = WorkingMemory(original_goal="Scrape product stats and calculate average price.")
    memory.current_plan = [
        "Fetch raw API data", 
        "Parse prices", 
        "Calculate mean price", 
        "Generate summary report"
    ]

    # Simulate Step 1 Completion
    memory.mark_completed("Fetch raw API data", {"status": 200})
    memory.add_observation("raw_prices", [10, 20, 30])

    # Simulate Step 2 Failure & Exceeded Budget
    memory.mark_unresolvable("Parse prices", "API format continuously returned 503")

    # Print Formatted Context
    print(memory.to_context_prompt())