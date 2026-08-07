import json
import os
from typing import Dict, Any, List


class ObservabilityLogger:
    """Manages structured logging and exports evaluation trace files."""

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_run_log(
        self,
        run_id: str,
        goal: str,
        execution_trace: List[Dict[str, Any]],
        final_status: str,
    ) -> str:
        """Saves a structured JSON file for an agent execution run."""
        log_payload = {
            "run_id": run_id,
            "goal": goal,
            "final_status": final_status,
            "total_steps": len(execution_trace),
            "trace": execution_trace,
        }
        filepath = os.path.join(self.output_dir, f"{run_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_payload, f, indent=2)
        return filepath


class CLIViewer:
    """Renders structured JSON logs as a formatted CLI table in the terminal."""

    @staticmethod
    def render(log_data: Dict[str, Any]):
        print("\n" + "=" * 70)
        print(
            f" AGENT EXECUTION LOG VIEWER | RUN ID: {log_data.get('run_id')}"
        )
        print("=" * 70)
        print(f"Goal: {log_data.get('goal')}")
        print(f"Final Status: {log_data.get('final_status')}")
        print(f"Total Steps Executed: {log_data.get('total_steps')}")
        print("-" * 70)

        for item in log_data.get("trace", []):
            step_num = item.get("step")
            event_type = item.get("type")
            details = item.get("details", {})

            if event_type == "REACT_TRACE":
                print(f"\n[STEP {step_num}] 🧠 REASONING & ACTION")
                print(f"  Thought: {details.get('thought')}")
                print(
                    f"  Tool Call: {details.get('tool')}({details.get('args')})"
                )

            elif event_type == "SELF_CORRECTION":
                print(f"\n[STEP {step_num}] ⚠️ SELF-CORRECTION TRIGGERED")
                print(f"  Failure Mode: {details.get('failure_mode')}")
                print(f"  Reason: {details.get('reason')}")
                print(
                    f"  Recovery Strategy: {details.get('recovery_plan', {}).get('message')}"
                )

            elif event_type == "STEP_SUCCESS":
                print(f"\n[STEP {step_num}] ✅ ACTION SUCCESS")
                print(f"  Result Output: {details.get('result')}")

            elif event_type == "FINAL_ANSWER":
                print(f"\n[STEP {step_num}] 🎯 FINAL ANSWER DELIVERED")
                print(f"  Output: {details.get('output')}")

        print("=" * 70 + "\n")


class HTMLViewerGenerator:
    """Generates a visual HTML log viewer file for browser viewing."""

    @staticmethod
    def generate_html(
        log_data: Dict[str, Any], output_path: str = "logs/viewer.html"
    ) -> str:
        # Ensure parent output directory exists before writing file
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Agent Execution Log - {log_data.get('run_id')}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid #334155; }}
        .title {{ color: #38bdf8; font-size: 1.25rem; font-weight: bold; margin-bottom: 0.5rem; }}
        .tag {{ display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.85rem; font-weight: bold; }}
        .tag-success {{ background: #166534; color: #4ade80; }}
        .tag-correction {{ background: #991b1b; color: #fca5a5; }}
        .tag-react {{ background: #1e40af; color: #93c5fd; }}
        pre {{ background: #0f172a; padding: 1rem; border-radius: 6px; overflow-x: auto; color: #cbd5e1; }}
    </style>
</head>
<body>
    <h1>🤖 Agent Execution Log Viewer</h1>
    <div class="card">
        <div class="title">Goal: {log_data.get('goal')}</div>
        <p><strong>Run ID:</strong> {log_data.get('run_id')} | <strong>Status:</strong> {log_data.get('final_status')}</p>
    </div>
"""
        for item in log_data.get("trace", []):
            event_type = item.get("type")
            details = item.get("details", {})
            tag_class = "tag-react"
            if event_type == "SELF_CORRECTION":
                tag_class = "tag-correction"
            elif event_type in ["STEP_SUCCESS", "FINAL_ANSWER"]:
                tag_class = "tag-success"

            html_content += f"""
    <div class="card">
        <span class="tag {tag_class}">{event_type} (Step {item.get('step')})</span>
        <pre>{json.dumps(details, indent=2)}</pre>
    </div>"""

        html_content += "\n</body>\n</html>"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path