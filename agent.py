import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
from openai import OpenAI

# Force project root directory into sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local package imports
try:
    from src.memory import WorkingMemory
    from src.tools import execute_tool_safely
    from src.evaluator import StepEvaluator, RecoveryEngine, FailureMode
    from src.observability import ObservabilityLogger, CLIViewer, HTMLViewerGenerator
    from src.schemas import AgentStepResponse
except ModuleNotFoundError:
    # Fallback if executing directly inside src directory
    from memory import WorkingMemory
    from tools import execute_tool_safely
    from evaluator import StepEvaluator, RecoveryEngine, FailureMode
    from observability import ObservabilityLogger, CLIViewer, HTMLViewerGenerator
    from schemas import AgentStepResponse
