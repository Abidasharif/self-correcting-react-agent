import sys
import os
from pathlib import Path

# Fix Streamlit Cloud root path resolution
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure src directory is explicitly resolvable
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
import streamlit as st
from openai import OpenAI

# Safe import trying package resolution first, then direct module fallback
try:
    from src.agent import SelfCorrectingAgent
except ModuleNotFoundError:
    from agent import SelfCorrectingAgent

st.set_page_config(
    page_title="Self-Correcting ReAct Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Autonomous Self-Correcting ReAct Agent")
st.markdown("Enter a task goal below. The agent will execute step-by-step actions, monitor for tool failures or invalid schema outputs, and self-correct automatically.")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input("OpenAI API Key (Optional if set in Secrets):", type="password")
    api_key = api_key_input if api_key_input else os.getenv("OPENAI_API_KEY")
    
    max_steps = st.slider("Max Execution Steps:", min_value=3, max_value=15, value=8)
    recovery_budget = st.slider("Subtask Retry Budget:", min_value=1, max_value=5, value=2)

# Goal Input Form
goal = st.text_area(
    "Goal Prompt:",
    placeholder="e.g., Fetch data from intermittent endpoint /api/users and compute metrics.",
    height=100
)

if st.button("🚀 Run Agent Task", type="primary"):
    if not goal.strip():
        st.warning("Please enter a goal prompt before running.")
    else:
        st.info("Initializing Agent Loop...")
        
        client = OpenAI(api_key=api_key) if api_key else None
        
        agent = SelfCorrectingAgent(
            goal=goal,
            max_steps=max_steps,
            recovery_budget=recovery_budget,
            client=client
        )
        
        with st.spinner("Agent is processing and self-correcting..."):
            result = agent.run()
            
        st.success("Task Execution Finished!")
        
        st.subheader("🎯 Final Result")
        st.write(result)
        
        st.subheader("📊 Structured Execution Trace")
        for log in agent.execution_logs:
            with st.expander(f"Step {log['step']}: {log['type']}"):
                st.json(log['details'])