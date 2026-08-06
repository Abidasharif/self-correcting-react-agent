import sys
from pathlib import Path

# Fix Python path so Streamlit Cloud can find the src module
sys.path.append(str(Path(__file__).resolve().parent))

# Your existing imports continue below:
import os
import json
import streamlit as st
from openai import OpenAI
from src.agent import SelfCorrectingAgent

st.set_page_config(
    page_title="Self-Correcting ReAct Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Autonomous Self-Correcting ReAct Agent")
st.markdown("Enter a task goal below. The agent will execute step-by-step actions, monitor for tool failures or invalid schema outputs, and self-correct automatically.")

# Sidebar - API Key and Settings Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Allow user to provide API Key or fall back to Streamlit Secrets / Env Vars
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
        
        # Initialize OpenAI Client
        client = OpenAI(api_key=api_key) if api_key else None
        
        # Initialize Agent
        agent = SelfCorrectingAgent(
            goal=goal,
            max_steps=max_steps,
            recovery_budget=recovery_budget,
            client=client
        )
        
        # Container for streaming traces
        trace_container = st.container()
        
        with st.spinner("Agent is processing and self-correcting..."):
            result = agent.run()
            
        st.success("Task Execution Finished!")
        
        # Display Final Answer
        st.subheader("🎯 Final Result")
        st.write(result)
        
        # Display Execution Logs
        st.subheader("📊 Structured Execution Trace")
        for log in agent.execution_logs:
            with st.expander(f"Step {log['step']}: {log['type']}"):
                st.json(log['details'])
