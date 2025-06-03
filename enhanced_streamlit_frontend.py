import streamlit as st
import pandas as pd
import numpy as np
import dspy
from dotenv import load_dotenv
import os
import sys
import io
import traceback
import contextlib
from datetime import datetime

# Attempt to import from the new 'src' package structure
try:
    from src.agents import AutoAnalyst, AutoAnalystIndividual, AVAILABLE_CORE_AGENTS, REGISTERED_ENHANCED_AGENTS
    from src.retrievers import make_data, initiatlize_retrievers, styling_instructions
    from src.memory_agents import memory_summarize_agent, error_memory_agent # These are signatures
except ImportError as e:
    st.error(f"Failed to import necessary modules from 'src'. Ensure the 'src' package is correctly structured and in PYTHONPATH. Error: {e}")
    # Add dummy classes/functions to prevent Streamlit from crashing immediately if imports fail
    class AutoAnalyst: pass
    class AutoAnalystIndividual: pass
    AVAILABLE_CORE_AGENTS = {}
    REGISTERED_ENHANCED_AGENTS = {}
    def make_data(*args, **kwargs): return "Error: make_data not loaded"
    def initiatlize_retrievers(*args, **kwargs): return {}
    styling_instructions = "Error: styling_instructions not loaded"
    class memory_summarize_agent: pass
    class error_memory_agent: pass


# --- Environment and Configuration ---
load_dotenv() # Load environment variables from .env file

# Configure DSPy LLM (OpenAI GPT-4o-mini by default)
# Ensure OPENAI_API_KEY is set in your environment or .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please set it in your environment variables or a .env file.")
    st.stop()

try:
    # dspy.configure(lm=dspy.OpenAI(model='gpt-4o-mini', api_key=OPENAI_API_KEY, max_tokens=4096, temperature=0.1))
    # For more complex tasks from enhanced agents, a larger model might be better if budget allows.
    # Using gpt-4-turbo as an example for potentially better results with complex agent interactions.
    # If using gpt-4o-mini, ensure prompts are very concise or use few-shot examples.
    llm_model_to_use = os.getenv("DSPY_LLM_MODEL", 'gpt-4o-mini') # Default to gpt-4o-mini
    max_tokens_for_model = int(os.getenv("DSPY_LLM_MAX_TOKENS", 4096))
    
    # Check for Groq API key if user wants to use Llama3 via Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if GROQ_API_KEY and llm_model_to_use.startswith("llama3"): # Example condition
        st.sidebar.info(f"Using Groq LLM: {llm_model_to_use}")
        dspy.configure(lm=dspy.GROQ(model=llm_model_to_use, api_key=GROQ_API_KEY, max_tokens=max_tokens_for_model))
    else:
        st.sidebar.info(f"Using OpenAI LLM: {llm_model_to_use}")
        dspy.configure(lm=dspy.OpenAI(model=llm_model_to_use, api_key=OPENAI_API_KEY, max_tokens=max_tokens_for_model, temperature=0.1))

except Exception as e:
    st.error(f"Error configuring DSPy LLM: {e}. Check your API key and model availability.")
    st.stop()

# Configure LlamaIndex Embeddings (OpenAI default)
try:
    from llama_index.core import Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)
except ImportError:
    st.warning("LlamaIndex or OpenAIEmbedding not found. Retrieval capabilities might be limited.")
except Exception as e:
    st.error(f"Error configuring LlamaIndex embeddings: {e}")


# --- Helper Functions ---
@contextlib.contextmanager
def stdout_capture():
    """Capture standard output for integration with Streamlit."""
    old_stdout = sys.stdout
    string_io = io.StringIO()
    sys.stdout = string_io
    try:
        yield string_io
    finally:
        sys.stdout = old_stdout

def reset_app_state():
    """Resets relevant parts of the session state, e.g., when a new file is uploaded."""
    st.cache_data.clear() # Clear all cached data functions
    st.cache_resource.clear() # Clear all cached resource functions
    
    keys_to_reset = [
        "df", "df_description", "retrievers", "auto_analyst_system", 
        "auto_analyst_individual_system", "messages", "st_memory", 
        "current_excel_sheets", "selected_excel_sheet", "data_loaded"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    # Re-initialize basic states
    st.session_state.messages = []
    st.session_state.st_memory = []
    st.session_state.data_loaded = False
    st.info("Application state has been reset. Please upload a new file or reload existing data.")

# --- Streamlit App Layout and Logic ---
st.set_page_config(page_title="Enhanced Auto-Analyst", layout="wide", initial_sidebar_state="expanded")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "st_memory" not in st.session_state:
    st.session_state.st_memory = [] # For agent context and short-term memory
if "df" not in st.session_state:
    st.session_state.df = None
if "df_description" not in st.session_state:
    st.session_state.df_description = ""
if "retrievers" not in st.session_state:
    st.session_state.retrievers = None
if "auto_analyst_system" not in st.session_state: # Planner-based system
    st.session_state.auto_analyst_system = None
if "auto_analyst_individual_system" not in st.session_state: # Direct agent call system
    st.session_state.auto_analyst_individual_system = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "current_excel_sheets" not in st.session_state:
    st.session_state.current_excel_sheets = []
if "selected_excel_sheet" not in st.session_state:
    st.session_state.selected_excel_sheet = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# --- Sidebar for File Upload and Configuration ---
with st.sidebar:
    st.image("images/Auto-analysts icon small.png", width=70)
    st.title("Enhanced Auto-Analyst 🚀")
    st.markdown("Upload your data (CSV or Excel) and let the AI agents assist you with in-depth analysis.")

    uploaded_file = st.file_uploader(
        "Upload Data File",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel file for analysis.",
        on_change=reset_app_state # Reset state if a new file is chosen
    )

    if uploaded_file:
        st.session_state.uploaded_file_name = uploaded_file.name
        if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
            try:
                # Get sheet names without parsing the whole file yet for performance
                xls = pd.ExcelFile(uploaded_file)
                st.session_state.current_excel_sheets = xls.sheet_names
                if not st.session_state.selected_excel_sheet or st.session_state.selected_excel_sheet not in st.session_state.current_excel_sheets:
                    st.session_state.selected_excel_sheet = st.session_state.current_excel_sheets[0]
                
                st.session_state.selected_excel_sheet = st.selectbox(
                    "Select Excel Sheet:",
                    st.session_state.current_excel_sheets,
                    index=st.session_state.current_excel_sheets.index(st.session_state.selected_excel_sheet)
                )
            except Exception as e:
                st.error(f"Error reading Excel file sheets: {e}")
                st.session_state.current_excel_sheets = []
    else: # Clear sheet selection if no file or CSV
        st.session_state.current_excel_sheets = []
        st.session_state.selected_excel_sheet = None


    dataset_name_user_input = st.text_input(
        "Dataset Name (Optional):",
        value=st.session_state.uploaded_file_name or "My Dataset",
        help="A descriptive name for your dataset."
    )
    
    dataset_description_user_input = st.text_area(
        "Dataset Description (Optional but Recommended):",
        height=100,
        placeholder="e.g., Monthly sales data for Q1 2023, including product category, region, and revenue. Or, customer survey responses regarding product satisfaction.",
        help="Provide context about your data for better analysis."
    )

    if st.button("Load Data & Initialize Agents", type="primary", disabled=not uploaded_file):
        with st.spinner("Processing data and initializing AI agents..."):
            try:
                # For Excel, pass the selected sheet name to make_data
                file_source_for_make_data = uploaded_file
                if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
                    # make_data needs the stream, not just the name
                    # We need to ensure the stream is reset if read multiple times
                    uploaded_file.seek(0) 
                
                df_desc_text = make_data(
                    file_source_for_make_data, 
                    dataset_name=dataset_name_user_input,
                    target_sheet_name=st.session_state.selected_excel_sheet if st.session_state.current_excel_sheets else None
                )
                st.session_state.df_description = df_desc_text
                
                # Load the actual DataFrame for agents to use
                # Reset stream before reading again
                uploaded_file.seek(0)
                if uploaded_file.name.lower().endswith(".csv"):
                    st.session_state.df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
                    st.session_state.df = pd.read_excel(uploaded_file, sheet_name=st.session_state.selected_excel_sheet)
                
                st.session_state.retrievers = initiatlize_retrievers(
                    st.session_state.df_description,
                    styling_instructions # Global styling instructions from retrievers.py
                )
                
                # Initialize agent systems
                st.session_state.auto_analyst_system = AutoAnalyst(retrievers=st.session_state.retrievers)
                st.session_state.auto_analyst_individual_system = AutoAnalystIndividual(retrievers=st.session_state.retrievers)
                
                st.session_state.data_loaded = True
                st.success("Data loaded and agents initialized successfully!")
                st.expander("View Dataset Schema Description").markdown(f"```text\n{st.session_state.df_description}\n```")
                if st.session_state.df is not None:
                    st.expander("View Loaded DataFrame (First 5 rows)").dataframe(st.session_state.df.head())

            except Exception as e:
                st.error(f"Error during data loading or agent initialization: {e}")
                st.exception(e)
                st.session_state.data_loaded = False
    
    st.markdown("---")
    if st.button("Reset Application State"):
        reset_app_state()

# --- Main Chat Area ---
st.header("Chat with Auto-Analyst")

if not st.session_state.data_loaded:
    st.info("Please upload a data file and click 'Load Data & Initialize Agents' in the sidebar to begin.")
else:
    # Display chat messages
    for msg_idx, message_content in enumerate(st.session_state.messages):
        role = message_content.get("role", "assistant") # Default to assistant if role not set
        content = message_content.get("content", "")
        avatar_map = {"user": "🧑‍💻", "assistant": "🤖", "system": "⚙️"}
        
        with st.chat_message(role, avatar=avatar_map.get(role)):
            # Content can be a simple string or a list of dicts for complex outputs
            if isinstance(content, list): # Handle structured content
                for item in content:
                    if item["type"] == "text":
                        st.markdown(item["text"])
                    elif item["type"] == "code":
                        st.code(item["code_content"], language=item.get("language", "python"))
                    elif item["type"] == "dataframe":
                        st.dataframe(pd.read_json(item["dataframe_json"], orient="split") if isinstance(item["dataframe_json"], str) else item["dataframe_json"])
                    elif item["type"] == "plot": # Assuming plot is a Plotly JSON spec
                        st.plotly_chart(item["plot_json"], use_container_width=True)
                    elif item["type"] == "kpi_card":
                        st.metric(label=item["label"], value=item["value"], delta=item.get("delta"))
                    # Add more types as needed (e.g., error, progress)
            else: # Simple string content
                st.markdown(content)

    # User input
    user_query = st.chat_input("Ask your data analysis question or specify an agent (e.g., @DataQualityAgent describe data issues)...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_query)

        # Process user query
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty() # For streaming-like updates
            full_response_content = [] # To store parts of a complex response

            with response_placeholder.status("Auto-Analyst is thinking...", expanded=True) as current_status:
                try:
                    # Check if it's a direct agent call
                    specified_agent_name = None
                    query_for_agent = user_query
                    
                    # List all available agent names for parsing
                    all_available_agent_names = list(AVAILABLE_CORE_AGENTS.keys()) + list(REGISTERED_ENHANCED_AGENTS.keys())

                    if user_query.startswith("@"):
                        parts = user_query.split(" ", 1)
                        potential_agent_name = parts[0][1:]
                        if potential_agent_name in all_available_agent_names:
                            specified_agent_name = potential_agent_name
                            query_for_agent = parts[1] if len(parts) > 1 else "Perform your default action or describe yourself."
                            st.write(f"Directing to **{specified_agent_name}**...")
                            current_status.update(label=f"{specified_agent_name} is processing...")
                            
                            agent_output = st.session_state.auto_analyst_individual_system.forward(
                                query=query_for_agent,
                                specified_agent_name=specified_agent_name,
                                df_description=st.session_state.df_description,
                                styling_desc=styling_instructions
                            )
                            # The AutoAnalystIndividual.forward method should ideally structure its output
                            # For now, let's assume it returns a dict that we can format nicely.
                            
                            if agent_output.get("error"):
                                full_response_content.append({"type": "text", "text": f"**Error from {specified_agent_name}:** {agent_output['error']}"})
                                if agent_output.get("traceback"):
                                    full_response_content.append({"type": "code", "code_content": agent_output['traceback'], "language":"text"})
                                if agent_output.get("fixed_code_attempt"):
                                    full_response_content.append({"type": "text", "text": "**Code Fix Attempt:**"})
                                    full_response_content.append({"type": "code", "code_content": agent_output['fixed_code_attempt']})
                                    if agent_output.get("fix_explanation"):
                                         full_response_content.append({"type": "text", "text": f"**Fix Explanation:** {agent_output['fix_explanation']}"})
                            else:
                                if agent_output.get("commentary"):
                                    full_response_content.append({"type": "text", "text": f"**{agent_output.get('agent_name', 'Agent')} Says:**\n{agent_output['commentary']}"})
                                if agent_output.get("code"):
                                    full_response_content.append({"type": "text", "text": "**Generated Code:**"})
                                    full_response_content.append({"type": "code", "code_content": agent_output['code']})
                                if agent_output.get("kpis"):
                                    full_response_content.append({"type": "text", "text": f"**KPIs:**\n{agent_output['kpis']}"})
                                if agent_output.get("execution_stdout"):
                                    full_response_content.append({"type": "text", "text": "**Execution Output:**"})
                                    full_response_content.append({"type": "code", "code_content": agent_output['execution_stdout'], "language":"text"})
                                # Add other fields from agent_output as needed

                        else:
                            full_response_content.append({"type": "text", "text": f"Sorry, agent '{potential_agent_name}' not found. Please use a valid agent name or ask a general question."})
                    else:
                        # General query for the planner-based system
                        st.write("Auto-Analyst (Planner Mode) is processing...")
                        current_status.update(label="Auto-Analyst (Planner Mode) is processing...")
                        
                        planner_output = st.session_state.auto_analyst_system.forward(
                            query=user_query,
                            df_description=st.session_state.df_description,
                            styling_desc=styling_instructions
                        )
                        # The AutoAnalyst.forward method should also structure its output
                        if planner_output.get("status") == "error":
                            full_response_content.append({"type": "text", "text": f"**Error:** {planner_output['message']}"})
                            if planner_output.get("traceback"):
                                full_response_content.append({"type": "code", "code_content": planner_output['traceback'], "language":"text"})
                        elif planner_output.get("status") == "clarification_needed":
                             full_response_content.append({"type": "text", "text": f"**Clarification Needed:** {planner_output['message']}"})
                        else: # Success
                            if planner_output.get("plan"):
                                full_response_content.append({"type": "text", "text": f"**Executed Plan:** `{planner_output['plan']}`\n\n**Plan Description:** {planner_output.get('plan_description', '')}"})
                            if planner_output.get("final_commentary"):
                                full_response_content.append({"type": "text", "text": f"**Final Commentary:**\n{planner_output['final_commentary']}"})
                            if planner_output.get("final_code"):
                                full_response_content.append({"type": "text", "text": "**Final Generated Code:**"})
                                full_response_content.append({"type": "code", "code_content": planner_output['final_code']})
                            if planner_output.get("final_kpis"):
                                full_response_content.append({"type": "text", "text": f"**Final KPIs:**\n{planner_output['final_kpis']}"})
                            if planner_output.get("execution_stdout"):
                                full_response_content.append({"type": "text", "text": "**Execution Output:**"})
                                full_response_content.append({"type": "code", "code_content": planner_output['execution_stdout'], "language":"text"})
                            # Add story if generated and returned by forward method

                    current_status.update(label="Processing complete!", state="complete", expanded=False)

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    full_response_content.append({"type": "text", "text": f"**System Error:** {str(e)}"})
                    st.exception(e)
                    current_status.update(label="Error occurred", state="error", expanded=True)

            # Update the placeholder with the final structured content
            response_placeholder.empty() # Clear the "thinking..." message
            # Render the full_response_content (which is now done by the loop at the top of the chat area after appending)
            st.session_state.messages.append({"role": "assistant", "content": full_response_content})
            st.rerun() # Rerun to display the new message immediately

    # Memory management: Keep last N interactions in st_memory
    if len(st.session_state.st_memory) > 10: # Keep last 10 memory items
        st.session_state.st_memory = st.session_state.st_memory[:10]

    with st.sidebar:
        st.markdown("---")
        st.subheader("Session Memory (Last 5)")
        if st.session_state.st_memory:
            for i, mem_item in enumerate(st.session_state.st_memory[:5]):
                st.text_area(f"Memory {i+1}", value=str(mem_item), height=50, disabled=True, key=f"mem_disp_{i}")
        else:
            st.caption("No memory items yet.")

st.markdown("---")
st.caption("Enhanced Auto-Analyst - Your AI Data Partner")
