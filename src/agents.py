import dspy
import streamlit as st
import pandas as pd
import numpy as np
import sys
import io
import traceback
import contextlib

# Assuming memory_agents.py is in the same directory or accessible in PYTHONPATH
from .memory_agents import memory_summarize_agent, error_memory_agent

# Placeholders for enhanced agents - these will be imported from src.enhanced_agents
# This allows the file to be parsed and core logic to be defined,
# actual enhanced agent classes will be in their respective files.
try:
    from .enhanced_agents.statistical import EnhancedStatisticalAgent
    from .enhanced_agents.ml import AdvancedMLAgent
    from .enhanced_agents.kpi import KPIGeneratorAgent
    from .enhanced_agents.quality import DataQualityAgent
    from .enhanced_agents.visualization import AdvancedVisualizationAgent
    from .enhanced_agents.excel_integration import ExcelIntegrationAgent
    from .enhanced_agents.coordinator import AgentCoordinator # Key orchestrator
except ImportError as e:
    print(f"Warning: Could not import one or more enhanced agents: {e}. Some features might be unavailable.")
    EnhancedStatisticalAgent = None
    AdvancedMLAgent = None
    KPIGeneratorAgent = None
    DataQualityAgent = None
    AdvancedVisualizationAgent = None
    ExcelIntegrationAgent = None
    AgentCoordinator = None


# Context manager for capturing stdout
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

# --- Agent Signatures ---

class AnalyticalPlanner(dspy.Signature):
    """
    You are a data analytics planner agent. You have access to:
    1. Dataset(s) description (including column names, types, and potentially relationships if multiple files/sheets).
    2. Descriptions of available Data Agents and their capabilities.
    3. A User-defined Goal.

    Your task is to develop a comprehensive, step-by-step plan to achieve the user-defined goal using the available data and agents.
    The plan should be a sequence of agent calls. If the goal involves multiple datasets or Excel sheets, include the ExcelIntegrationAgent first.
    If data quality is a concern or implied by the goal, consider starting with the DataQualityAgent.
    For complex tasks requiring multiple analysis steps, sequence the agents logically (e.g., Preprocessing -> ML -> Visualization).
    The AgentCoordinator will handle the execution and combination of outputs from the planned agents.

    Available Agents:
    - PreprocessingAgent: Cleans data, handles missing values, feature scaling, type conversion.
    - BasicStatisticalAgent: Performs basic statistical tests, descriptive statistics.
    - BasicMLAgent: Builds simple machine learning models using scikit-learn.
    - BasicVisualizationAgent: Generates standard plots using Plotly.
    - EnhancedStatisticalAgent: Advanced time series analysis, complex hypothesis testing, Bayesian methods.
    - AdvancedMLAgent: Automated feature engineering, hyperparameter optimization, ensemble models, deep learning.
    - KPIGeneratorAgent: Identifies and calculates relevant Key Performance Indicators for business analysis.
    - DataQualityAgent: In-depth data profiling, quality assessment, and cleaning recommendations.
    - AdvancedVisualizationAgent: Creates interactive dashboards, complex multi-dimensional plots.
    - ExcelIntegrationAgent: Handles multiple Excel files/sheets, merges data, resolves schema conflicts.
    - AgentCoordinator: Orchestrates the execution of a multi-agent plan, combines their outputs (code, commentary, KPIs) into a cohesive result. It is implicitly used to run the plan.

    Output Format:
    plan: AgentName1->AgentName2->AgentName3
    plan_desc: Justification for each step in the plan. Explain why each agent is chosen and in that order.
    If the user's goal is infeasible or unclear, ask for clarification instead of creating a plan.
    You don't have to use all agents. The AgentCoordinator will manage the overall execution flow.
    """
    dataset_description = dspy.InputField(desc="Description of available dataset(s), including column names, types, and relationships. For Excel, mention sheet names and their purposes if known.")
    agent_descriptions = dspy.InputField(desc="Detailed descriptions of all available specialized AI agents and their capabilities.")
    user_goal = dspy.InputField(desc="The user-defined objective for the data analysis.")
    plan = dspy.OutputField(desc="A sequence of agent names (e.g., ExcelIntegrationAgent->DataQualityAgent->PreprocessingAgent->AdvancedMLAgent->AdvancedVisualizationAgent) to achieve the goal. This plan will be executed by the AgentCoordinator.", prefix="Plan:")
    plan_desc = dspy.OutputField(desc="A detailed explanation of the plan, justifying the choice and order of agents.")

class GoalRefinerAgent(dspy.Signature):
    """
    You are an AI assistant. Given a user's data analysis goal, dataset description, and available agent descriptions,
    your task is to refine the goal to be more specific, actionable, and elaborate. This helps the AnalyticalPlanner create a better plan.
    If the goal is already clear and detailed, you can state that no refinement is needed.
    """
    dataset_description = dspy.InputField(desc="Description of available dataset(s).")
    agent_descriptions = dspy.InputField(desc="Descriptions of available AI agents.")
    user_goal = dspy.InputField(desc="The user-defined goal.")
    refined_goal = dspy.OutputField(desc="A more elaborate and specific version of the user's goal, or a statement that the goal is already clear.")

class PreprocessingAgent(dspy.Signature):
    """
    Given a user-defined analysis goal and a pre-loaded dataset (as a pandas DataFrame named `df`),
    generate Python code using NumPy and Pandas for data preprocessing.
    Tasks include:
    - Identifying numeric and categorical columns.
    - Handling missing values (imputation or removal based on context).
    - Converting data types (e.g., string dates to datetime objects, object to numeric).
    - Feature scaling or normalization if appropriate for the goal.
    - Basic feature engineering (e.g., extracting year from date) if simple and directly relevant.
    - Ensure the DataFrame `df` is modified in place or reassigned.

    Use `st.write()` for any textual output. For visualizations (e.g., missing value plots), use Plotly and `st.plotly_chart()`.
    The input `dataset_description` provides schema info. `hint` provides recent interaction history.
    If the `DataQualityAgent` was run previously, its findings might be in the `hint`.
    """
    dataset_description = dspy.InputField(desc="Schema of the DataFrame `df` (columns, dtypes).")
    user_goal = dspy.InputField(desc="The user's overall analysis goal.")
    hint = dspy.InputField(desc="Recent interaction history or outputs from other agents.", default="")
    code = dspy.OutputField(desc="Python code for data preprocessing.")
    commentary = dspy.OutputField(desc="Explanation of the preprocessing steps taken and why.")
    kpis = dspy.OutputField(desc="Optional: Key metrics related to preprocessing, e.g., percentage of missing values handled.", default="")


class BasicStatisticalAgent(dspy.Signature):
    """
    You are a statistical analytics agent. Given a dataset (`df`) and a user goal,
    output Python code using `statsmodels` or `scipy.stats` for basic statistical analysis.
    Capabilities:
    - Descriptive statistics (mean, median, mode, variance, quantiles).
    - Basic hypothesis tests (e.g., t-tests, chi-square tests for simple comparisons).
    - Basic correlation analysis (Pearson correlation for numeric features).
    - Confidence interval estimation for means or proportions.

    Ensure code is executable. Use `st.write()` for results and `st.plotly_chart()` for any simple plots (e.g., histograms for distributions, basic scatter for correlation).
    The `EnhancedStatisticalAgent` handles more complex analyses.
    """
    dataset_description = dspy.InputField(desc="Schema of the DataFrame `df`.")
    user_goal = dspy.InputField(desc="The user's goal for statistical analysis.")
    hint = dspy.InputField(desc="Recent interaction history.", default="")
    code = dspy.OutputField(desc="Python code for basic statistical analysis.")
    commentary = dspy.OutputField(desc="Explanation of the analysis and interpretation of results.")
    kpis = dspy.OutputField(desc="Optional: Key statistical results, e.g., p-values, correlation coefficients.", default="")


class BasicMLAgent(dspy.Signature):
    """
    You are a machine learning agent. Given a dataset (`df`) and a user goal (e.g., "predict price", "classify customers"),
    output Python code using `scikit-learn` for basic ML tasks.
    Capabilities:
    - Simple classification (e.g., Logistic Regression, Decision Tree Classifier).
    - Simple regression (e.g., Linear Regression, Decision Tree Regressor).
    - Basic model training and prediction.
    - Simple evaluation metrics (e.g., accuracy for classification, MSE/R2 for regression).

    Assume data is already preprocessed. Use `st.write()` for model summaries and evaluation metrics.
    The `AdvancedMLAgent` handles complex feature engineering, hyperparameter tuning, and advanced models.
    """
    dataset_description = dspy.InputField(desc="Schema of the DataFrame `df`.")
    user_goal = dspy.InputField(desc="The user's machine learning goal.")
    hint = dspy.InputField(desc="Recent interaction history.", default="")
    code = dspy.OutputField(desc="Python code for basic machine learning.")
    commentary = dspy.OutputField(desc="Explanation of the model, training process, and evaluation results.")
    kpis = dspy.OutputField(desc="Optional: Key ML performance metrics, e.g., accuracy, R-squared.", default="")

class BasicVisualizationAgent(dspy.Signature):
    """
    You are an AI agent that generates Python code for data visualizations using Plotly.
    Given a dataset (`df`), a user goal, and styling instructions, create appropriate basic charts.
    Capabilities:
    - Bar charts, line charts, scatter plots, histograms, pie charts.
    - Basic customization (titles, labels, colors).

    Use `st.plotly_chart(fig, use_container_width=True)` to display plots.
    The `AdvancedVisualizationAgent` handles complex dashboards and interactive plots.
    The `dataset_description` provides schema, `styling_instructions` provides Plotly styling hints.
    """
    dataset_description = dspy.InputField(desc="Schema of the DataFrame `df` and descriptions of columns.")
    user_goal = dspy.InputField(desc="User's goal for visualization (e.g., 'show sales trend', 'compare feature distributions').")
    styling_instructions = dspy.InputField(desc="Contextual information or style guidelines for Plotly charts.")
    hint = dspy.InputField(desc="Recent interaction history.", default="")
    code = dspy.OutputField(desc="Python code using Plotly to generate the visualization.")
    commentary = dspy.OutputField(desc="Explanation of the chart and what it represents.")
    kpis = dspy.OutputField(desc="Optional: Key insights derived from the visualization.", default="")


class StoryTellerAgent(dspy.Signature):
    """
    You are a data storyteller. Given a list of analyses performed by different AI agents (each with commentary and code/KPIs),
    compose a coherent and compelling narrative that summarizes the entire data analysis journey, key findings, and insights.
    The story should be easy to understand for a non-technical audience if specified by the goal.
    """
    analysis_summary_list = dspy.InputField(desc="A list of strings, where each string is a summary of an agent's action, its commentary, and key findings/KPIs.")
    user_goal = dspy.InputField(desc="The original user goal for the entire analysis.")
    target_audience = dspy.InputField(desc="Intended audience for the story (e.g., 'technical team', 'business stakeholders').", default="technical team")
    story = dspy.OutputField(desc="A comprehensive narrative summarizing the analysis.")

class CodeFixAgent(dspy.Signature):
    """
    You are an AI specializing in fixing faulty Python data analytics code.
    Given the faulty code, the error message, the dataset description, and the original goal,
    your task is to identify the error and provide the corrected code.
    Focus on fixing only the problematic parts while preserving the original intent.
    Ensure the corrected code is executable in a Streamlit environment (use `st.write`, `st.plotly_chart`).
    """
    faulty_code = dspy.InputField(desc="The Python code that produced an error.")
    error_message = dspy.InputField(desc="The error message generated by the faulty code.")
    dataset_description = dspy.InputField(desc="Schema of the DataFrame `df` used in the code.")
    user_goal = dspy.InputField(desc="The original goal the faulty code was trying to achieve.")
    hint = dspy.InputField(desc="Recent interaction history or previous code attempts.", default="")
    fixed_code = dspy.OutputField(desc="The corrected Python code.")
    explanation = dspy.OutputField(desc="A brief explanation of the fix.", default="")


# --- Agent Modules ---

AVAILABLE_CORE_AGENTS = {
    "PreprocessingAgent": PreprocessingAgent,
    "BasicStatisticalAgent": BasicStatisticalAgent,
    "BasicMLAgent": BasicMLAgent,
    "BasicVisualizationAgent": BasicVisualizationAgent,
}

# Attempt to register enhanced agents if they were imported successfully
REGISTERED_ENHANCED_AGENTS = {}
if EnhancedStatisticalAgent: REGISTERED_ENHANCED_AGENTS["EnhancedStatisticalAgent"] = EnhancedStatisticalAgent
if AdvancedMLAgent: REGISTERED_ENHANCED_AGENTS["AdvancedMLAgent"] = AdvancedMLAgent
if KPIGeneratorAgent: REGISTERED_ENHANCED_AGENTS["KPIGeneratorAgent"] = KPIGeneratorAgent
if DataQualityAgent: REGISTERED_ENHANCED_AGENTS["DataQualityAgent"] = DataQualityAgent
if AdvancedVisualizationAgent: REGISTERED_ENHANCED_AGENTS["AdvancedVisualizationAgent"] = AdvancedVisualizationAgent
if ExcelIntegrationAgent: REGISTERED_ENHANCED_AGENTS["ExcelIntegrationAgent"] = ExcelIntegrationAgent
# AgentCoordinator is handled separately as it's an orchestrator, not a typical "doer" agent in the planner's list.


class AutoAnalystIndividual(dspy.Module):
    """
    Handles direct calls to a specified agent.
    """
    def __init__(self, retrievers):
        super().__init__()
        self.retrievers = retrievers
        self.agents = {}
        self.agent_inputs = {} # To store input fields for each agent

        all_agent_signatures = {**AVAILABLE_CORE_AGENTS, **REGISTERED_ENHANCED_AGENTS}

        for name, sig_class in all_agent_signatures.items():
            self.agents[name] = dspy.ChainOfThought(sig_class)
            # Dynamically get input fields from the signature
            # Assuming signature fields are accessible via .signature or similar dspy mechanism
            # For simplicity, we'll manually list common ones or expect them if introspection is complex
            # This part might need adjustment based on DSPy's API for signature introspection
            self.agent_inputs[name] = list(sig_class.inputs().keys())


        self.memory_summarizer = dspy.ChainOfThought(memory_summarize_agent)
        self.code_fixer = dspy.ChainOfThought(CodeFixAgent)
        self.error_memory_agent = dspy.ChainOfThought(error_memory_agent)


    def forward(self, query, specified_agent_name, df_description, styling_desc=""):
        st.write(f"User explicitly chose **{specified_agent_name}** to answer the query: '{query}'")

        if specified_agent_name not in self.agents:
            st.error(f"Agent '{specified_agent_name}' not recognized.")
            return {"error": f"Agent '{specified_agent_name}' not recognized."}

        agent_to_call = self.agents[specified_agent_name]
        agent_input_fields = self.agent_inputs[specified_agent_name]

        # Prepare inputs for the specified agent
        inputs = {}
        if 'user_goal' in agent_input_fields: inputs['user_goal'] = query
        if 'dataset_description' in agent_input_fields: inputs['dataset_description'] = df_description
        if 'styling_instructions' in agent_input_fields: inputs['styling_instructions'] = styling_desc
        if 'hint' in agent_input_fields: inputs['hint'] = str(st.session_state.get("st_memory", [])[:5]) # Pass recent memory

        # For agents that might not have all these fields, dspy.ChainOfThought should ignore extra kwargs
        # or we can filter inputs based on agent_input_fields strictly
        strict_inputs = {k: v for k, v in inputs.items() if k in agent_input_fields}

        try:
            with st.spinner(f"**{specified_agent_name}** is working..."):
                response = agent_to_call(**strict_inputs)
        except Exception as e:
            st.error(f"Error during {specified_agent_name} execution: {str(e)}")
            traceback.print_exc()
            return {"error": str(e), "traceback": traceback.format_exc()}

        output_dict = {"agent_name": specified_agent_name}
        code_to_execute = None
        commentary = ""

        if hasattr(response, 'code'):
            code_to_execute = response.code
            st.code(code_to_execute, language='python')
            output_dict['code'] = code_to_execute
        if hasattr(response, 'commentary'):
            commentary = response.commentary
            st.markdown(f"**Commentary from {specified_agent_name}:**\n{commentary}")
            output_dict['commentary'] = commentary
        if hasattr(response, 'kpis') and response.kpis:
            st.markdown(f"**KPIs from {specified_agent_name}:**\n{response.kpis}")
            output_dict['kpis'] = response.kpis
        
        # Add other potential fields from response to output_dict
        for key, value in response.items():
            if key not in output_dict and key != 'rationale': # rationale is dspy internal
                output_dict[key] = value
                st.markdown(f"**{key.capitalize()}:**")
                st.write(value)


        # Execute code if generated
        if code_to_execute:
            st.markdown("--- \n**Execution Output:**")
            try:
                # Ensure 'df' is available in the execution scope if agents assume it
                # This requires df to be passed or globally available in Streamlit context.
                # For safety, it's better if agents generate code that loads/receives df.
                # Here, we assume 'df' is in st.session_state.df
                exec_globals = {'st': st, 'pd': pd, 'np': np, 'df': st.session_state.df}
                
                # Capture stdout
                old_stdout = sys.stdout
                redirected_output = io.StringIO()
                sys.stdout = redirected_output
                
                exec(code_to_execute, exec_globals)
                
                sys.stdout = old_stdout # Restore stdout
                execution_stdout = redirected_output.getvalue()
                if execution_stdout:
                    st.text_area("Captured Output:", value=execution_stdout, height=200)
                output_dict['execution_stdout'] = execution_stdout

            except Exception as e:
                error_trace = traceback.format_exc()
                st.error(f"Error executing the generated code:\n{error_trace}")
                output_dict['execution_error'] = str(e)
                output_dict['execution_traceback'] = error_trace

                # Attempt to fix the code
                st.info("Attempting to fix the code...")
                try:
                    fix_response = self.code_fixer(
                        faulty_code=code_to_execute,
                        error_message=str(e),
                        dataset_description=df_description,
                        user_goal=query,
                        hint=str(st.session_state.get("st_memory", [])[:3])
                    )
                    fixed_code = fix_response.fixed_code
                    st.markdown("**Attempted Fix:**")
                    st.code(fixed_code, language='python')
                    output_dict['fixed_code_attempt'] = fixed_code
                    if hasattr(fix_response, 'explanation') and fix_response.explanation:
                        st.markdown(f"**Fix Explanation:** {fix_response.explanation}")
                        output_dict['fix_explanation'] = fix_response.explanation
                    
                    # Optionally, try executing the fixed code (with caution)
                    # For now, just display it.

                    # Log error and fix attempt to error memory
                    error_summary = self.error_memory_agent(
                        original_query=query,
                        agent_name=specified_agent_name,
                        faulty_code=code_to_execute,
                        error_message=str(e),
                        fixed_code_attempt=fixed_code,
                        fix_explanation=fix_response.explanation if hasattr(fix_response, 'explanation') else ""
                    ).error_summary
                    st.session_state.st_memory.insert(0, f"ERROR_LOG ({specified_agent_name}): {error_summary}")


                except Exception as fix_e:
                    st.error(f"CodeFixAgent also failed: {fix_e}")
                    output_dict['fix_agent_error'] = str(fix_e)


        # Summarize interaction for memory
        memory_payload = f"Agent: {specified_agent_name}. Goal: {query}. Commentary: {commentary}."
        if 'kpis' in output_dict and output_dict['kpis']:
            memory_payload += f" KPIs: {output_dict['kpis']}."
        if 'execution_error' in output_dict:
            memory_payload += f" Execution Error: {output_dict['execution_error']}."

        summary = self.memory_summarizer(agent_response=memory_payload, user_goal=query).summary
        st.session_state.st_memory.insert(0, f"SUMMARY ({specified_agent_name}): {summary}")

        return output_dict


class AutoAnalyst(dspy.Module):
    """
    The main Auto-Analyst system using a planner and an agent coordinator.
    """
    def __init__(self, retrievers):
        super().__init__()
        self.retrievers = retrievers # For dataset schema and styling

        # --- Core Agents ---
        self.planner = dspy.ChainOfThought(AnalyticalPlanner)
        self.goal_refiner = dspy.ChainOfThought(GoalRefinerAgent)
        self.story_teller = dspy.ChainOfThought(StoryTellerAgent)
        self.memory_summarizer = dspy.ChainOfThought(memory_summarize_agent)
        self.code_fixer = dspy.ChainOfThought(CodeFixAgent) # For post-coordinator execution errors
        self.error_memory_agent = dspy.ChainOfThought(error_memory_agent)


        # --- "Doer" Agents (Basic and Enhanced) ---
        self.doer_agents = {} # Stores instantiated dspy.Module for each agent
        self.agent_input_fields = {} # Stores input fields for each agent to prepare calls

        all_agent_signatures = {
            **AVAILABLE_CORE_AGENTS,
            **REGISTERED_ENHANCED_AGENTS
        }
        if AgentCoordinator: # AgentCoordinator is special
            all_agent_signatures["AgentCoordinator"] = AgentCoordinator
        else:
            st.warning("AgentCoordinator is not available. Multi-agent orchestration will be limited.")


        for name, sig_class in all_agent_signatures.items():
            self.doer_agents[name] = dspy.ChainOfThought(sig_class)
            self.agent_input_fields[name] = list(sig_class.inputs().keys())

        # Agent descriptions for the planner
        self.agent_descriptions_for_planner = self._generate_agent_descriptions(all_agent_signatures)


    def _generate_agent_descriptions(self, agent_signatures_map):
        descriptions = []
        for name, sig_class in agent_signatures_map.items():
            # Extract description from the docstring of the signature class
            docstring = sig_class.__doc__
            # Simplified extraction: take the first paragraph of the docstring
            desc = docstring.strip().split('\n\n')[0] if docstring else "No description available."
            descriptions.append(f"- {name}: {desc}")
        return "\n".join(descriptions)

    def forward(self, query, df_description, styling_desc=""):
        st.markdown(f"**Received Query:** {query}")
        
        # --- Plan Generation ---
        progress_bar = st.progress(0, text="🤖 Thinking about a plan...")
        
        planner_response = self.planner(
            dataset_description=df_description,
            agent_descriptions=self.agent_descriptions_for_planner,
            user_goal=query
        )
        plan_str = planner_response.plan.replace("Plan:", "").strip()
        plan_desc = planner_response.plan_desc
        
        st.markdown("### Execution Plan:")
        st.markdown(f"**Plan:** `{plan_str}`")
        st.markdown(f"**Description:** {plan_desc}")
        
        if not plan_str or "ask for clarification" in plan_desc.lower():
            st.warning("Planner suggests clarification is needed or could not form a plan.")
            progress_bar.progress(100, text="Clarification needed.")
            return {"status": "clarification_needed", "message": plan_desc}

        planned_agent_names = [name.strip() for name in plan_str.split("->") if name.strip()]
        if not planned_agent_names:
            st.error("Planner returned an empty plan.")
            progress_bar.progress(100, text="Planning failed.")
            return {"status": "error", "message": "Planner returned an empty plan."}

        progress_bar.progress(10, text="Plan received. Initializing Agent Coordinator...")

        # --- Orchestration by AgentCoordinator ---
        if "AgentCoordinator" not in self.doer_agents or not AgentCoordinator:
            st.error("AgentCoordinator is not available. Cannot execute multi-agent plan.")
            return {"status": "error", "message": "AgentCoordinator not available."}

        coordinator = self.doer_agents["AgentCoordinator"]
        coordinator_input_fields = self.agent_input_fields["AgentCoordinator"]
        
        # Prepare inputs for the coordinator
        coordinator_inputs = {}
        if 'user_goal' in coordinator_input_fields: coordinator_inputs['user_goal'] = query
        if 'dataset_description' in coordinator_input_fields: coordinator_inputs['dataset_description'] = df_description
        if 'planned_agent_names' in coordinator_input_fields: coordinator_inputs['planned_agent_names'] = planned_agent_names
        if 'agent_descriptions' in coordinator_input_fields: coordinator_inputs['agent_descriptions'] = self.agent_descriptions_for_planner # Coordinator might need this
        if 'styling_instructions' in coordinator_input_fields: coordinator_inputs['styling_instructions'] = styling_desc
        if 'hint' in coordinator_input_fields: coordinator_inputs['hint'] = str(st.session_state.get("st_memory", [])[:5])
        
        # Other inputs the coordinator might need, e.g., access to doer_agents or their signatures
        # This part depends on how AgentCoordinator is designed to receive agent definitions.
        # For now, let's assume AgentCoordinator can internally access/instantiate required agents based on names.
        # Alternatively, we could pass the self.doer_agents dictionary or relevant parts.
        if 'doer_agents_modules' in coordinator_input_fields:
             # Pass instantiated agent modules if coordinator expects them
            coordinator_inputs['doer_agents_modules'] = {
                name: self.doer_agents[name] for name in planned_agent_names if name in self.doer_agents
            }
        if 'doer_agents_input_fields' in coordinator_input_fields:
            coordinator_inputs['doer_agents_input_fields'] = {
                name: self.agent_input_fields[name] for name in planned_agent_names if name in self.agent_input_fields
            }


        strict_coordinator_inputs = {k: v for k, v in coordinator_inputs.items() if k in coordinator_input_fields}

        st.markdown("--- \n**🚀 Handing off to Agent Coordinator...**")
        final_output_code = None
        final_commentary = ""
        final_kpis = ""
        all_agent_outputs_summary = [] # For story teller

        try:
            with st.spinner("Agent Coordinator is orchestrating the plan..."):
                # The AgentCoordinator will internally call the planned agents
                # and manage the flow of data (df, context) between them.
                # Its output should be the final combined code, commentary, and KPIs.
                coordinator_response = coordinator(**strict_coordinator_inputs)
                progress_bar.progress(80, text="Coordinator finished. Compiling final output...")

            if hasattr(coordinator_response, 'final_code'):
                final_output_code = coordinator_response.final_code
                st.markdown("### Final Combined Code:")
                st.code(final_output_code, language='python')
            if hasattr(coordinator_response, 'final_commentary'):
                final_commentary = coordinator_response.final_commentary
                st.markdown("### Final Combined Commentary:")
                st.markdown(final_commentary)
            if hasattr(coordinator_response, 'final_kpis') and coordinator_response.final_kpis:
                final_kpis = coordinator_response.final_kpis
                st.markdown("### Final Combined KPIs:")
                st.markdown(final_kpis) # Assuming KPIs are markdown-friendly strings
            
            # Collect summaries from coordinator if provided (e.g., list of individual agent actions)
            if hasattr(coordinator_response, 'agent_execution_summaries'):
                all_agent_outputs_summary = coordinator_response.agent_execution_summaries
                st.markdown("### Agent Execution Log (from Coordinator):")
                for summary_item in all_agent_outputs_summary:
                    st.info(f"- {summary_item}")


        except Exception as e:
            error_trace = traceback.format_exc()
            st.error(f"Error during Agent Coordinator execution: {str(e)}")
            st.code(error_trace)
            progress_bar.progress(100, text="Error during coordination.")
            # Log error to memory
            error_summary = self.error_memory_agent(
                original_query=query,
                agent_name="AgentCoordinator",
                faulty_code="N/A (Coordination Error)",
                error_message=str(e),
                fixed_code_attempt="N/A",
                fix_explanation=error_trace
            ).error_summary
            st.session_state.st_memory.insert(0, f"ERROR_LOG (AgentCoordinator): {error_summary}")
            return {"status": "error", "message": f"Coordinator error: {str(e)}", "traceback": error_trace}

        # --- Execute Final Code ---
        if final_output_code:
            st.markdown("--- \n**⚙️ Executing Final Combined Code:**")
            try:
                exec_globals = {'st': st, 'pd': pd, 'np': np, 'df': st.session_state.df}
                
                # Capture stdout for execution
                with stdout_capture() as captured_output:
                    exec(final_output_code, exec_globals)
                
                execution_stdout = captured_output.getvalue()
                if execution_stdout:
                    st.text_area("Captured Execution Output:", value=execution_stdout, height=300)
                
                progress_bar.progress(90, text="Final code executed.")

            except Exception as e:
                error_trace = traceback.format_exc()
                st.error(f"Error executing the final combined code:\n{error_trace}")
                progress_bar.progress(100, text="Error in final execution.")
                
                # Attempt to fix the final code
                st.info("Attempting to fix the final code...")
                try:
                    fix_response = self.code_fixer(
                        faulty_code=final_output_code,
                        error_message=str(e),
                        dataset_description=df_description,
                        user_goal=query,
                        hint=str(st.session_state.get("st_memory", [])[:3]) + f"\nCoordinator Commentary: {final_commentary}"
                    )
                    fixed_code = fix_response.fixed_code
                    st.markdown("**Attempted Fix for Final Code:**")
                    st.code(fixed_code, language='python')
                    if hasattr(fix_response, 'explanation') and fix_response.explanation:
                        st.markdown(f"**Fix Explanation:** {fix_response.explanation}")
                    
                    # Log error and fix attempt
                    error_summary = self.error_memory_agent(
                        original_query=query,
                        agent_name="FinalExecution",
                        faulty_code=final_output_code,
                        error_message=str(e),
                        fixed_code_attempt=fixed_code,
                        fix_explanation=fix_response.explanation if hasattr(fix_response, 'explanation') else ""
                    ).error_summary
                    st.session_state.st_memory.insert(0, f"ERROR_LOG (FinalExecution): {error_summary}")

                except Exception as fix_e:
                    st.error(f"CodeFixAgent also failed for the final code: {fix_e}")

                return {"status": "error", "message": f"Final execution error: {str(e)}", "traceback": error_trace, "final_code": final_output_code}
        else:
            st.info("No final code was generated by the Agent Coordinator.")
            progress_bar.progress(90, text="No final code to execute.")


        # --- Story Telling (Optional) ---
        if st.checkbox("Generate a story summary of the analysis?", value=False):
            if all_agent_outputs_summary:
                story_payload = "\n".join(all_agent_outputs_summary)
            else: # Fallback if coordinator didn't provide detailed summaries
                story_payload = f"Plan: {plan_str}. Overall Goal: {query}. Final Commentary: {final_commentary}. KPIs: {final_kpis}"

            target_audience = st.selectbox("Select target audience for the story:", ["Technical Team", "Business Stakeholders"], index=0)
            story_response = self.story_teller(
                analysis_summary_list=[story_payload], # StoryTeller expects a list
                user_goal=query,
                target_audience=target_audience.lower().replace(" ", "_")
            )
            st.markdown("### 📖 Analysis Story:")
            st.markdown(story_response.story)


        # --- Final Memory Update ---
        memory_payload = f"Overall Goal: {query}. Plan: {plan_str}. Final Commentary: {final_commentary}."
        if final_kpis:
            memory_payload += f" Final KPIs: {final_kpis}."
        
        summary = self.memory_summarizer(agent_response=memory_payload, user_goal=query).summary
        st.session_state.st_memory.insert(0, f"OVERALL_SUMMARY: {summary}")
        
        progress_bar.progress(100, text="Analysis complete!")
        
        return {
            "status": "success",
            "plan": plan_str,
            "plan_description": plan_desc,
            "final_code": final_output_code,
            "final_commentary": final_commentary,
            "final_kpis": final_kpis,
            "execution_stdout": execution_stdout if 'execution_stdout' in locals() else None
        }

# Example of how to get agent descriptions for the planner (can be part of AutoAnalyst init)
# def get_all_agent_descriptions():
#     # This function would dynamically generate descriptions for all available agents
#     # including core, enhanced, and coordinator.
#     # For now, it's manually handled in AnalyticalPlanner prompt or AutoAnalyst._generate_agent_descriptions
#     pass
