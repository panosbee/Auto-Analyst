import dspy

class EnhancedMemorySummarizeAgent(dspy.Signature):
    """
    You are an AI assistant responsible for creating concise and structured summaries of interactions
    between a user and various data analysis AI agents. Your goal is to capture the essence of
    an agent's response in relation to the user's goal, creating a memory item that will be useful
    for providing context to subsequent agents in a multi-step analysis.

    The summary should be optimized for contextual understanding by other AI agents.
    It should highlight:
    1.  The specific task/question the agent addressed.
    2.  The main action(s) taken by the agent (e.g., "performed time series analysis", "generated a scatter plot", "cleaned missing data").
    3.  Key findings, results, or outputs (e.g., "identified a positive correlation of 0.75", "forecasted sales increase by 10%", "imputed 5% missing values in 'age' column").
    4.  Any important parameters, decisions, or assumptions made by the agent.
    5.  If the agent produced code, mention the primary purpose of the code.
    6.  If KPIs were generated, list the most important ones.

    Keep the summary factual, concise, and directly relevant to the analysis progression.
    Avoid conversational fluff. Structure it clearly, perhaps using bullet points or key-value pairs if appropriate for clarity.
    """
    agent_response = dspy.InputField(desc="A detailed string containing the agent's name, the code it generated (if any), its commentary, and any KPIs or specific results it produced.")
    user_goal = dspy.InputField(desc="The original user-defined goal for the current phase of analysis or the overall task.")
    summary = dspy.OutputField(desc="A structured and concise summary of the agent's interaction, optimized for providing context to other AI agents.")

class EnhancedErrorMemoryAgent(dspy.Signature):
    """
    You are an AI assistant that specializes in logging and summarizing errors encountered during
    AI-driven data analysis tasks. Your purpose is to create a detailed and structured error report
    that can be used for debugging, identifying patterns in errors, and improving agent performance over time.

    The error summary should capture:
    1.  User's Original Query/Goal: What the user was trying to achieve.
    2.  Agent Involved: Which AI agent was executing when the error occurred.
    3.  Problematic Code: The snippet of code that caused the error.
    4.  Error Message: The exact error message produced.
    5.  Error Type: (e.g., SyntaxError, ValueError, TypeError, APIError, LogicalError - infer if possible).
    6.  Context of Error: Briefly describe what the code was attempting to do (e.g., "accessing a non-existent column", "dividing by zero", "incorrect API call parameters").
    7.  Attempted Fix (if any): The code snippet that was proposed as a fix.
    8.  Explanation of Fix (if any): Why the fix was proposed or what it aimed to correct.
    9.  Outcome of Fix (if known/applicable): Was the fix successful, partially successful, or did it fail? (This might be added in a subsequent step).
    10. Key Data Aspects: Mention any specific data columns, types, or values that were central to the error, if discernible.

    Structure the output clearly for easy parsing and analysis later.
    This information will contribute to a knowledge base of errors and their resolutions.
    """
    original_query = dspy.InputField(desc="The initial user query or goal that led to the error.")
    agent_name = dspy.InputField(desc="The name of the AI agent that was running or generated the faulty code.")
    faulty_code = dspy.InputField(desc="The piece of code that resulted in an error.")
    error_message = dspy.InputField(desc="The specific error message thrown by the interpreter or system.")
    data_context = dspy.InputField(desc="Brief description of the dataset schema or relevant data parts involved in the error, if known.", default="No specific data context provided.")
    fixed_code_attempt = dspy.InputField(desc="The code suggested as a fix for the error. Can be empty if no fix was attempted.", default="No fix attempted.")
    fix_explanation = dspy.InputField(desc="The reasoning behind the attempted fix. Can be empty.", default="No explanation for fix provided.")
    error_summary = dspy.OutputField(desc="A structured, detailed summary of the error, its context, and any resolution attempts, for logging and future learning.")

# For backward compatibility or simpler use cases, you might keep the original signatures
# or alias the new ones if the old names are hardcoded elsewhere.
# For this enhancement, we are replacing them with the new, more capable versions.

memory_summarize_agent = EnhancedMemorySummarizeAgent
error_memory_agent = EnhancedErrorMemoryAgent

# Example usage (conceptual, not executed here):
# dspy.ChainOfThought(EnhancedMemorySummarizeAgent)(agent_response="...", user_goal="...")
# dspy.ChainOfThought(EnhancedErrorMemoryAgent)(original_query="...", agent_name="...", ...)
