# Enhanced Auto-Analyst: Complete Setup & Implementation Plan

This document provides a comprehensive guide to setting up, running, and understanding the complete Enhanced Auto-Analyst system. This system incorporates advanced AI agents, a refined project structure, and sophisticated data analysis capabilities.

## 0. Prerequisites

Before you begin, ensure you have the following installed:
*   Python 3.9+
*   `pip` (Python package installer)
*   `git` (for version control, if cloning from a repository)
*   An OpenAI API Key (and optionally a Groq API Key if you plan to use Llama3 via Groq).

## 1. Obtaining the Application Files

You should have the `AutoAnalyst_Enhanced_Package.zip` file (or access to the repository if it's hosted).

*   **If you have the ZIP file**: Extract it to a desired location on your local machine. This will create the `auto-analyst/` project directory.
*   **If cloning from a repository (example)**:
    ```bash
    git clone https://github.com/your-org/auto-analyst.git
    cd auto-analyst
    ```

This guide assumes you are now in the root `auto-analyst/` directory.

## 2. Setting Up the Python Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Navigate to the project root directory (e.g., auto-analyst/)
cd path/to/your/auto-analyst

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS and Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install the required Python packages
pip install -r requirements.txt
```

This command will install all necessary libraries, including `dspy-ai`, `streamlit`, `flask`, `pandas`, `llama-index`, `openai`, etc.

## 3. Configuring Environment Variables

The application requires API keys and other configurations, which are managed through an `.env` file.

1.  In the project root directory, find the `.env.example` file.
2.  Copy it to a new file named `.env`:
    ```bash
    cp .env.example .env
    ```
3.  Open the `.env` file in a text editor and fill in your actual API keys and desired configurations:

    ```env
    # --- Core AI Service Keys ---
    OPENAI_API_KEY="sk-YourActualOpenAIAPIKeyHere" # REQUIRED

    # Optional: If you want to use Llama3 or other models via Groq
    # GROQ_API_KEY="gsk_YourActualGroqAPIKeyHere"

    # --- DSPy LLM Configuration ---
    # Default is 'gpt-4o-mini' if not set, but you can override
    DSPY_LLM_MODEL="gpt-4o-mini" # Or "gpt-4o", "gpt-4-turbo", "llama3-70b-8192" (for Groq)
    DSPY_LLM_MAX_TOKENS="4096"   # Adjust based on model and complexity

    # --- Flask Application Settings (for flask_app) ---
    FLASK_ENV="development" # "production" for Gunicorn deployment
    # DATABASE_URL="sqlite:///response.db" # Default is used if not set
    ```

**Important**: Ensure your `OPENAI_API_KEY` is correctly set. The application will not function without it.

## 4. Understanding the Project Structure

The enhanced Auto-Analyst has a refined project structure:

```
auto-analyst/
├── .env                   # Your local environment variables (created from .env.example)
├── .env.example           # Example environment configuration
├── .gitignore             # Specifies intentionally untracked files that Git should ignore
├── README.md              # Overview of the project
├── requirements.txt       # Python package dependencies
├── enhanced_streamlit_frontend.py # Main Streamlit application file
├── download_package.py    # Utility script to package the application (developer tool)
│
├── src/                   # Core Python source code for the application
│   ├── __init__.py
│   ├── agents.py          # Core agent signatures, planner, main AutoAnalyst modules
│   ├── memory_agents.py   # Agents for managing interaction and error memory
│   ├── retrievers.py      # Data schema generation and LlamaIndex retriever setup
│   └── enhanced_agents/   # Package for advanced, specialized AI agents
│       ├── __init__.py
│       ├── statistical.py
│       ├── ml.py
│       ├── kpi.py
│       ├── quality.py
│       ├── visualization.py
│       ├── excel_integration.py
│       └── coordinator.py # Agent that orchestrates multi-agent plans
│
├── flask_app/             # Flask backend API
│   ├── __init__.py
│   ├── flask_app.py       # Main Flask application instance setup
│   ├── routes.py          # API endpoint definitions
│   └── db_models.py       # SQLAlchemy database models (for logging, etc.)
│
├── data/                  # Sample data files
│   └── Housing.csv
│
├── images/                # Static images for the UI
│   ├── Auto-analysts icon small.png
│   └── Auto-Analyst Banner.png
│
└── docs/                  # Documentation files
    ├── architecture.md
    └── agents.md
    └── (this file: complete_implementation_plan.md)
```

## 5. Running the Streamlit Application (Primary UI)

The Streamlit interface is the primary way to interact with Auto-Analyst for data analysis.

1.  Ensure your virtual environment is activated.
2.  Ensure your `.env` file is correctly configured with at least `OPENAI_API_KEY`.
3.  From the project root directory (`auto-analyst/`), run:

    ```bash
    streamlit run enhanced_streamlit_frontend.py
    ```

4.  This will start the Streamlit development server, and your web browser should automatically open to the application's URL (usually `http://localhost:8501`).
5.  **Using the App**:
    *   Upload a CSV or Excel file using the sidebar.
    *   If it's an Excel file with multiple sheets, select the desired sheet.
    *   Optionally, provide a dataset name and description.
    *   Click "Load Data & Initialize Agents".
    *   Once loaded, you can interact with Auto-Analyst via the chat interface.
        *   Ask general analysis questions (e.g., "What are the key trends in this data?").
        *   Request specific analyses (e.g., "Perform time series analysis on the 'sales' column and forecast for 6 months.").
        *   Call specific agents directly (e.g., `@DataQualityAgent profile this data`, `@KPIGeneratorAgent find financial KPIs`).

## 6. Running the Flask API (Optional / For Production or Programmatic Access)

The Flask API provides backend endpoints for Auto-Analyst functionalities.

### Development Mode:

1.  Ensure your virtual environment is activated and `.env` is configured.
2.  From the project root directory, you can run the Flask development server:
    ```bash
    # Ensure FLASK_APP is implicitly set by directory structure or explicitly:
    # export FLASK_APP=flask_app.flask_app (macOS/Linux)
    # set FLASK_APP=flask_app.flask_app (Windows)
    # Then:
    flask run
    # Or, more directly if flask_app.py has app.run():
    # python -m flask_app.flask_app
    ```
    The API will typically be available at `http://localhost:5000`.

### Production Mode (using Gunicorn):

Gunicorn is a robust WSGI HTTP server for running Python web applications in production.

1.  Ensure Gunicorn is installed (it's in `requirements.txt`).
2.  From the project root directory:
    ```bash
    gunicorn "flask_app.flask_app:app" -b 0.0.0.0:8000
    ```
    *   `"flask_app.flask_app:app"` tells Gunicorn to look for the `app` instance in `flask_app/flask_app.py`.
    *   `-b 0.0.0.0:8000` binds the server to all network interfaces on port 8000.

    The API will be available at `http://your_server_ip:8000`.

**API Endpoints**: Refer to `flask_app/routes.py` for defined endpoints (e.g., `/api/chat`, `/api/upload_data`).

## 7. Key System Components and Functionality

*   **Agent System (`src/`)**:
    *   **Planner (`AnalyticalPlanner` in `src/agents.py`)**: Devises a sequence of agent tasks based on your goal.
    *   **Coordinator (`AgentCoordinator` in `src/enhanced_agents/coordinator.py`)**: Executes the plan, managing data flow and combining results from various "doer" agents.
    *   **Doer Agents (in `src/agents.py` and `src/enhanced_agents/`)**:
        *   `DataQualityAgent`: Profiles data, identifies issues, suggests cleaning.
        *   `EnhancedStatisticalAgent`: Advanced statistics, time series (ARIMA), hypothesis tests.
        *   `AdvancedMLAgent`: Automated feature engineering, model selection, tuning.
        *   `KPIGeneratorAgent`: Identifies and visualizes KPIs for finance, sales, operations.
        *   `AdvancedVisualizationAgent`: Creates complex, interactive Plotly dashboards.
        *   `ExcelIntegrationAgent`: Handles multi-sheet Excel data, merging and aggregation.
        *   And more (basic agents for preprocessing, stats, ML, viz).
*   **Data Handling (`src/retrievers.py`)**:
    *   `make_data()`: Parses uploaded files (CSV/Excel) and generates a detailed schema description. For Excel, it provides an overview of all sheets and details for the selected one.
    *   `initiatlize_retrievers()`: Uses LlamaIndex to create semantic search capabilities over the data schema and Plotly styling instructions, providing relevant context to agents.
*   **Memory (`src/memory_agents.py` and Streamlit session state)**:
    *   Short-term memory of interactions and errors is maintained to provide context for ongoing analysis.

## 8. Customization and Further Development

*   **Changing LLMs**:
    *   The default LLM is configured in `enhanced_streamlit_frontend.py` and `flask_app/flask_app.py` (via `dspy.configure`).
    *   You can change the `DSPY_LLM_MODEL` and `DSPY_LLM_MAX_TOKENS` in your `.env` file.
    *   To use different LLM providers (e.g., Groq, Anthropic), modify the `dspy.configure` call and ensure necessary API keys are in `.env`.
*   **Adding New Agents**:
    1.  Define a new agent `Signature` in a Python file (e.g., within `src/enhanced_agents/`).
    2.  Implement the agent's logic (typically as static methods within the signature class that generate Python code strings, or as a `dspy.Module`).
    3.  Import and register your new agent in `src/enhanced_agents/__init__.py` and ensure it's available to `src/agents.py` (e.g., by adding to `REGISTERED_ENHANCED_AGENTS`).
    4.  Update the `AnalyticalPlanner` prompt in `src/agents.py` to include a description of your new agent's capabilities so it can be included in plans.
*   **Modifying Prompts**: Agent behavior is heavily influenced by their prompts (docstrings in `dspy.Signature` classes). Edit these to tune performance or capabilities.
*   **Database for Flask App**: The Flask app uses SQLite by default (`response.db`). For production, you might want to configure a more robust database (e.g., PostgreSQL) by setting the `DATABASE_URL` in your `.env` file and ensuring the appropriate database driver is installed.

## 9. Testing (Conceptual)

A full testing suite would involve:
*   **Unit Tests (`pytest`)**: For individual functions in `retrievers.py`, helper utilities, and potentially for checking the structure of agent-generated code (e.g., does it import necessary libraries?).
*   **Integration Tests**: Testing the flow through the `AutoAnalyst` and `AutoAnalystIndividual` modules, ensuring agents are called correctly and outputs are combined.
*   **API Tests (for Flask)**: Using tools like `pytest-flask` or `httpx` to test API endpoints.
*   **UI Tests (for Streamlit)**: Using `streamlit-testing-library` or Selenium for end-to-end UI interaction tests.

For now, manual testing by interacting with the Streamlit UI and providing various queries and datasets is the primary way to verify functionality.

## 10. Troubleshooting Common Issues

*   **`OPENAI_API_KEY not found`**: Ensure your `.env` file exists in the project root, is correctly named, and contains a valid `OPENAI_API_KEY`. Also, ensure `load_dotenv()` is called early in your application scripts.
*   **Module Not Found Errors**:
    *   Make sure your virtual environment is activated.
    *   Ensure all dependencies from `requirements.txt` are installed.
    *   If you've modified the project structure, check that Python's import paths (`sys.path`) can find your `src/` directory. Running scripts from the project root usually handles this.
*   **DSPy Configuration Errors**: Double-check the LLM model name and API keys. Some models might require specific DSPy versions or additional packages.
*   **LlamaIndex Embedding Errors**: Ensure `Settings.embed_model` is configured correctly (default uses OpenAI). If using other embedding models, configure them as per LlamaIndex documentation.
*   **File Upload Issues (Streamlit)**: Check file size limits (Streamlit default is 200MB, configurable) and supported file types.
*   **Agent Not Generating Expected Code/Commentary**:
    *   Review the agent's signature (prompt) in its Python file. Clarity and detail in the prompt are crucial.
    *   Consider using a more powerful LLM model (e.g., `gpt-4o` instead of `gpt-4o-mini`) via the `.env` file, as this can significantly impact the quality of generated outputs for complex tasks.
    *   Check the session memory in the Streamlit sidebar to see what context the agents are receiving.

This plan should guide you through setting up and running your Enhanced Auto-Analyst system. Happy analyzing!
