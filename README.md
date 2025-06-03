# Auto-Analyst 2.0  
AI-driven multi-agent analytics platform

![Auto-Analyst banner](images/Auto-Analyst%20Banner.png)

Auto-Analyst turns raw CSV/Excel data into executive-ready insights with **one conversation**.  
Version 2.0 introduces a brand-new _enhanced agent system_, deep KPI generation, automated data-quality profiling and a cleaned, production-ready codebase.

---

## ✨ What’s new in 2.0
| Area | Upgrade |
|------|---------|
| 🧠 Agent Engine | 7 advanced agents (Statistical, ML, KPI, Quality, Viz, Excel, Coordinator) built on **DSPy**. |
| 📊 Analytics Depth | ARIMA/SARIMAX, seasonal decomposition, hypothesis tests, automatic feature engineering, hyper-parameter tuning, DBSCAN outlier detection. |
| 📈 Visualisations | Interactive Plotly dashboards, correlation networks, Pareto & KPI cards, PCA plots for clustering. |
| 📋 KPI Generator | Auto-detects finance/sales/operational columns → outputs ready-to-use metric cards (Total, CAGR, ROAS, etc.). |
| 🧹 Data Quality | Full profiling (missing-values heatmaps, duplicate scan, cardinality, VIF, normality, quality score). |
| 📂 Excel Integration | Multi-sheet overview, relationship inference, merge suggestions. |
| 🗂️ Clean Repo | React prototype removed, logic moved to `src/`, docs to `docs/`. Streamlit + Flask remain. |

---

## 🔧 Installation

### 1. Clone & set up
```bash
git clone https://github.com/<your-org>/auto-analyst.git
cd auto-analyst
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables
Create `.env`:
```
OPENAI_API_KEY=sk-********************************
# optional
GROQ_API_KEY=your_groq_key           # if you switch to Llama-3 via Groq
DSPY_LLM_MODEL=gpt-4o-mini           # or gpt-4o, gpt-4-turbo, llama3-70b
```

---

## 🚀 Quick start

### Streamlit workstation
```bash
streamlit run enhanced_streamlit_frontend.py
```
Upload a CSV or Excel, then chat:

```
Show sales trend and forecast next 6 months
```
or call an agent directly:

```
@DataQualityAgent profile the dataset
```

### REST API (production)
```bash
export OPENAI_API_KEY=...
gunicorn -b 0.0.0.0:8000 flask_app.flask_app:app
```
```
POST /chat
{
  "query": "Build a churn prediction model"
}
```

---

## 🏗️ Architecture

```
┌────────────┐  HTTP / Websocket  ┌─────────────────────┐
│  Frontend  │ ─────────────────▶ │  Flask REST API     │
│ • Streamlit│                   │  (routes.py)        │
└────────────┘                   └────────┬────────────┘
                                          │
                     planner / coordinator│
                                          ▼
                              ┌─────────────────────────┐
                              │  Agent Engine (DSPy)    │
                              │  • Planner             │
                              │  • AgentCoordinator    │
                              │  • 11 specialised agents│
                              └────────┬────────────────┘
                        context / embeddings│
                                          ▼
                         ┌────────────────────────┐
                         │  Llama-Index Retrievers │
                         │  • Data schema index   │
                         │  • Plot styling index  │
                         └────────────────────────┘
```

### Key folders
```
src/
 ├─ agents.py                core & planner
 ├─ enhanced_agents/         advanced capabilities
 ├─ memory_agents.py         summaries & error logs
 ├─ retrievers.py            schema + style indices
flask_app/                   production API
new_frontend.py              legacy Streamlit UI
enhanced_streamlit_frontend.py (new UI)
docs/                        design docs & HOW-TOs
```

---

## 🤖 Agent roster

| Agent | Purpose |
|-------|---------|
| **DataQualityAgent** | Profiling, cleaning recommendations, quality score |
| **EnhancedStatisticalAgent** | Time-series, hypothesis tests, correlation network, anomaly detection |
| **AdvancedMLAgent** | Auto feature engineering, model selection, CV & tuning |
| **KPIGeneratorAgent** | Finance, sales & ops KPI cards + dashboards |
| **AdvancedVisualizationAgent** | Multi-dimensional Plotly & dashboards |
| **ExcelIntegrationAgent** | Multi-sheet merge, relationship mapping |
| **AgentCoordinator** | Executes planner plan, merges code, resolves dependencies |
| Core agents | Preprocessing, BasicStats, BasicML, BasicViz |

Agents share short-term memory and are orchestrated via **Planner-Doer** pattern.

---

## 🛠️ Customization

* Switch LLM: set `DSPY_LLM_MODEL` env or edit `enhanced_streamlit_frontend.py`.
* Extend agents: add a new `Signature` under `src/enhanced_agents/`, then import in `__init__.py`.
* Change vector DB: adjust `retrievers.py` to use your preferred store.

---

## 🧪 Testing
```bash
pytest tests/          # unit tests for agents & retrievers
```
CI pipeline (GitHub Actions) runs: lint ➜ unit ➜ integration (Streamlit & API).

---

## 🤝 Contributing

1. Fork → new branch → PR  
2. Follow Black / Ruff formatting (`ruff check .`, `ruff format .`)  
3. Add/extend unit tests for new code  
4. Describe enhancement in PR description

---

## 📄 License
MIT © 2025 Your Company
