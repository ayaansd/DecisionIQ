DecisionIQ — Analyst Agent (Full-Stack)

📊 DecisionIQ is a full-stack Analyst Agent that helps users explore their data, generate insights, and automate business analysis workflows.

It combines a FastAPI backend with a React + Vite (MUI) frontend, integrates FAISS retrieval + LLMs, and supports both local models (Phi-2) and cloud LLMs (Llama-3.x via Hugging Face router).

🚩 What It Solves

❌ Problem: Business users spend hours running ad-hoc analysis, building repetitive KPI dashboards, and manually summarizing CSV/XLSX data.

✅ Solution: A one-stop AI analyst that:

Uploads datasets (CSV/XLSX/JSON).

Runs EDA (descriptive stats, correlations, distributions).

Extracts KPIs automatically.

Generates charts and exports summaries.

Supports Q&A over your data via vector search + LLM.

Allows iterative refinement with feedback loops.

🌐 API Endpoints

Key backend endpoints from fastapp.py:

GET /health → service health

POST /api/upload → dataset upload (CSV/XLSX/JSON)

POST /api/eda → exploratory analysis

POST /api/kpis → KPI extraction

POST /api/charts → chart generation (PNG)

POST /api/summary → auto-summary of data

POST /api/regenerate_summary → regenerate w/ feedback

POST /api/qa → Q&A over data (RAG + LLM)

POST /api/goal → analysis planning

POST /api/alerts → alert summarization

GET /api/sessions → session list

POST /api/log_feedback → log user feedback

POST /api/slack/send_summary → send Slack message

POST /api/auto_analysis/start + GET /api/auto_analysis/status → async jobs

🔧 Quickstart
1. Backend
cd insight
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn fastapp:app --reload --port 8000

2. Frontend
cd decisioniq_frontend
npm install
npm run dev



Dashboard Features (Frontend)

File upload UI (CSV/XLSX/JSON).

EDA panel: summary statistics + distributions.

KPI cards: auto-extracted insights.

Charts: bar, line, scatter, histograms.

Q&A box: natural language queries, LLM-powered answers.

Export: summaries → PDF/HTML.

⚙️ Configuration

Copy .env.example → .env:

HF_TOKEN=hf_xxx                   # Hugging Face token
CLOUD_MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct:together
LOCAL_MODEL_NAME=microsoft/phi-2
ALLOWED_ORIGINS=http://localhost:5173
FAISS_DIR=./faiss_index

🗺️ Roadmap

Streaming endpoints for long analyses.

Multi-tenant auth & session storage.

MongoDB Atlas logging of runs.

Deployment via Docker + Render/Cloud.

Extended report generator.

🛠️ Architecture & Flow
<img width="378" height="280" alt="image" src="https://github.com/user-attachments/assets/7965ca5f-0cc9-4aa2-8119-f95ba433e083" />


📜 License

MIT — free for use and adaptation.
Open http://localhost:5173
.

Backend runs at http://localhost:8000


