# InsightGPT

InsightGPT is a local, AI-powered Business Analyst Agent that automates EDA, KPI extraction, charting, and smart insights from CSV, Excel, or JSON files — all with zero coding. Built with open-source LLMs like Phi-2 and TinyLlama.

## 🔍 What It Does

- 📊 Performs Exploratory Data Analysis (EDA)
- 🧮 Extracts KPIs and key statistics
- 📈 Auto-generates charts and visualizations
- 🤖 Generates natural language insights using LLMs
- 💬 Answers business questions from your data (RAG-powered)
- 🧠 Remembers feedback and session goals
- 📤 Exports full reports to PDF and HTML
- 🛡️ Runs fully offline with local models (Phi-2)

---

## ⚙️ Tech Stack

| Layer         | Tools Used |
|---------------|------------|
| Language      | Python |
| LLMs          | HuggingFace Transformers, Phi-2, TinyLlama |
| Frontend      | Streamlit |
| Charts        | Matplotlib |
| EDA / KPIs    | pandas, NumPy |
| Export        | reportlab, HTML |
| Others        | Docker, GitHub Actions (CI/CD), .env security |

---

## 📦 Features

- ✅ Cloud vs. Local model toggle
- ✅ Memory recall and feedback loops
- ✅ Smart domain detection (Retail, Healthcare, Finance, etc.)
- ✅ Modular architecture (easy to extend)
- ✅ Clean, professional UI
- ✅ Built for demos, resumes, and real-world impact

---

## 📸 Screenshots

Coming soon! (Live demo GIF + chart samples)

---

## 🚀 How to Run

```bash
git clone https://github.com/ayaansd/InsightGPT.git
cd InsightGPT
pip install -r requirements.txt
streamlit run app.py
