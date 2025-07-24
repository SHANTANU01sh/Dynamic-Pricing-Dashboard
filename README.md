# Dynamic-Pricing-Dashboard
A professional dynamic pricing dashboard that predicts product prices, tracks ML experiments with MLflow, manages data pipelines with Prefect, and features an AI chatbot powered by Groq &amp; LangChain. Built with Python, Streamlit, and a clean animated UI for interactive analytics.
📊 Dynamic Pricing Dashboard
This project is a machine learning–powered dynamic pricing web application for B2B E-commerce scenarios.
It includes:

Interactive dashboards for top-selling products, margin analysis & dynamic price suggestions

A secured login/signup flow

A real-time AI chatbot using Groq + LangChain

Pipeline orchestration with Prefect

Experiment tracking with MLflow

A modern animated UI built entirely with Streamlit

🚀 Setup Instructions
Follow these steps to run the project locally:

bash
# 1️⃣ Create a conda environment with Python 3.11
conda create -p venv python=3.11

# 2️⃣ Activate the environment
conda activate ./venv

# 3️⃣ Install all Python dependencies
pip install -r requirements.txt
⚙️ How to Run the Pipeline
Run the Prefect flow:
bash
python flow.py
This executes your ETL pipeline, generates predictions, and outputs results.

Start the MLflow UI (optional):
bash
mlflow ui
Visit http://127.0.0.1:5000 to explore model experiments & runs.

Launch the Streamlit app:
bash
streamlit run app.py
The web dashboard will open in your browser automatically.

🗂️ Project Structure
.
├── app.py                  # Main Streamlit web app
├── flow.py                 # Prefect pipeline
├── requirements.txt        # Python dependencies
├── output/                 # Generated predictions (.parquet)
├── users.db                # SQLite database for auth & chat
├── README.md               # Project overview & instructions
└── (other helper modules)
📌 Notes
Make sure you have your GROQ_API_KEY set in a .env file:
GROQ_API_KEY=your_groq_api_key_here

This project uses:
Streamlit for UI
Prefect for workflow orchestration
MLflow for model tracking
LangChain + Groq for the chatbot
All code is fully commented and modular for easy understanding.

🤖 AI-Assisted Tools
Some parts of this project (like LangChain usage & Groq integration) were accelerated using AI coding tools for faster prototyping.
