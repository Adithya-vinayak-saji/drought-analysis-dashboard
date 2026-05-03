@echo off
C:
cd C:\drought-analysis-dashboard
echo 🔌 Activating Virtual Environment...
call venv\Scripts\activate
echo 🚀 Launching Climate Dashboard...
streamlit run app.py
pause