@echo off
cd /d "C:\Users\saji\OneDrive\Desktop\climate-anomaly-dashboard"
echo Activating Virtual Environment...
call .\venv\Scripts\activate
echo Launching Streamlit App...
streamlit run app.py
pause