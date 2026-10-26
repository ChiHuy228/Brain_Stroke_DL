@echo off
rem Chuyen tu scripts ve thu muc goc chua app.py.
cd /d "%~dp0.."
python -m streamlit run app.py
pause
