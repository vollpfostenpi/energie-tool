@echo off
setlocal
cd /d "%~dp0"
python --version >nul 2>&1 || (echo Bitte Python 3 installieren: https://www.python.org/downloads/ & pause & exit /b 1)
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
pause
