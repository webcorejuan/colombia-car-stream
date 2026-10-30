@echo off
title Colombia 24/7 Street AI Stream (RTX 4060)
echo ===================================================
echo   COLOMBIA 24/7 TIKTOK AI CAR COUNTER - RTX 4060
echo ===================================================

if not exist venv (
    echo [1/3] Creating Python virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

echo [2/3] Checking dependencies...
pip install -r requirements.txt

echo [3/3] Starting AI Vision Server on http://localhost:8000 ...
python main.py
pause
