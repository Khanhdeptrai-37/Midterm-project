@echo off
title EE136IU Comprehensive Model Benchmark Dashboard
echo ========================================================
echo   EE136IU - Topic 1: Comprehensive Model Benchmark
echo   Starting Streamlit Web Dashboard...
echo ========================================================
echo.
cd /d "%~dp0"
python -m streamlit run app.py
pause
