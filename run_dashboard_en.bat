@echo off
title EE136IU Comprehensive Model Benchmark Dashboard (English Edition)
echo ========================================================
echo   EE136IU - Topic 1: Comprehensive Model Benchmark
echo   Starting English Web Dashboard...
echo ========================================================
echo.
cd /d "%~dp0"
python -m streamlit run app_en.py
pause
