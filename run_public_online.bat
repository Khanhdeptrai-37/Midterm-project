@echo off
title EE136IU AMC Dashboard - Public Online Launcher
echo ========================================================
echo   EE136IU - Topic 1: Comprehensive Model Benchmark
echo   Starting Streamlit Dashboard + Public Cloudflare Tunnel...
echo ========================================================
echo.
cd /d "%~dp0"

echo 1. Starting Streamlit in background...
start /B python -m streamlit run app_en.py --server.headless true --server.port 8501

echo 2. Waiting 3 seconds for server to initialize...
timeout /t 3 /nobreak >nul

echo 3. Launching Cloudflare Tunnel for Public Online Access...
echo --------------------------------------------------------
echo Notice: Look for the line containing:
echo "https://......trycloudflare.com"
echo Copy that link and send to anyone on 4G/WiFi to access!
echo --------------------------------------------------------
echo.
cloudflared.exe tunnel --url http://localhost:8501
pause
