@echo off
title Push EE136 Project to GitHub
color 0A
echo ========================================================
echo   EE136IU - Pushing Project to GitHub
echo   Repository: https://github.com/Khanhdeptrai-37/Midterm-project
echo ========================================================
echo.

set "PATH=C:\Users\dkhanh\AppData\Local\MinGit\cmd;C:\Users\dkhanh\AppData\Local\gh\bin;%PATH%"
cd /d "%~dp0"

echo [1/3] Setting up Git Credential Helper...
call gh auth setup-git

echo.
echo [2/3] Checking GitHub Authentication...
call gh auth status >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo -------------------------------------------------------------------
    echo Ban can dang nhap GitHub 1 lan duy nhat:
    echo 1. Trinh duyet web se tu mo (hoac nhan Enter de mo).
    echo 2. Dan ma code (neu co) va bam "Authorize github".
    echo -------------------------------------------------------------------
    echo.
    call gh auth login -w -p https
)

echo.
echo [3/3] Dang day toan bo du an len GitHub (main branch)...
git push -u origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo ========================================================
    echo   [THANH CONG] DA UP TOAN BO CODE LEN GITHUB!
    echo   Xem repository tai:
    echo   https://github.com/Khanhdeptrai-37/Midterm-project
    echo ========================================================
) else (
    echo ========================================================
    echo   [THONG BAO] Co van de xay ra khi push.
    echo   Hay xem thong bao loi o tren de biet chi tiet.
    echo ========================================================
)
echo.
pause
