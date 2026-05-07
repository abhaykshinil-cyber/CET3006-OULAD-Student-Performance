@echo off
setlocal enabledelayedexpansion
title Fixing + Pushing OULAD Repo

echo.
echo ============================================================
echo   Fixing OULAD Git Repo and Pushing to GitHub
echo ============================================================
echo.

cd /d "C:\Users\abhay\OneDrive\Documents\RESEARCH PAPER-OULAD Paper"

echo [Step 1] Removing broken .git folder...
if exist ".git" (
    rd /s /q ".git"
    echo   Removed old .git folder.
) else (
    echo   No .git folder found, skipping.
)

echo [Step 2] Initialising fresh git repo...
git init -b main
git config user.name "Abhay Kalathil Shinil"
git config user.email "abhaykshinil@gmail.com"

echo [Step 3] Staging all files...
git add .
echo   Done staging.

echo [Step 4] Creating initial commit...
git commit -m "Initial commit: CET3006 OULAD student performance prediction research"

echo [Step 5] Adding remote and pushing...
git remote add origin git@github.com:abhaykshinil-cyber/CET3006-OULAD-Student-Performance.git
git push -u origin main

echo.
echo ============================================================
echo   OULAD repo pushed!
echo   https://github.com/abhaykshinil-cyber/CET3006-OULAD-Student-Performance
echo ============================================================
echo.
pause
