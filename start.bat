@echo off
REM IWriting launcher v5 - simple & clean

cd /d "C:\Users\22645\Desktop\Thesis_Agent"

echo.
echo ============================================================
echo    IWriting  Chinese Writing Feedback System
echo ============================================================
echo.
echo    Starting Gradio... please wait
echo    Keep this window open while using the app
echo    Close this window to quit
echo.
echo ============================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv\Scripts\python.exe not found
    pause
    exit /b 1
)

REM Run Gradio app in THIS window
call venv\Scripts\python.exe final_agent.py

if errorlevel 1 (
    echo.
    echo [FAILED] exit code %errorlevel%
    pause
)
