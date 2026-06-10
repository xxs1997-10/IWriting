@echo off
REM IWriting launcher v4 - pure ASCII, calls PowerShell via separate .ps1

cd /d "C:\Users\22645\Desktop\Thesis_Agent"

echo.
echo ============================================================
echo    IWriting  Chinese Writing Feedback System
echo ============================================================
echo.
echo    Starting... keep this window open
echo    Browser will open automatically when ready
echo    Close this window to quit
echo.
echo ============================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv\Scripts\python.exe not found
    pause
    exit /b 1
)

REM Launch the wait-and-open script in a NEW separate window
start "" powershell -NoProfile -ExecutionPolicy Bypass -File "wait_and_open.ps1"

REM Run Gradio app in THIS window (foreground)
call venv\Scripts\python.exe final_agent.py

if errorlevel 1 (
    echo.
    echo [FAILED] exit code %errorlevel%
    pause
)
