@echo off
REM IWriting launcher - pure ASCII to avoid CMD encoding issues

cd /d "C:\Users\22645\Desktop\Thesis_Agent"

echo.
echo ============================================================
echo    IWriting  Chinese Writing Feedback System
echo ============================================================
echo.
echo    Starting... keep this window open
echo    Browser will open in ~3 seconds at http://localhost:7860
echo    Close this window to quit
echo.
echo ============================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv\Scripts\python.exe not found
    echo         Please ensure venv is in the project root
    pause
    exit /b 1
)

REM Open browser after 3s in a new window, then close itself
start "" cmd /c "ping 127.0.0.1 -n 4 >nul && start http://localhost:7860"

REM Run Gradio app (foreground)
call venv\Scripts\python.exe final_agent.py

if errorlevel 1 (
    echo.
    echo [FAILED] exit code %errorlevel%
    pause
)
