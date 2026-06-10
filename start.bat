@echo off
REM IWriting launcher v3 - pure ASCII + port polling to avoid 7860 not ready

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

REM Wait for port 7860 to be open, then launch browser in a new window
REM This avoids "ERR_CONNECTION_REFUSED" when browser opens before Gradio is ready
start "" cmd /c "powershell -NoProfile -Command ^
    \"for(\\\$i=0;\\\$i -lt 60;\\\$i++){^
        try{^
            \\\$c=New-Object System.Net.Sockets.TcpClient;^
            \\\$c.Connect('127.0.0.1',7860);^
            \\\$c.Close();^
            Start-Process 'http://127.0.0.1:7860';^
            exit^
        }catch{Start-Sleep -Seconds 1}}^
        Write-Host 'Browser auto-open timeout (60s) - please open http://127.0.0.1:7860 manually'\""

REM Run Gradio app (foreground - this window must stay open)
call venv\Scripts\python.exe final_agent.py

if errorlevel 1 (
    echo.
    echo [FAILED] exit code %errorlevel%
    pause
)
