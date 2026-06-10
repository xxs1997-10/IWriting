@echo off
setlocal
chcp 65001 >nul

REM ==================== IWriting 启动器 ====================
REM 用途：双击即可启动国际中文写作智能反馈系统
REM 作者：Mavis · 2026-06-10

cd /d "C:\Users\22645\Desktop\Thesis_Agent"

echo.
echo ============================================================
echo    IWriting  国际中文写作智能反馈系统
echo ============================================================
echo.
echo    启动中... 请勿关闭此窗口
echo    Gradio 启动后会自动打开浏览器
echo    关闭本窗口即可退出程序
echo.
echo ============================================================
echo.

REM 检查 venv
if not exist "venv\Scripts\python.exe" (
    echo [错误] 未找到 venv\Scripts\python.exe
    echo        请确认 venv 已创建在当前目录
    pause
    exit /b 1
)

REM 启动并自动打开浏览器
REM 启动后等 3 秒，让 Gradio 起服，再用默认浏览器打开 localhost:7860
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:7860"

call venv\Scripts\python.exe final_agent.py

if errorlevel 1 (
    echo.
    echo [启动失败] 错误码 %errorlevel%
    pause
)
