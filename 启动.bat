@echo off
chcp 65001 >nul
cd /d "%~dp0backend"
echo 正在启动图片高清化服务，首次启动约需 10 秒...
start "" /b cmd /c "timeout /t 6 >nul && start http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
