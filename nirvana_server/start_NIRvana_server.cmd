@echo off
cd /d "%~dp0"
if not exist ".nirvana_venv\Scripts\python.exe" (
    echo Cannot find .nirvana_venv\Scripts\python.exe. Use your existing LightField Python environment.
    pause
    exit /b 1
)
".nirvana_venv\Scripts\python.exe" -u NIRvana_device_server.py
pause
