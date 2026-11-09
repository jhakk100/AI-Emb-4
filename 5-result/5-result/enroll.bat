@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
set /p NAME=Name: 
if "%NAME%"=="" exit /b 1
".venv\Scripts\python.exe" integrate.py enroll --name "%NAME%" --show %*
pause
