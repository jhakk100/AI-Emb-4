@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" stages\models.py
if errorlevel 1 goto :failed
echo Setup complete. Next: run.bat
pause
exit /b 0
:failed
echo Setup failed. Check the error above.
pause
exit /b 1
