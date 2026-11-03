@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" models.py
if errorlevel 1 goto :failed
echo Setup complete. Double-click run.bat to start.
pause
exit /b 0
:failed
echo Setup failed. Check the error above.
pause
exit /b 1
