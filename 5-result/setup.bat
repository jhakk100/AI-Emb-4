@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto :failed
rem MediaPipe uses contrib: remove old cv2 distributions before installing one provider.
".venv\Scripts\python.exe" -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
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
