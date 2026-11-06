@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
echo CricketShot will be available at http://127.0.0.1:6040
".venv\Scripts\python.exe" app.py
pause
