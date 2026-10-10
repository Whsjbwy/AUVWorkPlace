@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0run_remus.py" %*
if errorlevel 1 (
  echo Simulation failed. See the error above.
  pause
  exit /b 1
)
start "" "%~dp0results\remus100\states.png"
start "" "%~dp0results\remus100\trajectory.gif"
pause
