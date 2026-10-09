@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Run Launch_Web.bat once first to set up the environment.
    pause
    exit /b 1
)
set "PY=.venv\Scripts\python.exe"

rem Allow inbound connections on the UI port (needs admin; ignored with a hint if it fails)
netsh advfirewall firewall show rule name="Vector Search UI" >nul 2>&1
if errorlevel 1 (
    netsh advfirewall firewall add rule name="Vector Search UI" dir=in action=allow protocol=TCP localport=5151 profile=private >nul 2>&1
    if errorlevel 1 echo Note: could not add a firewall rule. Run once as Administrator if other PCs can't connect.
)

start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:5151"

%PY% "Vector_Search_UI\app.py" --lan
if errorlevel 1 pause
endlocal
