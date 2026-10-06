@echo off
setlocal
cd /d "%~dp0"

rem Create the virtual environment on first run
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create .venv. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)
set "PY=.venv\Scripts\python.exe"

rem Install dependencies if the UI's key packages are missing
%PY% -c "import flask, faiss, numpy, requests, fitz" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies from requirements.txt ^(first run can take a while^) ...
    %PY% -m pip install --upgrade pip
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed.
        pause
        exit /b 1
    )
)

echo Starting Vector Search UI at http://127.0.0.1:5151 ...
rem Open the browser a few seconds after the server starts
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5151"

%PY% "Vector_Search_UI\app.py"
if errorlevel 1 pause
endlocal
