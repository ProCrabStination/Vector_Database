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

rem The UI needs the Ollama server (embedding model lookup happens at import time)
if not defined OLLAMA_HOST set "OLLAMA_HOST=http://localhost:11434"
%PY% -c "import os,requests; requests.get(os.environ['OLLAMA_HOST'].rstrip('/')+'/api/tags', timeout=3)" >nul 2>&1
if errorlevel 1 (
    echo Ollama is not running at %OLLAMA_HOST%. Starting it ...
    where ollama >nul 2>&1
    if errorlevel 1 (
        echo Ollama is not installed or not on PATH. Install it from https://ollama.com and re-run.
        pause
        exit /b 1
    )
    start "Ollama" /min ollama serve
    set /a tries=0
    :wait_ollama
    timeout /t 1 /nobreak >nul
    %PY% -c "import os,requests; requests.get(os.environ['OLLAMA_HOST'].rstrip('/')+'/api/tags', timeout=3)" >nul 2>&1
    if errorlevel 1 (
        set /a tries+=1
        if %tries% lss 30 goto wait_ollama
        echo Ollama did not start within 30 seconds.
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
