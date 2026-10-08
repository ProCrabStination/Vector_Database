@echo off
rem Shared setup for the launcher .bat files (not meant to be run directly).
rem   call "%~dp0_env.bat" [gpu]
rem Creates .venv on first run, installs requirements.txt if key packages are missing, and sets PY.
rem Pass "gpu" for programs that force CUDA (Docling conversion); it fails fast if torch has no CUDA.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create .venv. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)
set "PY=%~dp0.venv\Scripts\python.exe"

"%PY%" -c "import importlib.util as u,sys; sys.exit(any(u.find_spec(m) is None for m in ('flask','faiss','numpy','requests','fitz','docling','torch','spacy','mcp')))" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies from requirements.txt ^(first run can take a while^) ...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed.
        pause
        exit /b 1
    )
)

"%PY%" -c "import spacy.util as u,sys; sys.exit(0 if u.is_package('en_core_web_sm') else 1)" >nul 2>&1
if errorlevel 1 (
    echo Downloading spaCy model en_core_web_sm ^(used for keyword extraction^) ...
    "%PY%" -m spacy download en_core_web_sm
)

if /i "%~1"=="gpu" (
    "%PY%" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo ERROR: PyTorch cannot see a CUDA GPU ^(a CPU-only build is probably installed^).
        echo Document conversion forces CUDA and will crash without it. Fix with:
        echo   "%PY%" -m pip uninstall -y torch torchvision
        echo   "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
        pause
        exit /b 1
    )
)
exit /b 0
