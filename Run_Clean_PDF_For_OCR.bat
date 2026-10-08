@echo off
rem Deskew/clean a poor scan before ingesting it (needs Tesseract on PATH; unpaper for --clean).
rem   Run_Clean_PDF_For_OCR.bat "bad_scan.pdf" "bad_scan_cleaned.pdf" [--clean] [--oversample 300]
setlocal
if "%~2"=="" (
    echo Usage: %~nx0 "input.pdf" "output.pdf" [--clean] [--oversample 300]
    pause
    exit /b 1
)
call "%~dp0_env.bat" || exit /b 1
"%PY%" "%~dp0Clean_PDF_For_OCR.py" %*
pause
endlocal
