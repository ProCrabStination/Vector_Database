@echo off
rem Re-convert already-ingested PDFs whose picture crops were never saved to disk.
setlocal
call "%~dp0_env.bat" gpu || exit /b 1
"%PY%" "%~dp0Backfill_Missing_Images.py" %*
pause
endlocal
