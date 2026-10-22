@echo off
rem Runs the ingest pipeline and restarts it whenever the Python process dies (crash, OOM kill, etc.).
rem Finished documents are skipped on re-run, so each restart picks up where the last one stopped.
rem Stops when Python exits cleanly (code 0) or after MAX_RESTARTS consecutive crashes.
rem   Run_Pipeline_AutoRestart.bat                        everything in Documents\Inputs
rem   Run_Pipeline_AutoRestart.bat --in-place "D:\Shared" scan a folder, reference originals
rem   Run_Pipeline_AutoRestart.bat --retry-failed         try documents that failed before again
setlocal
set "MAX_RESTARTS=20"
set "WAIT_SECONDS=15"
call "%~dp0_env.bat" gpu || exit /b 1

set /a RESTARTS=0
:run
echo.
echo [%date% %time%] Starting pipeline (restart %RESTARTS% of %MAX_RESTARTS%) ...
"%PY%" -X faulthandler "%~dp0Run_Pipeline_Resilient.py" %*
set "RC=%errorlevel%"
if "%RC%"=="0" (
    echo [%date% %time%] Pipeline finished cleanly.
    goto done
)
echo [%date% %time%] Python exited with code %RC% - it was killed or crashed.
set /a RESTARTS+=1
if %RESTARTS% gtr %MAX_RESTARTS% (
    echo Giving up after %MAX_RESTARTS% restarts.
    goto done
)
echo Restarting in %WAIT_SECONDS% seconds so GPU/RAM can be released. Press Ctrl+C to stop.
timeout /t %WAIT_SECONDS% /nobreak >nul
goto run

:done
pause
endlocal
