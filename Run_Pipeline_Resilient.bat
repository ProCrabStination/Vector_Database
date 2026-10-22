@echo off
rem Crash-proof ingest: runs every document in its own process, carries on past a crash, and
rem tries every document. Safe to re-run: failed documents are retried and finished ones resume.
rem   Run_Pipeline_Resilient.bat                        everything in Documents\Inputs
rem   Run_Pipeline_Resilient.bat --in-place "D:\Shared" scan a folder, reference originals (no copies)
setlocal
call "%~dp0_env.bat" gpu || exit /b 1
"%PY%" "%~dp0Run_Pipeline_Resilient.py" %*
pause
endlocal
