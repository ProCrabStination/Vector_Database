@echo off
rem Crash-proof ingest: runs every document in its own process, skips any document that crashes, and
rem carries on until all documents have been tried. Safe to re-run; finished documents are skipped.
rem   Run_Pipeline_Resilient.bat                        everything in Documents\Inputs
rem   Run_Pipeline_Resilient.bat --in-place "D:\Shared" scan a folder, reference originals (no copies)
rem   Run_Pipeline_Resilient.bat --retry-failed         try documents that failed before again
setlocal
call "%~dp0_env.bat" gpu || exit /b 1
"%PY%" "%~dp0Run_Pipeline_Resilient.py" %*
pause
endlocal
