@echo off
rem One-off: migrate vec_store\meta.db paths to the portable relative layout (backs up meta.db first).
rem   Run_Migrate_Paths.bat --dry-run     preview only
rem   Run_Migrate_Paths.bat               apply
setlocal
call "%~dp0_env.bat" || exit /b 1
"%PY%" "%~dp0Migrate_Paths_To_Relative.py" %*
pause
endlocal
