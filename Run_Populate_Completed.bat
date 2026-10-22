@echo off
rem One-off: build "Completed Documents For Reference" from the old Test_Documents layout (copies only).
rem   Run_Populate_Completed.bat --legacy-inputs "D:\Test_Documents" --legacy-working "D:\Test_Documents_Markdown" [--dry-run]
setlocal
if "%~1"=="" (
    echo Usage: %~nx0 --legacy-inputs ^<Test_Documents dir^> --legacy-working ^<Test_Documents_Markdown dir^> [--dry-run]
    pause
    exit /b 1
)
call "%~dp0_env.bat" || exit /b 1
"%PY%" "%~dp0Populate_Completed_From_Legacy.py" %*
pause
endlocal
