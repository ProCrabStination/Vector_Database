@echo off
rem Ingest every supported document in a folder and its subfolders WITHOUT copying the originals:
rem the database stores each document's original path (picture crops are still kept under Documents).
rem   Run_Pipeline_In_Place.bat "D:\Shared\Manuals"
rem   Run_Pipeline_In_Place.bat            (prompts for the folder)
setlocal
set "SCAN_DIR=%~1"
if "%SCAN_DIR%"=="" set /p "SCAN_DIR=Folder to scan (including subfolders): "
if "%SCAN_DIR%"=="" (
    echo No folder given.
    exit /b 1
)
if not exist "%SCAN_DIR%\" (
    echo Folder not found: %SCAN_DIR%
    pause
    exit /b 1
)
call "%~dp0_env.bat" gpu || exit /b 1
"%PY%" "%~dp0Document_To_Database_Pipeline_Ollama.py" --in-place "%SCAN_DIR%"
pause
endlocal
