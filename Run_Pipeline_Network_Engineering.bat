@echo off
rem Crash-proof ingest of the Engineering_P2 network share (and all subfolders).
rem Originals stay on the share; the database stores their network paths (no local copies).
rem Safe to re-run: finished documents are skipped, and documents that crashed are skipped
rem until you pass --retry-failed.
rem   Run_Pipeline_Network_Engineering.bat
rem   Run_Pipeline_Network_Engineering.bat --retry-failed
setlocal
set "SHARE=\\azdc01\Engineering_P2"

echo Checking that %SHARE% is reachable ...
if not exist "%SHARE%\" (
    echo Cannot reach %SHARE%
    echo Check the network connection / VPN and that you have access to the share, then try again.
    pause
    exit /b 1
)

echo Checking the Python environment and GPU ...
call "%~dp0_env.bat" gpu || exit /b 1
echo Starting the pipeline ...
"%PY%" "%~dp0Run_Pipeline_Resilient.py" --in-place "%SHARE%" %*
pause
endlocal
