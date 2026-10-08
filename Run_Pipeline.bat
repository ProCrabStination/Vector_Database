@echo off
rem Document -> database pipeline. Ingests everything under Documents\Inputs, or one file if given:
rem   Run_Pipeline.bat
rem   Run_Pipeline.bat "C:\path\to\document.pdf"
setlocal
call "%~dp0_env.bat" gpu || exit /b 1
"%PY%" "%~dp0Document_To_Database_Pipeline_Ollama.py" %*
pause
endlocal
