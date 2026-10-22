@echo off
rem Convert every document in Documents\Inputs to Docling JSON/markdown in Documents\Working Directory
rem (conversion only; nothing is added to the database - use Run_Pipeline.bat for that).
setlocal
call "%~dp0_env.bat" gpu || exit /b 1
"%PY%" "%~dp0Docling_Convert_Ollama.py" %*
pause
endlocal
