@echo off
rem Ask Ollama a question grounded in the database. Takes a JSON prompt file
rem (default: Testing\Example_Prompt.md); add --only-response to print just the answer.
rem   Run_Query.bat
rem   Run_Query.bat "my_prompt.md" --only-response
setlocal
call "%~dp0_env.bat" || exit /b 1
"%PY%" "%~dp0Query_Ollama_With_Database.py" %*
pause
endlocal
