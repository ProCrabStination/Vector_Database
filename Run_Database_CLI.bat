@echo off
rem Command-line access to the vector store (search, list documents, ...). With no arguments it
rem prints the help guide. Run "Run_Database_CLI.bat --help" for the full usage and quoting notes.
rem   Run_Database_CLI.bat list_documents
rem   Run_Database_CLI.bat search_vectors_keywords --json "[\"gearbox torque\", 5, 20, true, [\"gearbox\"]]"
setlocal
call "%~dp0_env.bat" || exit /b 1
if "%~1"=="" (
    "%PY%" "%~dp0Vector_Database_Ollama.py" --help
) else (
    "%PY%" "%~dp0Vector_Database_Ollama.py" %*
)
pause
endlocal
