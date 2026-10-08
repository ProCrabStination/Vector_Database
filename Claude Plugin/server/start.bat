@echo off
rem Launches the vector-docs MCP server over stdio. stdout is the MCP protocol channel: print nothing here.
setlocal
set "HERE=%~dp0"

rem Project folder, first match wins:
rem   1. VECDB_HOME environment variable
rem   2. %APPDATA%\vector-docs\project_home.txt  (written by set_project_home.bat; survives plugin copies)
rem   3. two folders up from this script (plugin used in place inside the project)
if not defined VECDB_HOME if exist "%APPDATA%\vector-docs\project_home.txt" set /p VECDB_HOME=<"%APPDATA%\vector-docs\project_home.txt"
if not defined VECDB_HOME set "VECDB_HOME=%HERE%..\.."

rem Prefer the project's virtual environment
set "PY=python"
if exist "%VECDB_HOME%\.venv\Scripts\python.exe" set "PY=%VECDB_HOME%\.venv\Scripts\python.exe"

set PYTHONUTF8=1
"%PY%" "%HERE%vector_docs_mcp.py"
