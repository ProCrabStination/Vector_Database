@echo off
rem Run this ONCE from inside the project ("<project>\Claude Plugin\server") to register where the
rem Vector_Database project lives. Needed only when the plugin is installed as a copy (e.g. in Cowork).
setlocal
for %%I in ("%~dp0..\..") do set "PROJECT_DIR=%%~fI"

if not exist "%PROJECT_DIR%\Vector_Database_Ollama.py" goto notfound

if not exist "%APPDATA%\vector-docs" mkdir "%APPDATA%\vector-docs"
>"%APPDATA%\vector-docs\project_home.txt" echo %PROJECT_DIR%
echo Registered project folder: %PROJECT_DIR%
echo Restart Cowork or Claude so the vector-docs server picks it up.
pause
exit /b 0

:notfound
echo Run this script from inside the Vector_Database project, in "Claude Plugin\server".
echo Could not find Vector_Database_Ollama.py in: %PROJECT_DIR%
pause
exit /b 1
