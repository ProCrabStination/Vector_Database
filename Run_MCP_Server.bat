@echo off
rem Start the vector-docs MCP server (stdio) by hand for testing. Normally Claude launches it itself
rem through the plugin; see "Claude Plugin\README.md".
call "%~dp0Claude Plugin\server\start.bat"
