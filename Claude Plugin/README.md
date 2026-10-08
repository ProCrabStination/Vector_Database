# vector-docs plugin

Lets Claude (Cowork or Claude Code) search the Vector_Database reference library.

| Part | Purpose |
|---|---|
| `.mcp.json` + `server/` | MCP server with tools `search_documents`, `list_documents`, `read_document_pages` |
| `skills/search-reference-documents` | Teaches Claude when and how to use the tools and how to cite results |

## Requirements

- The Vector_Database project folder with its `.venv` (run `Launch_Web.bat` once to create it; it installs `requirements.txt`, which includes `mcp`).
- Ollama running with the embedding model, for `search_documents` (listing and page reading work without it).

## Install

**Cowork:** Customize > Plugins > upload `vector-docs.plugin` (in the project root).
**Claude Code:** `claude --plugin-dir "Claude Plugin"` (from the project root), or add it to a marketplace.

The server has to know where the project lives. Used in place (`<project>/Claude Plugin`) it works out
of the box. Cowork copies an uploaded plugin elsewhere, so run this **once**, from inside the project,
then restart Cowork:

    "Claude Plugin\server\set_project_home.bat"

That saves the project location to `%APPDATA%\vector-docs\project_home.txt`. Re-run it if you move the
project. (`VECDB_HOME`, if set, takes priority.)

## Test the server by hand

    .venv\Scripts\python.exe "Claude Plugin\server\vector_docs_mcp.py"

It speaks MCP over stdio, so it will sit waiting for input; use an MCP client or `npx @modelcontextprotocol/inspector`.

## Rebuild the .plugin file

    powershell -Command "Compress-Archive -Path 'Claude Plugin\*' -DestinationPath vector-docs.zip -Force; Move-Item vector-docs.zip vector-docs.plugin -Force"
