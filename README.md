# Vector Database

A local, offline-friendly pipeline that turns engineering reference documents (PDFs, Word, PowerPoint,
Excel, images) into a searchable vector database, plus the tools to search it and ask questions about it.
Everything runs on your machine: [Docling](https://github.com/docling-project/docling) for conversion,
[Ollama](https://ollama.com) for embeddings, summaries and vision, FAISS + SQLite for storage.

## What it does

| Stage | What happens | Code |
|---|---|---|
| **Ingest** | Drop files in `Documents/Inputs`. PDFs are split into 10-page chunks, converted with Docling (layout, tables, OCR), and pictures are cropped and saved. | `Document_To_Database_Pipeline_Ollama.py`, `Docling_Convert_Ollama.py` |
| **Understand** | Tables are summarized by an LLM. Schematic-like pages are transcribed by a vision model. | `LLM_Summary_Ollama.py`, `Docling_Convert_Ollama.py` |
| **Chunk** | Docling's hybrid chunker splits text on structure, sized to the embedding model's context window. | `Hybrid_Chunking_Ollama.py` |
| **Embed + store** | Each chunk is embedded with Ollama and stored in a FAISS index (`vec_store/index.faiss`) with metadata in SQLite (`vec_store/meta.db`). | `Text_Vectorizing_Ollama.py`, `Vector_Database_Ollama.py` |
| **Search** | Semantic search with optional keyword filtering/boosting and per-document filters. | `Vector_Database_Ollama.py` |
| **Web UI** | Browser search with page previews, images, AI-suggested queries and reference summaries. | `Vector_Search_UI/` |
| **Ask** | Retrieval-augmented answers from an Ollama model, grounded in search results. | `Query_Ollama_With_Database.py` |
| **Claude plugin** | An MCP server so Claude (Cowork or Claude Code) can search, list and read the library. | `Claude Plugin/` |

Re-running the pipeline is safe: chunks already in the database are skipped.

## Requirements

- **Windows** with **Python 3** on PATH (developed on 3.14) (used only to create `.venv` on first run).
- **NVIDIA GPU with a CUDA build of PyTorch.** Document conversion forces CUDA
  ([Docling_Convert_Ollama.py:53](Docling_Convert_Ollama.py#L53)); a CPU-only torch crashes the converter.
  `requirements.txt` points pip at the `cu128` wheels. If a CPU build is already in the venv, see
  [Troubleshooting](#troubleshooting).
- **[Ollama](https://ollama.com)** running locally (`http://localhost:11434`) with these models pulled:

  | Purpose | Default model | Override with |
  |---|---|---|
  | Embeddings | `nomic-embed-text-v2-moe` | `OLLAMA_EMBEDDING_MODEL` |
  | Table / reference summaries, answers | `qwen3-vl:4b-instruct` | `AI_SUMMARY_MODEL` (summaries) |
  | Page-image transcription (vision) | `gemma4:12b` | `AI_VISION_MODEL`, `AI_VISION_HOST` |

  ```
  ollama pull nomic-embed-text-v2-moe
  ollama pull qwen3-vl:4b-instruct
  ollama pull gemma4:12b
  ```
- **LibreOffice** at `C:\Program Files\LibreOffice\program\soffice.exe` for Word/PowerPoint/Excel input.
- *Optional:* **Tesseract** (and **unpaper** for `--clean`) on PATH, only for `Run_Clean_PDF_For_OCR.bat`.

## Setup

1. Install the prerequisites above and start Ollama.
2. Double-click any launcher below. On first run it creates `.venv`, installs `requirements.txt`, and
   downloads the spaCy model `en_core_web_sm`. This takes a while once, then never again.

To set up by hand instead:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

## Launchers

Every program has a batch file in the project root. They all use the project's `.venv`, set it up if it
is missing, and keep the window open when finished. Arguments are passed straight through.

| Batch file | Runs | Arguments |
|---|---|---|
| `Launch_Web.bat` | Search web UI at <http://127.0.0.1:5151> (opens your browser) | none |
| `Run_Pipeline.bat` | Full ingest into the database | optional path to one file; default is everything in `Documents\Inputs` |
| `Run_Query.bat` | Ask Ollama a question grounded in the database | optional prompt file (default `Testing\Example_Prompt.md`), `--only-response` |
| `Run_Database_CLI.bat` | Command-line access to the vector store | function name and args; none prints the help guide |
| `Run_Docling_Convert.bat` | Convert `Documents\Inputs` to Docling output only, no database writes | none |
| `Run_Backfill_Missing_Images.bat` | Re-convert ingested PDFs whose picture crops were never saved | none |
| `Run_Clean_PDF_For_OCR.bat` | Deskew/clean a poor scan before ingesting it | `"in.pdf" "out.pdf" [--clean] [--oversample 300]` |
| `Run_Migrate_Paths.bat` | One-off: convert old absolute paths in `meta.db` to the relative layout | `--dry-run` to preview |
| `Run_Populate_Completed.bat` | One-off: build the Completed folder from the old `Test_Documents` layout | `--legacy-inputs <dir> --legacy-working <dir> [--dry-run]` |
| `Run_MCP_Server.bat` | Start the Claude plugin's MCP server by hand (testing) | none |

`_env.bat` is the shared setup script the launchers call; don't run it directly.

## Adding documents

1. Copy files into `Documents\Inputs` (subfolders are fine). Supported: PDF, Word (`.doc/.docx/.rtf/.odt`), PowerPoint (`.ppt/.pptx/.odp`) and images
   (`.jpg/.png/.tif/...`). Spreadsheets (`.xls/.xlsx/.csv`) are skipped by the batch run.
2. Make sure Ollama is running, then run `Run_Pipeline.bat`.
3. Watch the log. Each 10-page PDF unit is converted, chunked, summarized where needed, embedded and stored.
4. Finished documents appear under `Documents\Completed Documents For Reference\<doc>\`.

If a scan is poor, run it through `Run_Clean_PDF_For_OCR.bat` first and ingest the cleaned file.

### Folder layout

```
Documents/
  Inputs/                              drop new documents here
  Working Directory/                   scratch: 10-page PDF chunks, conversion output
  Completed Documents For Reference/   <doc>/<doc>.pdf plus <doc>/images/
vec_store/
  index.faiss                          vector index
  meta.db                              chunk text, metadata, relative document paths
```

The database stores paths **relative to `Documents/`** with forward slashes, pointing at the full
completed document. Chunk page ranges are kept in the row metadata, so the UI opens the right page. This
keeps the project portable: move the folder and nothing breaks. `Documents/` is git-ignored.

## Using it

### Web search UI (`Launch_Web.bat`)

- Natural-language query plus optional keywords, top-K per query and overall, and a document filter.
- **Search with AI suggested prompt** rewrites your question into a keyword-dense query and keywords.
- Results show the matching text, tables (with summaries), and picture crops; click to open the source
  PDF at the right page.
- **Summarize the most important references** has the LLM condense the top hits.

### Asking questions (`Run_Query.bat`)

The prompt file is JSON:

```json
{"prompt": "What circuit breakers are available for a 10k rated panel?",
 "setup": "You are an Electrical Engineering expert assistant...",
 "chat_history": []}
```

It extracts keywords with spaCy, searches the database, and sends the retrieved text, tables and pictures
to the model. Output is JSON; `--only-response` trims it to `{"ai_response": ...}`.
Environment: `AI_GATEWAY_HOST` (default `http://localhost:11434`), `AI_GATEWAY_API_KEY`, `OLLAMA_TEMPERATURE`.

### Database CLI (`Run_Database_CLI.bat`)

```
Run_Database_CLI.bat list_documents
Run_Database_CLI.bat search_vectors_keywords --json "[\"gearbox torque\", 5, 20, true, [\"gearbox\"]]"
Run_Database_CLI.bat --help
```

Functions: `document_exists`, `processed_chunk_indices`, `list_documents`, `add_vector`, `search_vectors`,
`search_vectors_keywords`, `format_search_results`, `reset_database` (**deletes the store**). Each call prints one
line of JSON on stdout. `--help` explains the `--json` mode, which avoids shell-quoting problems.

### Claude plugin

`vector-docs.plugin` (built from `Claude Plugin/`) gives Claude the tools `search_documents`, `list_documents`
and `read_document_pages`. Install steps and the one-time `set_project_home.bat` are in
[Claude Plugin/README.md](Claude%20Plugin/README.md).

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `OLLAMA_HOST` | Ollama server for model info and summaries | `http://localhost:11434` |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model | `nomic-embed-text-v2-moe` |
| `AI_SUMMARY_MODEL` | Summary model | `qwen3-vl:4b-instruct` |
| `AI_VISION_MODEL` / `AI_VISION_HOST` | Vision model and host | `gemma4:12b` / `http://localhost:11434` |
| `AI_GATEWAY_HOST` / `AI_GATEWAY_API_KEY` | Host and key for `Run_Query.bat` | `http://localhost:11434` / none |
| `VECDB_STORE_DIR` | Where `index.faiss` and `meta.db` live | `vec_store/` |
| `VECDB_DOCS_ROOT` | Root of the `Documents` folder | `Documents/` |

Conversion settings (image scale, OCR, accelerator, page batch size) are at the top of
[Docling_Convert_Ollama.py](Docling_Convert_Ollama.py). PDFs are chunked 10 pages at a time
(`PDF_PAGES_PER_CHUNK` in the pipeline).

## Other files

- `Hybrid_Chunking_Ollama.py`, `LLM_Summary_Ollama.py`, `Text_Vectorizing_Ollama.py`, `Ollama_Model_Info.py`:
  library modules used by the pipeline. Each can also be run directly as a demo or smoke test with
  `.venv\Scripts\python.exe <file>`.
- `Testing/`: experiment and smoke-test scripts, plus `Example_Prompt.md`.
- `API version (deprecated)/`: the earlier gateway-API implementation, kept for reference only.

## Troubleshooting

**Pipeline exits silently during conversion, or `CUDA is not available`.**
The venv has a CPU-only PyTorch. Forcing CUDA on it either errors or crashes (Windows heap-corruption
code `0xc0000374`). The launchers detect this and tell you; to fix it:

```
.venv\Scripts\python.exe -m pip uninstall -y torch torchvision
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

The last command must print `True`. As a fallback without a GPU, set the accelerator to
`AcceleratorDevice.AUTO` in `Docling_Convert_Ollama.py` (much slower).

**`rapidocr cannot be used because onnxruntime is not installed`.** Run `pip install -r requirements.txt` again
(`onnxruntime` is listed).

**Search or ingest fails with a connection error.** Ollama isn't running, or a model isn't pulled. Check
`ollama list`.

**`std::bad_alloc` on large drawings.** Lower `images_scale` or `page_batch_size` in
`Docling_Convert_Ollama.py`; the comments there explain the trade-off.

**`Run_Query.bat` fails to load spaCy.** Run `.venv\Scripts\python.exe -m spacy download en_core_web_sm`.
