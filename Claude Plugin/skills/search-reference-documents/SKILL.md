---
name: search-reference-documents
description: Answer questions from the local engineering reference library (NFPA 70/NEC and other electrical codes, product catalogs such as Appleton and Belden, industry standards, motor/electrical textbooks) by searching it with the vector-docs tools. Use whenever a question may be answered by those documents, needs a code article or product spec, or the user asks to look something up in the reference documents.
---

# Search the reference documents

The `vector-docs` MCP server searches a local library of engineering documents. Prefer it over memory for anything citeable: code requirements, article numbers, product specifications, tables.

## Tools

- `search_documents(query, keywords, documents, top_k)` - semantic search, optionally boosted by literal keywords and limited to named documents. Returns passages with `document`, `pages`, `headings`, `text`, `tables` and `images` (file paths).
- `list_documents()` - the documents available, and their exact names (needed for the `documents` filter).
- `read_document_pages(document, start_page, end_page)` - raw page text, for reading around a hit (max 20 pages).

## How to search well

1. Rewrite the user's question into a keyword-dense query: code/article numbers, equipment names, units. Do not just paste the question.
2. Put 2-3 alternative phrasings on separate lines of `query`; they are all searched and merged.
3. Add `keywords` (comma-separated literal terms likely to appear verbatim, e.g. `250.122, equipment grounding`) when you know specific terms. If a keyword search returns nothing, retry without keywords.
4. If you know which document applies (e.g. the NEC), pass it in `documents` using the name from `list_documents`.
5. If the first results are weak, vary the wording or raise `top_k` before concluding the library lacks the answer.
6. When a passage looks relevant but cut off, or a table continues, call `read_document_pages` with the `pages` range it reported.

## Answering

- Base the answer on the returned passages; quote exact figures and article numbers from the text, not from memory.
- Cite each claim with the document name and page range, e.g. *NFPA 70 NEC 2023, pp. 41-50*. `pages` is the 10-page range the passage came from; use `read_document_pages` to pin down the exact page when it matters.
- Say plainly when the library does not contain the answer, rather than filling the gap.
- Results include absolute `file` and `images` paths; use them if the user wants to open the source or see a figure.

## If search fails

Search needs the Ollama server running with the embedding model installed. If `search_documents` reports a connection error, tell the user to start Ollama; `list_documents` and `read_document_pages` still work without it.
