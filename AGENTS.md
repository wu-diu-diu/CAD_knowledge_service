# Repository Guidelines

## Project Structure & Module Organization
- `main.py` contains the FastAPI app, ingestion pipeline, and FAISS search logic.
- `docs/` is the default ingest source when calling `/ingest` without parameters.
- `rag_store/` holds persisted FAISS index and metadata (`index.faiss`, `metadata.json`) plus uploads.
- `output/` is a scratch/output directory (not used in app code yet).
- `Dockerfile`, `pyproject.toml`, `requirements.txt`, and `uv.lock` define build/runtime dependencies.

## Build, Test, and Development Commands
- `uv run uvicorn main:app --host 127.0.0.1 --port 8008` runs the API locally (single GPU).
- `RAG_EMBED_MULTI_GPU=1 RAG_EMBED_DEVICE=cuda RAG_EMBED_BATCH_SIZE=128 uvicorn main:app --host 0.0.0.0 --port 8008` runs with multi-GPU embeddings.
- `docker build -t autocad-ragserver:latest .` builds the image.
- `sudo docker run --gpus all -p 8008:8008 ... autocad-ragserver:latest` runs GPU-enabled container (see `README.md` for volumes).

## Coding Style & Naming Conventions
- Python 3.12+, 4-space indentation, follow PEP 8.
- Prefer type hints for public functions and dataclasses or Pydantic models for request bodies.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- No formatter or linter is configured; keep diffs minimal and readable.

## Testing Guidelines
- There is no automated test suite in this repo yet.
- Manual API checks are done via `curl` or PowerShell examples in `README.md`.
- If you add tests, document the framework and add a `tests/` directory with `test_*.py` files.

## Commit & Pull Request Guidelines
- Commit messages are descriptive sentences (often in Chinese); keep them clear and specific.
- PRs should include: summary, test evidence (commands + results), and any new env vars.
- If changing API behavior or response shape, include an example request/response.

## Configuration & Runtime Notes
- Key env vars: `RAG_EMBED_MODEL`, `RAG_EMBED_DEVICE`, `RAG_EMBED_BATCH_SIZE`,
  `RAG_EMBED_MULTI_GPU`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, and MinerU settings.
- Persist indexes by mounting `rag_store/` when running in containers.
