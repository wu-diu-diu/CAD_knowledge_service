import hashlib
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from mineru.cli.gradio_app import to_markdown

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
STORE_DIR = BASE_DIR / "rag_store"
UPLOAD_DIR = STORE_DIR / "uploads"
INDEX_PATH = STORE_DIR / "index.faiss"
META_PATH = STORE_DIR / "metadata.json"

EMBED_MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
EMBED_DEVICE = os.getenv("RAG_EMBED_DEVICE", "cuda")
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "64"))
EMBED_MULTI_GPU = os.getenv("RAG_EMBED_MULTI_GPU", "0").lower() in ("1", "true", "yes")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
MINERU_BACKEND = os.getenv("RAG_MINERU_BACKEND", "hybrid-auto-engine")
MINERU_LANG = os.getenv("RAG_MINERU_LANG", "ch")
MINERU_OCR = os.getenv("RAG_MINERU_OCR", "false").lower() in ("1", "true", "yes")
MINERU_MAX_PAGES = int(os.getenv("RAG_MINERU_MAX_PAGES", "1000"))

app = FastAPI()

_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_metadata: List[dict] = []


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


def ensure_dirs() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_markdown(md: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", md)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"`{3}.*?`{3}", " ", text, flags=re.S)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + CHUNK_SIZE)
        if end < length:
            split_at = text.rfind("\n", start, end)
            if split_at > start + 200:
                end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = EMBED_DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        _model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_model()
    if EMBED_MULTI_GPU and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        target_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        pool = model.start_multi_process_pool(target_devices=target_devices)
        try:
            vectors = model.encode_multi_process(
                texts,
                pool,
                batch_size=EMBED_BATCH_SIZE,
                normalize_embeddings=True,
            )
        finally:
            model.stop_multi_process_pool(pool)
    else:
        vectors = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
        )
    return np.asarray(vectors, dtype=np.float32)


def load_metadata() -> List[dict]:
    if META_PATH.exists():
        with META_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    return []


def save_metadata(data: List[dict]) -> None:
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def load_index() -> Optional[faiss.Index]:
    if INDEX_PATH.exists():
        return faiss.read_index(str(INDEX_PATH))
    return None


def save_index(index: faiss.Index) -> None:
    faiss.write_index(index, str(INDEX_PATH))


def ensure_index(dim: int) -> faiss.Index:
    global _index
    if _index is None:
        _index = load_index()
    if _index is None:
        _index = faiss.IndexFlatIP(dim)
    return _index


def rebuild_index_from_metadata() -> None:
    global _index
    if not _metadata:
        return
    texts = [item.get("text", "") for item in _metadata]
    vectors = embed_texts(texts)
    _index = faiss.IndexFlatIP(vectors.shape[1])
    _index.add(vectors)
    save_index(_index)


async def parse_pdf(path: Path) -> str:
    md_content, _, _, _ = await to_markdown(
        str(path),
        end_pages=MINERU_MAX_PAGES,
        is_ocr=MINERU_OCR,
        formula_enable=True,
        table_enable=True,
        language=MINERU_LANG,
        backend=MINERU_BACKEND,
        url=None,
    )
    return md_content


async def load_text_from_file(path: Path) -> Tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        md = await parse_pdf(path)
        return clean_markdown(md), "pdf"
    if suffix == ".md":
        content = path.read_text(encoding="utf-8", errors="ignore")
        return clean_markdown(content), "md"
    raise ValueError(f"Unsupported file type: {suffix}")


async def ingest_path(path: Path) -> Tuple[int, int]:
    if path.is_dir():
        files = [
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in (".pdf", ".md")
        ]
    else:
        files = [path]

    if not files:
        return 0, 0

    existing = {item.get("source_id") for item in _metadata}
    total_chunks = 0
    processed_files = 0

    for file_path in files:
        source_id = file_sha256(file_path)
        if source_id in existing:
            continue

        try:
            text, source_type = await load_text_from_file(file_path)
        except Exception:
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        vectors = embed_texts(chunks)
        index = ensure_index(vectors.shape[1])
        index.add(vectors)

        base_offset = len(_metadata)
        for i, chunk in enumerate(chunks):
            _metadata.append(
                {
                    "id": base_offset + i,
                    "text": chunk,
                    "source_path": str(file_path),
                    "source_type": source_type,
                    "source_id": source_id,
                    "chunk_index": i,
                }
            )

        total_chunks += len(chunks)
        processed_files += 1
        existing.add(source_id)

    if processed_files > 0:
        save_index(_index)
        save_metadata(_metadata)

    return processed_files, total_chunks


@app.on_event("startup")
async def startup_event() -> None:
    ensure_dirs()
    global _metadata, _index
    _metadata = load_metadata()
    _index = load_index()
    if _index is None and _metadata:
        rebuild_index_from_metadata()


@app.get("/status")
async def status() -> dict:
    count = len(_metadata)
    dim = _index.d if _index else None
    return {
        "chunks": count,
        "dim": dim,
        "model": EMBED_MODEL_NAME,
        "device": str(getattr(get_model(), "device", EMBED_DEVICE)),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "multi_gpu": EMBED_MULTI_GPU,
    }


@app.post("/ingest")
async def ingest(
    files: Optional[List[UploadFile]] = File(default=None),
    path: Optional[str] = Form(default=None),
) -> dict:
    ensure_dirs()

    targets: List[Path] = []
    if files:
        for upload in files:
            safe_name = Path(upload.filename or "upload").name
            dest = UPLOAD_DIR / safe_name
            content = await upload.read()
            dest.write_bytes(content)
            targets.append(dest)

    if path:
        targets.append(Path(path))

    if not targets:
        if DOCS_DIR.exists():
            targets.append(DOCS_DIR)
        else:
            raise HTTPException(status_code=400, detail="No files provided and docs directory not found.")

    total_files = 0
    total_chunks = 0
    for target in targets:
        files_count, chunks_count = await ingest_path(target)
        total_files += files_count
        total_chunks += chunks_count

    return {"files": total_files, "chunks": total_chunks}


@app.post("/search")
async def search(req: SearchRequest) -> dict:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty.")

    if _index is None or _index.ntotal == 0:
        return {"results": []}

    vector = embed_texts([req.query])
    scores, indices = _index.search(vector, req.top_k)

    results = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx < 0 or idx >= len(_metadata):
            continue
        item = _metadata[idx]
        results.append(
            {
                "score": float(score),
                "text": item["text"],
                "source_path": item.get("source_path"),
                "source_type": item.get("source_type"),
                "chunk_index": item.get("chunk_index"),
            }
        )

    return {"results": results}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8008,
        reload=False,
    )


if __name__ == "__main__":
    main()
