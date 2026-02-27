import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from mineru.cli.gradio_app import to_markdown
from utils.transfer import process_markdown_file, fix_md_headings, process_images_in_markdown
from kg.llm_query import llm_query_graph
from kg.pipeline import ingest_markdown_to_kg, kg_already_ingested
from kg.query import query_graph_requirements, kg_status
from kg.store_neo4j import close_driver as close_neo4j_driver, neo4j_enabled, get_driver as get_neo4j_driver, run_cypher_query

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
STORE_DIR = BASE_DIR / "rag_store"
KG_STORE_DIR = BASE_DIR / "kg_store"
UPLOAD_DIR = STORE_DIR / "uploads"
INDEX_PATH = STORE_DIR / "index.faiss"
META_PATH = STORE_DIR / "metadata.json"
CAD_SERVICE_DIR = BASE_DIR / "cad_images_service"

if CAD_SERVICE_DIR.exists():
    cad_service_path = str(CAD_SERVICE_DIR)
    if cad_service_path not in sys.path:
        sys.path.insert(0, cad_service_path)

from preprocess.cad_service import CADParams, CADResponse, cad_service
from preprocess.coordinate_converter import DEFAULT_CAD_PARAMS
from preprocess.logger import get_logger as get_cad_logger

EMBED_MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
EMBED_DEVICE = os.getenv("RAG_EMBED_DEVICE", "cuda")
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "64"))
EMBED_MULTI_GPU = os.getenv("RAG_EMBED_MULTI_GPU", "0").lower() in ("1", "true", "yes")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
MINERU_BACKEND = os.getenv("RAG_MINERU_BACKEND", "pipeline")
MINERU_LANG = os.getenv("RAG_MINERU_LANG", "ch")
MINERU_OCR = os.getenv("RAG_MINERU_OCR", "false").lower() in ("1", "true", "yes")
# 设置neo4j环境变量
os.environ['KG_NEO4J_ENABLED'] = '1'
os.environ['KG_NEO4J_URI'] = 'bolt://127.0.0.1:7687'
os.environ['KG_NEO4J_USER'] = 'neo4j'
os.environ['KG_NEO4J_PASSWORD'] = 'password'
os.environ['KG_NEO4J_DATABASE'] = 'neo4j' 
# 设置cad房间离散化步骤环境变量
os.environ["CAD_LIGHTING_USE_LLM"] = "1"
os.environ["CAD_LIGHTING_LLM_PROVIDER"] = "deepseek"
os.environ["CAD_LIGHTING_LLM_MODEL"] = "deepseek-chat"

MINERU_MAX_PAGES = int(os.getenv("RAG_MINERU_MAX_PAGES", "1000"))
cad_logger = get_cad_logger("cad_api")
_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_metadata: List[dict] = []
CAD_RESULT_CACHE_SECONDS = float(os.getenv("CAD_RESULT_CACHE_SECONDS", "60"))
_cad_recent_results: dict[str, tuple[float, dict]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    global _metadata, _index
    _metadata = load_metadata()
    _index = load_index()
    if _index is None and _metadata:
        rebuild_index_from_metadata()
    if neo4j_enabled():
        try:
            get_neo4j_driver()
        except Exception as e:
            cad_logger.error(f"neo4j init failed: {e}")
    yield
    close_neo4j_driver()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class KgQueryRequest(BaseModel):
    query: str = ""
    entity: Optional[str] = None
    metric: Optional[str] = None
    source_id: Optional[str] = None
    top_k: int = 10


class CypherQueryRequest(BaseModel):
    cypher: str
    params: Optional[dict] = None
    top_k: int = 200


class KgLlmQueryRequest(BaseModel):
    question: str
    entity: Optional[str] = None
    metric: Optional[str] = None
    source_id: Optional[str] = None
    top_k: int = 20
    model: Optional[str] = None
    synthesize_answer: bool = True


def _cad_response_to_payload(obj) -> dict:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}


def _cad_params_to_stable_dict(cad_params_obj: Optional[CADParams]) -> dict:
    if cad_params_obj is None:
        return {}
    if hasattr(cad_params_obj, "model_dump"):
        return cad_params_obj.model_dump()
    return cad_params_obj.dict()


def _build_cad_result_cache_key(file_contents: List[bytes], cad_params_obj: Optional[CADParams]) -> str:
    h = hashlib.sha256()
    content_hashes = [hashlib.sha256(content).hexdigest() for content in file_contents]
    content_hashes.sort()
    h.update(json.dumps(content_hashes, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    stable_params = _cad_params_to_stable_dict(cad_params_obj)
    h.update(json.dumps(stable_params, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def _cad_result_cache_get(key: str) -> Optional[dict]:
    now = time.time()
    expired = [
        cache_key
        for cache_key, (ts, _) in _cad_recent_results.items()
        if now - ts > CAD_RESULT_CACHE_SECONDS
    ]
    for cache_key in expired:
        _cad_recent_results.pop(cache_key, None)
    entry = _cad_recent_results.get(key)
    if entry is None:
        return None
    _, payload = entry
    return payload


def _cad_result_cache_set(key: str, status_code: int, content: dict) -> None:
    _cad_recent_results[key] = (time.time(), {"status_code": int(status_code), "content": content})


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    cad_logger.info(f"request started: {request.method} {request.url}")
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    cad_logger.info(
        f"request completed: {request.method} {request.url} "
        f"status={response.status_code} time={process_time:.4f}s"
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    cad_logger.error(f"unhandled exception: {request.method} {request.url} - {exc}")
    cad_logger.error(traceback.format_exc())
    error_detail = {
        "error": "Internal Server Error",
        "message": str(exc),
        "path": str(request.url),
        "method": request.method,
        "timestamp": time.time(),
    }
    if app.debug:
        error_detail["traceback"] = traceback.format_exc()
    return JSONResponse(status_code=500, content=error_detail)


def ensure_dirs() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    KG_STORE_DIR.mkdir(parents=True, exist_ok=True)


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


def _extract_inline_text(items: Optional[List[dict]]) -> str:
    if not items:
        return ""
    parts: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in ("text", "inline_equation"):
            parts.append(item.get("content", ""))
    return "".join(parts).strip()


def _render_block(block: dict, asset_root: Path) -> Optional[str]:
    block_type = block.get("type")
    content = block.get("content", {})

    if block_type in ("page_header", "page_footer", "page_number"):
        return None

    if block_type == "title":
        level = int(content.get("level", 1))
        level = min(max(level, 1), 6)
        title = _extract_inline_text(content.get("title_content", []))
        return f"{'#' * level} {title}".strip()

    if block_type == "paragraph":
        text = _extract_inline_text(content.get("paragraph_content", []))
        return text or None

    if block_type == "list":
        items = content.get("list_items", [])
        lines: List[str] = []
        for item in items:
            item_text = _extract_inline_text(item.get("item_content", []))
            if item_text:
                lines.append(f"- {item_text}")
        return "\n".join(lines) if lines else None

    if block_type == "table":
        caption_text = _extract_inline_text(content.get("table_caption", []))
        html = content.get("html")
        parts: List[str] = []
        if caption_text:
            parts.append(caption_text)
        if html:
            parts.append(html)
        return "\n".join(parts) if parts else None

    if block_type == "image":
        image_source = content.get("image_source", {}).get("path")
        if not image_source:
            return None
        image_path = (asset_root / image_source).resolve()
        caption_text = _extract_inline_text(content.get("image_caption", []))
        lines = [f"![]({image_path})"]
        if caption_text:
            lines.append(caption_text)
        return "\n".join(lines)

    if block_type == "equation_interline":
        math = content.get("math_content")
        if not math:
            return None
        return f"$$\n{math}\n$$"

    return None


def _find_latest_content_list(pdf_path: Path, suffix: str) -> Optional[Path]:
    output_root = BASE_DIR / "output"
    if not output_root.exists():
        return None
    pattern = f"{pdf_path.stem}*{suffix}.json"
    candidates = list(output_root.rglob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_markdown_with_pages(content_list_path: Path) -> str:
    with content_list_path.open("r", encoding="utf-8") as f:
        pages = json.load(f)

    asset_root = content_list_path.parent
    output_lines: List[str] = []
    for page_idx, blocks in enumerate(pages):
        output_lines.append(f"[PAGE:{page_idx + 1}]")
        for block in blocks:
            rendered = _render_block(block, asset_root)
            if rendered:
                output_lines.append(rendered)
        output_lines.append("")
    return "\n\n".join(output_lines).strip() + "\n"


def build_markdown_with_pages_from_flat(content_list_path: Path) -> str:
    with content_list_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    pages: dict[int, List[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        page_idx = item.get("page_idx")
        text = item.get("text", "")
        if page_idx is None or not text:
            continue
        pages.setdefault(int(page_idx), []).append(text)

    output_lines: List[str] = []
    for page_idx in sorted(pages.keys()):
        output_lines.append(f"[PAGE:{page_idx + 1}]")
        output_lines.append(" ".join(pages[page_idx]).strip())
        output_lines.append("")
    return "\n\n".join(output_lines).strip() + "\n"


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
            vectors = model.encode(
                texts,
                batch_size=EMBED_BATCH_SIZE,
                normalize_embeddings=True,
                pool=pool,
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
    content_list_v2_path = _find_latest_content_list(path, "_content_list_v2")
    if content_list_v2_path:
        md_content = build_markdown_with_pages(content_list_v2_path)
    else:
        content_list_path = _find_latest_content_list(path, "_content_list")
        if content_list_path:
            md_content = build_markdown_with_pages_from_flat(content_list_path)
    output_dir = BASE_DIR / "transfer_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{path.stem}_{timestamp}.md"

    output_path.write_text(md_content, encoding="utf-8")
    fix_md_headings(output_path)

    processed = output_path.read_text(encoding="utf-8", errors="ignore")
    processed = process_markdown_file(processed)
    processed = process_images_in_markdown(processed, output_path)
    output_path.write_text(processed, encoding="utf-8")

    return processed


async def load_text_from_file(path: Path) -> Tuple[str, str]:
    raw_text, clean_text, source_type = await load_source_contents(path)
    return clean_text, source_type


async def load_source_contents(path: Path) -> Tuple[str, str, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        md = await parse_pdf(path)
        return md, clean_markdown(md), "pdf"
    if suffix == ".md":
        content = path.read_text(encoding="utf-8", errors="ignore")
        return content, clean_markdown(content), "md"
    raise ValueError(f"Unsupported file type: {suffix}")


async def ingest_path(path: Path, mode: str = "vector") -> dict:
    if path.is_dir():
        files = [
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in (".pdf", ".md")
        ]
    else:
        files = [path]

    if not files:
        return {
            "files": 0,
            "chunks": 0,
            "kg_files": 0,
            "kg_nodes": 0,
            "kg_edges": 0,
            "kg_requirements": 0,
        }

    if mode not in {"vector", "kg", "both"}:
        raise ValueError(f"Unsupported ingest mode: {mode}")

    existing = {item.get("source_id") for item in _metadata}
    total_chunks = 0
    processed_files = 0
    kg_files = 0
    kg_nodes = 0
    kg_edges = 0
    kg_requirements = 0

    for file_path in files:
        source_id = file_sha256(file_path)

        try:
            raw_text, vector_text, source_type = await load_source_contents(file_path)
        except Exception:
            continue

        if mode in {"vector", "both"} and source_id not in existing:
            chunks = chunk_text(vector_text)
            if chunks:
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

        should_ingest_kg = False
        if mode in {"kg", "both"}:
            local_kg_exists = kg_already_ingested(KG_STORE_DIR, source_id)
            should_ingest_kg = (not local_kg_exists) or neo4j_enabled()

        if should_ingest_kg:
            kg_stats = ingest_markdown_to_kg(
                md_text=raw_text,
                store_dir=KG_STORE_DIR,
                source_id=source_id,
                source_path=str(file_path),
                source_type=source_type,
            )
            kg_files += 1
            kg_nodes += kg_stats.get("nodes", 0)
            kg_edges += kg_stats.get("edges", 0)
            kg_requirements += kg_stats.get("requirements", 0)

    if mode in {"vector", "both"} and processed_files > 0:
        save_index(_index)
        save_metadata(_metadata)

    return {
        "files": processed_files,
        "chunks": total_chunks,
        "kg_files": kg_files,
        "kg_nodes": kg_nodes,
        "kg_edges": kg_edges,
        "kg_requirements": kg_requirements,
    }


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


@app.get("/kg/status")
async def kg_status_endpoint() -> dict:
    ensure_dirs()
    stats = kg_status(KG_STORE_DIR)
    neo4j_ok = False
    neo4j_error = None
    if neo4j_enabled():
        try:
            neo4j_ok = get_neo4j_driver() is not None
        except Exception as e:
            neo4j_error = str(e)
    return {
        "store_dir": str(KG_STORE_DIR),
        "neo4j_enabled": neo4j_enabled(),
        "neo4j_ok": neo4j_ok if neo4j_enabled() else None,
        "neo4j_error": neo4j_error,
        **stats,
    }


@app.post("/ingest")
async def ingest(
    files: Optional[List[UploadFile]] = File(default=None),
    path: Optional[str] = Form(default=None),
    mode: str = Form(default="vector"),
) -> dict:
    ensure_dirs()
    mode = (mode or "vector").strip().lower()
    if mode not in {"vector", "kg", "both"}:
        raise HTTPException(status_code=400, detail="Invalid mode. Use vector, kg, or both.")

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

    total_stats = {
        "files": 0,
        "chunks": 0,
        "kg_files": 0,
        "kg_nodes": 0,
        "kg_edges": 0,
        "kg_requirements": 0,
    }
    for target in targets:
        try:
            stats = await ingest_path(target, mode=mode)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        for key in total_stats:
            total_stats[key] += int(stats.get(key, 0))

    return {"mode": mode, **total_stats}


@app.post("/kg/query")
async def kg_query(req: KgQueryRequest) -> dict:
    ensure_dirs()
    if not req.query.strip() and not (req.entity or "").strip() and not (req.metric or "").strip():
        raise HTTPException(status_code=400, detail="Provide query, entity, or metric.")
    return query_graph_requirements(
        store_dir=KG_STORE_DIR,
        query=req.query,
        entity=req.entity,
        metric=req.metric,
        source_id=req.source_id,
        top_k=req.top_k,
    )


@app.post("/kg/cypher")
async def kg_cypher(req: CypherQueryRequest) -> dict:
    ensure_dirs()
    try:
        return run_cypher_query(
            cypher=req.cypher,
            params=req.params,
            top_k=req.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/kg/llm-query")
async def kg_llm_query(req: KgLlmQueryRequest) -> dict:
    ensure_dirs()
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
    try:
        return llm_query_graph(
            question=req.question,
            entity=req.entity,
            metric=req.metric,
            source_id=req.source_id,
            top_k=req.top_k,
            model=req.model,
            synthesize_answer=req.synthesize_answer,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


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
        print(
            f"[search] score={float(score):.4f} "
            f"source={item.get('source_path')} "
            f"chunk={item.get('chunk_index')} "
            f"text={item.get('text')[:500]}"
        )
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


@app.get("/")
async def root() -> dict:
    return {
        "service": "CAD and RAG service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "rag_ingest": "/ingest",
            "rag_search": "/search",
            "rag_status": "/status",
            "kg_query": "/kg/query",
            "kg_cypher": "/kg/cypher",
            "kg_llm_query": "/kg/llm-query",
            "kg_status": "/kg/status",
            "cad_process": "/upload-and-process",
            "cad_health": "/health",
            "cad_default_params": "/default-cad-params",
        },
    }


@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "cad-rag-service",
    }


@app.get("/default-cad-params")
async def get_default_cad_params() -> dict:
    return {
        "default_cad_params": DEFAULT_CAD_PARAMS,
        "description": "default CAD parameter values",
    }


@app.post("/upload-and-process", response_model=CADResponse)
async def upload_and_process_cad(
    files: List[UploadFile] = File(..., description="PNG files"),
    cad_params: Optional[str] = Form(None, description="CAD params as JSON string"),
):
    cad_logger.info(f"upload request received: file_count={len(files)}")

    for file in files:
        filename = file.filename or ""
        if not filename.lower().endswith(".png"):
            raise HTTPException(
                status_code=400,
                detail=f"unsupported file type: {filename}, only PNG is supported",
            )

    try:
        file_contents: List[bytes] = []
        filenames: List[str] = []
        for file in files:
            content = await file.read()
            filename = file.filename or "upload.png"
            if not content:
                raise HTTPException(status_code=400, detail=f"file {filename} is empty")
            file_contents.append(content)
            filenames.append(filename)
            cad_logger.info(f"read file: {filename}, size={len(content)} bytes")

        cad_params_obj = None
        if cad_params:
            try:
                parsed_params = json.loads(cad_params)
                if not isinstance(parsed_params, dict):
                    raise HTTPException(
                        status_code=400,
                        detail="invalid cad_params: JSON body must be an object",
                    )
                cad_params_obj = CADParams(**parsed_params)
                cad_logger.info("using custom CAD parameters")
            except json.JSONDecodeError as exc:
                cad_logger.warning(f"failed to parse CAD parameters JSON: {exc}")
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "invalid cad_params JSON format: "
                        f"{exc.msg} (line {exc.lineno}, column {exc.colno})"
                    ),
                )
            except HTTPException:
                raise
            except Exception as exc:
                cad_logger.warning(f"failed to validate CAD parameters: {exc}")
                raise HTTPException(
                    status_code=400,
                    detail=f"invalid cad_params values: {exc}",
                )
        else:
            cad_logger.info("using default CAD parameters")

        cache_key = _build_cad_result_cache_key(file_contents, cad_params_obj)
        cached_response = _cad_result_cache_get(cache_key)
        if cached_response is not None:
            cad_logger.info("CAD result cache hit within 60s: reuse previous response")
            return JSONResponse(
                status_code=int(cached_response["status_code"]),
                content=cached_response["content"],
            )
        ## 真正处理图片的地方
        result = cad_service.process_uploaded_files(file_contents, filenames, cad_params_obj)
        cad_logger.info(
            f"CAD processing finished: success={result.processed_images}/{result.total_images}"
        )

        if not result.success:
            if result.total_images == 0:
                raise HTTPException(status_code=400, detail=result.message)
            payload = _cad_response_to_payload(result)
            _cad_result_cache_set(cache_key, 207, payload)
            return JSONResponse(status_code=207, content=payload)

        payload = _cad_response_to_payload(result)
        _cad_result_cache_set(cache_key, 200, payload)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        cad_logger.error(f"unexpected CAD processing error: {exc}")
        cad_logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"internal error while processing uploaded files: {exc}",
        )


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
