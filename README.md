#### 启动命令（单GPU）
- ` uv run uvicorn main:app --host 127.0.0.1 --port 8008`
#### 启动命令（多GPU）
```python
export RAG_EMBED_MULTI_GPU=1
export RAG_EMBED_DEVICE=cuda
export RAG_EMBED_BATCH_SIZE=128
uvicorn main:app --host 0.0.0.0 --port 8008
```

#### powershell 测试
- 使用ssh连接本机的8008端口和服务器的8008端口:`ssh -L 8008:127.0.0.1:8008 用户名@ip`
- 使用默认docx下的文件测试解析并向量化pdf文件，powershell命令：` Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8008/ingest`


Autocad RAG Server (MinerU + FastAPI + FAISS)
=============================================

This service parses PDF/MD files with MinerU, chunks text, embeds with a sentence-transformer,
stores vectors in FAISS, and exposes `/ingest` and `/search` APIs.

Requirements
------------
- Ubuntu server with NVIDIA GPU(s)
- Docker + NVIDIA Container Toolkit installed

Build the Docker image
----------------------
From the project root:

```bash
docker build -t autocad-ragserver:latest .
```

Run the container (GPU enabled)
-------------------------------
Mount persistent folders for cache, docs, and index storage:

```bash
sudo docker run --gpus all -p 8008:8008 \
  -e RAG_EMBED_DEVICE=cuda \
  -e RAG_EMBED_MULTI_GPU=1 \
  -e RAG_EMBED_BATCH_SIZE=128 \
  -v /data/hf:/data/hf \
  -v /data/rag_store:/app/rag_store \
  -v /data/docs:/app/docs \
  autocad-ragserver:latest
```

- `/data/docs` is the default ingest source if you call `/ingest` without parameters.
- `/data/rag_store` persists the FAISS index and metadata.
- `/data/hf` caches model files.

Test the ingest API (external)
------------------------------
1) Ingest the mounted `/data/docs` directory:

```bash
curl -X POST http://<server-ip>:8008/ingest
```

2) Upload a file directly:

```bash
curl -X POST http://<server-ip>:8008/ingest \
  -F "files=@/path/to/your.pdf"
```

3) Check status:

```bash
curl http://<server-ip>:8008/status
```

Windows test (PowerShell 7+)
-------------------------
```powershell
Invoke-RestMethod -Method Post -Uri http://<server-ip>:8008/ingest
Invoke-RestMethod -Method Post -Uri http://<server-ip>:8008/ingest -Form @{ files = Get-Item "C:\path\file.pdf" }
Invoke-RestMethod -Uri http://<server-ip>:8008/status
```

Notes
-----
- If you only see `chunks: 0`, the files were not parsed or indexed. Check server logs.
- MinerU first run may download models; it can take several minutes.
