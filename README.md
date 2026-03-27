#### 启动命令（单GPU）
- neo4j连接：`docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest`，可以在浏览器（服务器本地）访问`http://服务器ip:7474`来访问网页版neo4j
- 其中7687端口是用来python代码端访问
- 可以运行`docker ps -a`查看已停止运行的容器，然后重行运行docker start
- ` uv run uvicorn main:app --host 127.0.0.1 --port 8008`
- 多房间训练：` python ./RL/train.py --room_dir RL/room_gen/json  --curriculum`
#### 启动命令（多GPU）
```python
export RAG_EMBED_MULTI_GPU=1
export RAG_EMBED_DEVICE=cuda
export RAG_EMBED_BATCH_SIZE=128
uvicorn main:app --host 0.0.0.0 --port 8008
```

#### powershell 测试
- 使用ssh连接本机的8008端口和服务器的8008端口:`ssh -L 8008:127.0.0.1:8008 用户名@ip`
- 使用默认docx下的文件测试解析并向量化pdf文件，powershell命令：` Invoke-RestMethod -Method Post -Uri df`


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
curl -X POST http://127.0.0.1:8008/ingest
```

2) Upload a file directly:

```bash
curl -X POST http://127.0.0.1:8008/ingest \
  -F "files=@/path/to/your.pdf"
```

3) Check status:

```bash
curl http://127.0.0.1:8008/status
```

Windows test (PowerShell 7+)
-------------------------
访问ingest接口，使用默认模式
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8008/ingest   ## 默认
```
访问inget接口，自行上传文件
```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8008/ingest" `
  -Form @{
    mode  = "kg"
    files = Get-Item "C:\your\file.md"
  } ## 上传客户端的文件

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8008/ingest" `
  -Form @{
    mode  = "kg"
    files = @(
      Get-Item "C:\your\a.md"
      Get-Item "C:\your\b.md"
    )
  }  ## 上传多个文件
    
```
访问ingest接口，使用测试的文件
```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8008/ingest" `
  -Body @{
    path = "/home/chen/punchy/CAD_knowledge_service/docs"
    mode = "kg"
  }
```
访问status接口，查看后端服务的状态
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8008/status
```

Test kg/cypher接口
------------------------------
- 定义查询： 访问cypher接口，使用cypher直接查询
```powershell
$cypher = @'
MATCH (e:KGNode:DomainEntity)
WHERE e.name = $entity OR e.canonical_name = $entity
MATCH (r:KGNode:Requirement)-[:APPLIES_TO]->(e)
MATCH (r)-[:CONSTRAINS_METRIC]->(m:KGNode:Metric)
MATCH (r)-[:HAS_VALUE_SPEC]->(v:KGNode:ValueSpec)
OPTIONAL MATCH (r)-[:UNDER_CONDITION]->(c:KGNode:Condition)
WHERE m.name CONTAINS "照度标准值"
    OR m.canonical_name = "照度标准值"
    OR m.name = "照度"
RETURN
  e.name AS entity,
  c.text AS condition,
  m.name AS metric,
  v.raw_text AS raw_value,
  v.value AS value,
  v.unit AS unit,
  r.table_caption AS table_caption,
  r.page_no AS page_no
ORDER BY c.text, r.page_no
LIMIT $limit
'@

$body = @{
    cypher = $cypher
    params = @{
        entity = "起居室"
    }
    top_k = 20
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8008/kg/cypher" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```
访问llm-query接口，使用自然语言查询
```powershell
$body = @{
    question = "起居室的照度标准是多少？"
    top_k = 20
    synthesize_answer = $true
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8008/kg/llm-query" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body | ConvertTo-Json -Depth 20
```

Notes
-----
- If you only see `chunks: 0`, the files were not parsed or indexed. Check server logs.
- MinerU first run may download models; it can take several minutes.
