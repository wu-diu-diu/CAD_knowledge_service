#### 启动命令
- ` uv run uvicorn main:app --host 127.0.0.1 --port 8008`

#### powershell 测试
- 使用默认docx下的文件测试解析并向量化pdf文件，powershell命令：` Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8008/ingest`