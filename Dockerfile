FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m pip install --upgrade pip

RUN python3.12 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3.12 -m pip install -r /app/requirements.txt

COPY . /app

ENV HF_HOME=/data/hf
ENV TRANSFORMERS_CACHE=/data/hf
ENV RAG_EMBED_DEVICE=cuda

EXPOSE 8008

CMD ["python3.12", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008"]
