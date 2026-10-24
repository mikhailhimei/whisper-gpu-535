FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    pkg-config \
    build-essential \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# 1. Обновляем pip, чтобы избежать AssertionError
# 2. Устанавливаем whisper
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir "wyoming-faster-whisper==3.8.1" "ctranslate2==4.4.0"

COPY prompt_runtime.py default-initial-prompt.txt /app/

EXPOSE 10300 10301

ENTRYPOINT ["python3", "/app/prompt_runtime.py", "--uri", "tcp://0.0.0.0:10300", "--data-dir", "/data", "--download-dir", "/data"]
