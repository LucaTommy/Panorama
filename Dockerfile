FROM python:3.11-slim

# OpenCV runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY panorama.py server.py ./
COPY web ./web

ENV JOBS_ROOT=/data/jobs \
    CORS_ORIGINS=* \
    WORKER_CONCURRENCY=1 \
    MAX_UPLOAD_BYTES=1073741824
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
