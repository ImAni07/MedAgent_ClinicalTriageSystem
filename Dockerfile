FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt /tmp/requirements.docker.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.docker.txt

COPY server ./server
COPY models.py ./models.py
COPY client.py ./client.py
COPY inference.py ./inference.py
COPY ui.py ./ui.py
COPY openenv.yaml ./openenv.yaml
COPY config.yaml ./config.yaml
COPY params.yaml ./params.yaml
COPY README.md ./README.md
COPY start.sh ./start.sh
RUN chmod +x start.sh

RUN mkdir -p /app/artifacts/logs /app/artifacts/models /app/artifacts/reports

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["bash", "start.sh"]