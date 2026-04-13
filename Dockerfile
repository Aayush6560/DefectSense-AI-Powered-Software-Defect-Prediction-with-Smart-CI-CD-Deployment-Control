FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLASK_ENV=production \
    FLASK_DEBUG=false

WORKDIR /app

ARG REQUIREMENTS_FILE=requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements*.txt ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE}

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --no-create-home appuser

COPY --chown=appuser:appuser . .

RUN mkdir -p data ml/models && \
    chown -R appuser:appuser data ml/models

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -fsS http://127.0.0.1:5000/health || exit 1

CMD ["gunicorn", \
    "--workers", "2", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "300", \
     "--graceful-timeout", "30", \
     "--worker-class", "sync", \
     "--worker-connections", "100", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "run:app"]