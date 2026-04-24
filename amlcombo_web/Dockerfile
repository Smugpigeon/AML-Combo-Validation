# syntax=docker/dockerfile:1.6
#
# Multi-purpose image: same base runs the web app, the Celery worker, and
# one-shot scripts (init DB, create admin). Built once, selected via CMD.
#
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System deps:
#   - build-essential + libpq for psycopg
#   - git in case combo_val needs to be installed from a repo
#   - chromium for headless PDF rendering (pandoc chain uses it)
#   - pandoc (render Markdown → HTML/PDF)
#   - fonts-noto-cjk (Chinese rendering in PDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libpq-dev git \
      chromium pandoc fonts-noto-cjk \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV CHROME_BIN=/usr/bin/chromium

WORKDIR /app

# Install combo_val (the kit) from the parent repo as an editable install.
# Build context in docker-compose is the repo root, so combo_val sources
# are at ./src/combo_val relative to build context.
COPY pyproject.toml README.md /app/combo_val_pkg/
COPY src/ /app/combo_val_pkg/src/
RUN pip install /app/combo_val_pkg

# Install amlcombo_web itself
COPY amlcombo_web/pyproject.toml /app/amlcombo_web/
RUN pip install fastapi uvicorn[standard] jinja2 python-multipart \
                bcrypt pyjwt cryptography \
                sqlalchemy "psycopg[binary]" celery redis httpx \
                "pydantic[email]" pandas numpy joblib scipy \
                torch scikit-learn matplotlib pypdfium2 openpyxl

COPY amlcombo_web/app /app/amlcombo_web/app
COPY amlcombo_web/scripts /app/amlcombo_web/scripts

WORKDIR /app/amlcombo_web

EXPOSE 8000

# Default: run the FastAPI web server. Overridden by docker-compose for workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
