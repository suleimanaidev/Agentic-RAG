FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 nexusai \
    && mkdir -p /app/qdrant_store /home/nexusai/.cache \
    && chown -R nexusai:nexusai /app /home/nexusai/.cache

COPY --chown=nexusai:nexusai app.py evaluate_rag.py ./
COPY --chown=nexusai:nexusai .streamlit ./.streamlit
USER nexusai

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
