FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY configs ./configs

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

CMD ["python", "scripts/smoke_test.py"]
