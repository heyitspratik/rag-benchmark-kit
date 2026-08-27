# syntax=docker/dockerfile:1.7

# Build the environment separately from the image that runs it, so uv, the compilers and
# the wheel cache never reach production.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies resolve from the lockfile alone, so this layer is reused across every
# source change and only rebuilds when the lockfile does.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --no-editable

COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# torch ships its headers, its test suite and its CUDA codegen even in the CPU build.
# None of it is reachable at inference time and it is about 145 MB.
RUN rm -rf /app/.venv/lib/python3.12/site-packages/torch/test \
           /app/.venv/lib/python3.12/site-packages/torch/include \
           /app/.venv/lib/python3.12/site-packages/torch/_inductor/codegen/cuda \
    && find /app/.venv -name '*.pyi' -delete

FROM python:3.12-slim AS runtime

# A compromised process should not be able to write to its own code.
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --home-dir /app --shell /usr/sbin/nologin appuser

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=prod \
    HF_HOME=/app/.cache/huggingface

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser alembic ./alembic
COPY --chown=appuser:appuser configs ./configs
COPY --chown=appuser:appuser data/eval ./data/eval

# Created here rather than left to Docker, so the named volumes mounted over them
# inherit appuser's ownership instead of being created root-owned and unwritable.
RUN mkdir -p /app/data/corpus /app/.cache/huggingface /app/results \
    && chown -R appuser:appuser /app/data /app/.cache /app/results

USER appuser
EXPOSE 8000

# Uses the interpreter that is already present rather than adding curl to the image for
# the sake of one probe.
HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=5 \
    CMD ["python", "-c", "import urllib.request as r,sys; sys.exit(0 if r.urlopen('http://127.0.0.1:8000/health/live', timeout=4).status==200 else 1)"]

CMD ["uvicorn", "rag_bench.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
