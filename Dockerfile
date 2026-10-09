# Roopiee's server, as it runs in production. Build from the repository root:
#   docker build -t roopiee .
#
# The folders keep their depth (/srv/apps/api/app) because the settings code finds the
# repository root by walking up from its own file.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

WORKDIR /srv/apps/api
COPY apps/api/pyproject.toml apps/api/uv.lock ./

# The WHO guidance search (app/rag) runs two models through PyTorch. PyTorch and the NVIDIA
# runtime are several gigabytes, so they are left out of this image. The server starts
# without them, and that one lookup is then not offered to Claude on phone calls here. Web
# sessions and the medicines lookup need none of them. The names are read from the lock
# file. To have the guidance search in this image too, delete the $(...) part of this
# command (tests/test_knowledge.py checks the server still starts with them left out).
RUN uv sync --frozen --no-dev --no-install-project \
    $(grep -oE '^name = "(nvidia-[a-z0-9-]+|torch|triton|sentence-transformers|transformers|tokenizers|scikit-learn|scipy)"' uv.lock \
      | cut -d'"' -f2 | sort -u | sed 's/^/--no-install-package /')

COPY apps/api/app ./app

RUN useradd --system --no-create-home roopiee
USER roopiee

EXPOSE 8080
# Behind Fly's proxy, so the forwarded scheme and host are trusted. /web/session builds the
# wss:// address the browser connects to from them.
CMD ["/srv/apps/api/.venv/bin/uvicorn", "app.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
