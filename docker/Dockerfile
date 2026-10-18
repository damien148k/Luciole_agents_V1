# syntax=docker/dockerfile:1
# Image applicative Luciole (pipeline, watcher, test de fumée), une même
# recette pour amd64 (3080 Ti, A5000) et arm64 (GX10) :
#
#   docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile .
#
# Image de base épinglée par version ET par empreinte de liste
# multi-architecture ; uv installé depuis PyPI avec empreintes vérifiées
# (docker/uv-requirements.txt) ; dépendances verrouillées par uv.lock (extra « embed » :
# PyTorch de PyPI, CUDA 13 inclus sous Linux, pilote NVIDIA 580 ou plus).
# Aucun modèle dans l'image : ils sont montés dans /models (hors ligne).
FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

COPY docker/uv-requirements.txt /tmp/uv-requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/uv-requirements.txt
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app

# Dépendances d'abord (couche réutilisée tant que uv.lock ne change pas). Le
# cache de uv reste hors de l'image (montage de cache) : sinon chaque
# bibliothèque CUDA y serait deux fois.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra embed --no-install-project

COPY luciole ./luciole
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra embed

RUN useradd --create-home --uid 1000 luciole && mkdir -p /data /models \
    && chown luciole /data /models
USER luciole
ENV PATH=/opt/venv/bin:$PATH \
    LUCIOLE_MODELS_DIR=/models \
    LUCIOLE_DATA_DIR=/data \
    HF_HUB_OFFLINE=1 \
    LUCIOLE_QDRANT_URL=http://qdrant:6333 \
    LUCIOLE_OPENSEARCH_URL=http://opensearch:9200

CMD ["python", "-m", "luciole.smoke"]
