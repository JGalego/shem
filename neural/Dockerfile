# shem's neural side as an image: the model importer, the session driver and
# the model server, with PyTorch for the CPU. The BuildKit frontend runs
# `shem-neural-import` and `shem-neural-generate --warm` in it at build time.
#   docker build -t shem-neural neural
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /opt/shem-neural
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
ENV PATH=/opt/shem-neural/.venv/bin:$PATH
LABEL org.opencontainers.image.source="https://github.com/JGalego/shem"
ENTRYPOINT ["shem-neural-server"]
CMD ["--socket", "/run/shem/neural.sock"]
