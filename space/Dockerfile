# The shem my repo Space: shem and its neural package, with transformers to
# compare imports against, built from shem's source at SHEM_REF, and the app.
#   docker build -t shem-my-repo space
#   docker run -p 7860:7860 -e HF_TOKEN shem-my-repo
FROM rust:1.94-bookworm AS shem
ARG SHEM_REF=main
WORKDIR /src
RUN git clone https://github.com/JGalego/shem . && git checkout "$SHEM_REF" \
    && cargo build --release --locked -p shem

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
# Spaces run as user 1000. shem runs the neural package from its source, in
# the environment built here, which UV_NO_SYNC keeps as it is
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    UV_NO_SYNC=1 SHEM_NEURAL_PROJECT=/opt/shem/neural SHEM_BIN=/usr/local/bin/shem \
    PATH=/opt/shem/neural/.venv/bin:$PATH
COPY --from=shem /src/target/release/shem /usr/local/bin/shem
COPY --from=shem /src/neural /opt/shem/neural
WORKDIR /opt/shem/neural
RUN uv sync --frozen --no-dev --extra gguf \
    && uv pip install --python .venv/bin/python "gradio[oauth]>=5,<7" \
    && chown -R user /opt/shem
USER user
WORKDIR /home/user/app
COPY --chown=user app.py convert.py ./
EXPOSE 7860
CMD ["python", "app.py"]
