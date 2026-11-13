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
# Without uv's cache, which as root would leave the user's home unwritable.
# Gradio's reloader in a Space needs the spaces package, which Gradio Spaces
# come with and Docker Spaces don't
RUN uv sync --frozen --no-dev --extra gguf --no-cache \
    && uv pip install --no-cache --python .venv/bin/python "gradio[oauth]>=5,<7" "spaces>=0.43" \
    && chown -R user /opt/shem /home/user
USER user
# Gradio logs users in with Hugging Face's OAuth when SYSTEM is spaces and
# SPACE_ID names the Space, which Docker Spaces set but not SYSTEM; run
# elsewhere, without SPACE_ID, it mocks the login as before
ENV SYSTEM=spaces
WORKDIR /home/user/app
COPY --chown=user app.py convert.py ./
EXPOSE 7860
CMD ["python", "app.py"]
