---
title: shem my repo
emoji: 🗿
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
hf_oauth: true
hf_oauth_scopes:
  - read-repos
  - write-repos
  - manage-repos
license: apache-2.0
short_description: Turn a checkpoint into a shem image you can pull and run
---

# shem my repo

A Space that turns a Hugging Face checkpoint into a [shem](https://github.com/JGalego/shem) model image, compares it with transformers, and pushes it to a repo of your own as an OCI image layout. Run the result with `shem run hf.co/YOU/NAME-shem`, or pull it with any OCI client through `shem registry`.

The Space runs [`convert.py`](convert.py), which works on its own too:

```sh
uv run --project neural python space/convert.py Qwen/Qwen3-0.6B --system "You are a golem." --push YOU/qwen3-0.6b-shem
```

## shem replay

[`replay.py`](replay.py) and [`replay_app.py`](replay_app.py) are another Space's, [shem replay](https://huggingface.co/spaces/jgalego/shem-replay), which checks a receipt that `shem run --receipt` wrote by running it again. Its Dockerfile takes them from shem's source, so they need no upload.

## Deploying

The Space's files are this directory. [The Dockerfile](Dockerfile) builds shem from source at `SHEM_REF` (default `main`), so push the shem commit the Space needs before building it. On a free CPU Space, it converts checkpoints whose weights take up to 8 GB, primes system prompts up to 2B parameters, and compares models with transformers up to 1.5B. Set `SHEM_MAX_BYTES`, `SHEM_MAX_PRIMED` and `SHEM_MAX_VERIFIED` to raise these on larger hardware.

```sh
hf upload --repo-type space YOU/shem-my-repo space . --exclude "__pycache__/*" --exclude ".ruff_cache/*"
```
