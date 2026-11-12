# The shem handbook

## What shem is

shem runs language models as OCI containers. A model image holds the model's weights, one layer per block, and its session state. The same image runs with `shem run`, with `docker run` through shem's containerd shim, and on Kubernetes.

## Building a model image

Write a Shemfile, or a Dockerfile whose first line is `# syntax=ghcr.io/jgalego/shem/dockerfile:latest`. `FROM hf.co/ORG/NAME` imports a model from Hugging Face. `FROM hf.co/ORG/NAME-GGUF:Q4_K_M` imports one quantization of a GGUF repo. Build it with `shem build -t NAME DIRECTORY`, or with `docker build --platform linux/riscv32`.

The importer reads Llama, Mistral, Qwen, Gemma, Phi-3, OLMo, SmolLM3, Granite, Mixtral, OLMoE and Mamba checkpoints.

## Running a model

`shem run NAME "a prompt"` replies to one prompt. `docker run -i` takes turns on stdin. `--system` sets a system prompt, `--max-tokens` limits the reply, and `--stats` prints the token rate.

## Primed sessions

`WARM --until=prompt` reads the system prompt at build time and freezes every block's state after it into the image. A run starts from that state, so a long system prompt costs nothing at run time. This handbook is read that way.

## Saving a conversation

`docker commit` saves a live session into a new image, which continues the conversation where it stopped. `shem push --state` publishes a session as an artifact of its own, linked to its model, and `shem run --state REF IMAGE` continues it on any image of the same model.

## Sharing a model image

Install ORAS, log in to the registry, and run `shem push NAME ghcr.io/USER/NAME:TAG`. On another machine, `shem run ghcr.io/USER/NAME:TAG` pulls it. Images built on the same model share their weight layers in a registry.

## ModelPack

shem runs ModelPack artifacts as they are: `shem run REGISTRY/MODEL:TAG` pulls an artifact and keeps it as its image twin, over the same blobs. `shem push --modelpack` publishes a model image as a ModelPack artifact, for ModelPack's tools.

## Kubernetes

Install shem's RuntimeClass to run images as Pods. Any Pod can also mount a model image as an image volume and read `/model` as a Hugging Face checkpoint, which transformers loads as it is.

## Tools

`--tools DIRECTORY` offers the image's programs to the model. They run on a RISC-V CPU compiled into the model's weights.

## Getting help

Open an issue at github.com/JGalego/shem.
