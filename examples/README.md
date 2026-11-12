# Examples

Each directory holds a Shemfile, and the model images also a Dockerfile for `docker build` through shem's BuildKit frontend. Build and run one with:

```sh
shem build -t smollm2 examples/smollm2
shem run smollm2                              # the image's own prompt
shem run smollm2 "Name three uses of clay."   # or yours
```

## Models

| Example | Model | What it shows |
|---|---|---|
| [`qwen3`](qwen3/Shemfile) | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | A model image: its weights from Hugging Face, one layer per block |
| [`qwen3.5`](qwen3.5/Shemfile) | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Gated DeltaNet layers, which keep a fixed-size memory in place of a growing cache |
| [`smollm2`](smollm2/Shemfile) | [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | Llama's layout, in a model small enough to reply quickly on any CPU |
| [`granite-moe`](granite-moe/Shemfile) | [Granite 3.1 1B-A400M](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct) | A mixture of experts: 32 per block, 8 of them for each token |
| [`mamba`](mamba/Shemfile) | [Mamba-130M](https://huggingface.co/state-spaces/mamba-130m-hf) | A state space model, whose session stays 2.8 MB however long the text, and a base model, which continues a pattern primed at build time; `--stop` ends its reply at the line's end |
| [`llama-gguf`](llama-gguf/Shemfile) | [Llama 3.2 1B Instruct, Q4_K_M](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) | One quantization of a GGUF repo, dequantized at import |
| [`gemma3`](gemma3/Shemfile) | [Gemma 3 1B](https://huggingface.co/google/gemma-3-1b-it) | A gated model, built with your Hugging Face token, and attention over sliding windows |

## Use cases

| Example | What it shows |
|---|---|
| [`golem`](golem/Shemfile) | A persona primed at build time: `WARM --until=prompt` reads the system prompt once and freezes every block's state into the image |
| [`handbook`](handbook/Shemfile) | An assistant that answers from a document: `--system-file` adds the [handbook](handbook/handbook.md) to the system prompt, and `WARM` reads all of it at build time, so a question costs only its own tokens |
| [`tools`](tools/Shemfile) | A model that calls programs, which run on a RISC-V CPU compiled into the image's weights |
| [`modelpack`](modelpack/run.sh) | A model's round trip through a registry: modctl packs a checkpoint as a ModelPack artifact, shem runs it and builds on it, publishes a primed session as an artifact linked to the model, and publishes the result as a ModelPack artifact that modctl extracts |
| [`kubernetes/model-volume.yaml`](kubernetes/model-volume.yaml) | A model image mounted in a Pod as an image volume and run by transformers, without shem: the image's `/model` is a Hugging Face checkpoint |

## Programs and machines

| Example | What it shows |
|---|---|
| [`hello`](hello/Shemfile) | A C program cross-compiled for RV32 in a Docker stage, shipped `FROM scratch` |
| [`warm-shell`](warm-shell/Shemfile) | A BusyBox shell started at build time and frozen just before it reads its script |
| [`linux`](linux/Shemfile) | Linux booted once at build time and frozen at its prompt |
| [`kubernetes/hello.yaml`](kubernetes/hello.yaml) | An RV32 program in a Pod, on shem's RuntimeClass |
