# Examples

Run from the repository root with shem and uv installed:

```sh
shem build -t smollm2 examples/smollm2
shem run smollm2                              # the image's own prompt
shem run smollm2 "Name three uses of clay."   # or yours
```

Model examples also include Dockerfiles for the [BuildKit frontend](../docs/shemfile.md#with-docker-build).

## Models

| Example | Model | What it shows |
|---|---|---|
| [`qwen3`](qwen3/Shemfile) | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Hugging Face import, sharded by block |
| [`qwen3.5`](qwen3.5/Shemfile) | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Hybrid attention and fixed-size Gated DeltaNet state |
| [`smollm2`](smollm2/Shemfile) | [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | Small CPU model with Llama's layout |
| [`granite-moe`](granite-moe/Shemfile) | [Granite 3.1 1B-A400M](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct) | A mixture of experts: 32 per block, 8 of them for each token |
| [`mamba`](mamba/Shemfile) | [Mamba-130M](https://huggingface.co/state-spaces/mamba-130m-hf) | Fixed-size 2.8 MB session; primed text continuation with `--stop` |
| [`llama-gguf`](llama-gguf/Shemfile) | [Llama 3.2 1B Instruct, Q4_K_M](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) | One quantization of a GGUF repo, dequantized at import |
| [`gemma3`](gemma3/Shemfile) | [Gemma 3 1B](https://huggingface.co/google/gemma-3-1b-it) | Gated checkpoint and sliding-window attention; requires model access and a Hugging Face token |

## Use cases

| Example | What it shows |
|---|---|
| [`golem`](golem/Shemfile) | A system prompt primed with `WARM --until=prompt` |
| [`handbook`](handbook/Shemfile) | A [document](handbook/handbook.md) loaded with `--system-file` and cached with `WARM` |
| [`tools`](tools/Shemfile) | RV32 tool calls on compiled CPU weights |
| [`modelpack`](modelpack/run.sh) | Import, session artifacts and export through a registry with modctl |
| [`kubernetes/model-volume.yaml`](kubernetes/model-volume.yaml) | A model image mounted as a checkpoint and loaded by transformers, without shem |
| [`kubernetes/hub-volume.yaml`](kubernetes/hub-volume.yaml) | A Hugging Face checkpoint mounted straight from the Hub, through `shem registry`, and loaded by transformers |

## Programs and machines

| Example | What it shows |
|---|---|
| [`hello`](hello/Shemfile) | A C program cross-compiled for RV32 in a Docker stage, shipped `FROM scratch` |
| [`warm-shell`](warm-shell/Shemfile) | A BusyBox shell started at build time and frozen just before it reads its script |
| [`linux`](linux/Shemfile) | Linux booted once at build time and frozen at its prompt |
| [`kubernetes/hello.yaml`](kubernetes/hello.yaml) | An RV32 program in a Pod, on shem's RuntimeClass |
