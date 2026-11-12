# Model images

A model image packages a language model's weights, tokenizer and optional session state in OCI layers. `shem run` starts or resumes a text-generation session; `shem push` and `shem pull` transfer the image through a registry.

```sh
shem build -t qwen3 examples/qwen3             # Qwen3-0.6B from Hugging Face, one layer per block
shem run qwen3 "What is a golem?"              # a session: the prompt in, the reply out
shem push qwen3 ghcr.io/your-user/qwen3:0.6b   # to a registry
shem run ghcr.io/your-user/qwen3:0.6b "Olá"    # elsewhere: pulled on first use, then run
```

`docker build` builds model images too, through shem's BuildKit frontend: add `# syntax=ghcr.io/jgalego/shem/dockerfile:latest` as the first line ([shemfile.md](shemfile.md#with-docker-build)). The [Qwen3](../examples/qwen3/Dockerfile), [Qwen3.5](../examples/qwen3.5/Dockerfile) and [golem](../examples/golem/Dockerfile) examples include Dockerfiles.

## Shemfile

```dockerfile
FROM hf.co/Qwen/Qwen3.5-0.8B
ENTRYPOINT ["shem:generate", "--max-tokens", "128"]
CMD ["Introduce yourself in one sentence."]
```

`FROM hf.co/ORG/NAME[:REVISION]` imports a Hugging Face checkpoint. The base sets `ENTRYPOINT ["shem:generate"]`, the way `FROM python` brings `python3`, and an image can override it with options.

ENTRYPOINT and CMD keep their Docker meanings: the entrypoint is what runs, and CMD is its default input, which `shem run` arguments replace.

| Image | ENTRYPOINT | CMD | `shem run image …` |
|---|---|---|---|
| A program (`FROM scratch`) | A program in the image, run on the vCPU | Its default arguments | Replace the arguments |
| A model | A driver: `["shem:generate", "--max-tokens", "128"]` | The default prompt | Your prompt |

A `shem:` entrypoint names a host-side driver rather than a guest executable. `shem:generate` options include `--max-tokens N` (default 256) and `--system TEXT`. Decoding is greedy; results are repeatable for the same model, session, prompt and device.

## Sessions as layers

`WARM --until=prompt` primes a session at build time and freezes it into the image ([`examples/golem`](../examples/golem/Shemfile)):

```dockerfile
FROM hf.co/Qwen/Qwen3.5-0.8B
ENTRYPOINT ["shem:generate", "--system", "You are Shem, a golem made of clay and brought to life by a word."]
CMD ["Who are you?"]
WARM --until=prompt
```

The builder runs the model over the conversation up to where the user's prompt goes, here the system prompt, and saves every block's state in the image's state layer: the attention layers' key and value caches, and the Gated DeltaNet layers' convolution inputs and memory matrices. `shem run` continues that session, so the system prompt is never read again, and the reply is exactly what an unprimed session would give; a test checks this for both models.

Attention caches grow with the token count. Gated DeltaNet's convolution state and memory matrices are fixed-size. Qwen3.5 combines both block types, so its total session state still grows with its attention caches.

The session is saved as `session.safetensors`, every block's state by block number and name, and `session.json`, the tokens it has read. `shem run` unpacks them into `/.shem`, which is also where `shem save` puts them for `docker load`.

## The model as files

The importer (`shem_neural.model.importers`) writes the model into `/model`:

- `shem.json`, the manifest: the embedding, the blocks in order, and the head, each with its type, its parameters and its tensors, plus the tokenizer;
- `blocks/NNN-TYPE.safetensors`: one weights file per block, holding that block's tensors with the checkpoint's own names and types;
- `embed.safetensors` and `head.safetensors`;
- `model.safetensors.index.json`, the index that says which file holds each tensor;
- the checkpoint's `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`, and its licence and notices.

Each weights file is its own image layer, so images built on the same model share those blobs in stores and registries. A model with more than 95 weights files shares layers between neighbouring blocks, since overlay filesystems stack about 128 layers at most.

The directory is also a Hugging Face checkpoint, sharded by block: the index and the tensor names are the ones transformers expects, so `AutoModelForCausalLM.from_pretrained("/model")` loads it and gives the original's logits, which a test checks. A block in `shem.json` names its tensors as the checkpoint does, and the index finds their files, so whether the files follow the blocks or the checkpoint's own shards is a choice of layout rather than of format. That manifest format is `shem.model/v2`; `shem.model/v1` models, whose blocks each name one file holding their tensors under the block type's names, still run.

The format knows nothing about any one family. A model is a residual stream: an embedding, then blocks that each read the stream and add to it, then a head. A decoder layer is two blocks, a token mixer and an MLP.

| Block type | What it does | Its state in a session |
|---|---|---|
| `attention` | Grouped-query attention, with rotary positions on all or part of each head (scaled linearly or as Llama 3 does), optional biases, RMSNorm on queries and keys, a sliding window and an output gate | The key and value cache, which grows with the session |
| `gated_deltanet` | Gated DeltaNet: a causal depthwise convolution, then a matrix memory per head, written with the gated delta rule | The convolution's last inputs and the memory matrices, a fixed size however long the session |
| `mlp` | A SwiGLU feed-forward block, with optional biases | None |

Llama, Mistral, Qwen2, Qwen3 and Qwen3.5 share Llama's layout, so one importer reads them all. It turns on what each checkpoint's config and tensors call for: projection biases, norms on queries and keys, sliding windows, rotary scaling and output gates. It refuses what the blocks can't run yet, such as YaRN scaling or an activation other than SiLU, rather than run it wrong. A family with another layout needs an importer of its own that maps its tensors onto these types. A new kind of layer, such as Mamba or a mixture of experts, is a new block type in [`blocks.py`](../neural/src/shem_neural/model/blocks.py), registered by name. Neither changes the image format.

[`tests/test_importers.py`](../neural/tests/test_importers.py) builds a tiny random checkpoint of each family with transformers, imports it, and compares logits, in one pass and token by token, with no downloads.

## Imported models

| Model | Blocks | Image | Verified |
|---|---|---|---|
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 28 attention, 28 MLP | 59 layers, 892 MB | Logits within 1e-3 of transformers, and the same 24 greedy tokens |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 18 Gated DeltaNet, 6 attention with output gates and partial rotary positions, 24 MLP | 51 layers | The same, against transformers' Qwen3.5 |
| [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 32 attention, 32 MLP | 67 layers | The same, against transformers' Llama |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 24 attention with query, key and value biases, 24 MLP | 51 layers | The same, against transformers' Qwen2 |

Qwen3.5's vision encoder and multi-token prediction layer are left out, so its import is a text model. The others' imports load in transformers with the original's logits, exactly. [`tests/test_model.py`](../neural/tests/test_model.py) does these comparisons; it downloads the checkpoints, so it runs with `SHEM_MODEL_TESTS=1`. Mistral, Llama 3's rotary scaling and every family's other options are checked on tiny random checkpoints instead.

## Sessions under Docker

Model containers are live sessions, held by the model server (`shem-neural-server`) under the container's ID, so Docker's commands mean something for them:

| Command | What it does |
|---|---|
| `docker run IMAGE "prompt"` | Answers the prompt, then exits |
| `docker run -i IMAGE` | Takes each line of stdin as a turn of one conversation; with `-t`, the shim echoes typing and handles backspace, since there is no terminal inside |
| `docker run -d --name c IMAGE --keep "prompt"` | Answers, then keeps the session open |
| `docker exec c shem:generate "follow-up"` | Another turn of that session, which remembers the conversation |
| `docker commit c IMAGE2` | Docker pauses the container first, and the shim saves the session (the conversation and every block's state) into `/.shem`, so `IMAGE2` resumes the conversation |
| `docker logs`, `stop`, `rm`, `ps` | As for any container |

The server must be able to read containers' root filesystems, which containerd mounts as root, so for Docker run it as root:

```sh
sudo shem-neural-server --socket /tmp/shem-neural.sock
```

Its socket is open only to its user (and root). A CI job runs a prompt, piped turns, a live session with `docker exec`, and a `docker commit` whose image remembers the conversation.

## Compiled layers and tools

A model image can carry compiled layers: blocks whose weights are compiled from a specification rather than trained. The RV32IM CPU of the [neural vCPU](neural.md) is one, and a Shemfile adds it like any other layer ([`examples/tools`](../examples/tools/Shemfile)):

```dockerfile
FROM --platform=$BUILDPLATFORM alpine:3.22 AS tools
RUN apk add --no-cache zig
COPY mul.c .
RUN zig cc -target riscv32-linux-musl -mcpu=generic_rv32+m+a -static -O2 -o mul mul.c

FROM hf.co/Qwen/Qwen3-0.6B
COPY --from=shem:rv32im / /model/compiled/rv32im/
COPY --from=tools /mul /usr/local/bin/mul
COPY mul.json /usr/local/bin/mul.json
ENTRYPOINT ["shem:generate", "--tools", "/usr/local/bin", "--system", "Use the tools for arithmetic."]
CMD ["What is 123456789 times 987654321?"]
```

```
$ shem run tools
shem: ran mul 123456789 987654321 on the compiled RV32IM layer, 2,881 instructions
121932631112635269.
```

- `COPY --from=shem:rv32im` copies a layer shem compiles from the instruction set at build time, with `shem build` and `docker build` alike: the weights, a `config.json` that describes the machine, and a model card. Compiling is deterministic, so every build carries the same weights, and the release publishes them ([jgalego/shem-rv32im](https://huggingface.co/jgalego/shem-rv32im)).
- `--tools DIR` offers the image's programs in `DIR` to the model, through the chat template's own tool format. A program's interface is `NAME.json` beside it (a function's `description` and `parameters`, which become its arguments in order), or by default a list of `args` and `stdin`.
- Tool calls run the program with `shem run --vcpu neural`, using the image's compiled weights and root filesystem. The host returns the program's output as the tool reply. The CPU runs separately from the language model; it does not read or write the model's residual stream.
- The model must be one that calls tools: Qwen3-0.6B does, where Qwen3.5-0.8B answers the multiplication itself, and gets it wrong. Tool calls are read in either format templates ask for, JSON or `<function=...>` tags.

A test builds the example and checks the exact products of 9- and 11-digit numbers, through `shem run` and `docker run`. A 23-digit product takes the neural vCPU 9,794 instructions, about 10 seconds; the reply is the 45-digit answer, exact.

## Running

Run the model server on a GPU when there is one: `--device auto` (the default) picks one, and `--device cpu` or `--device cuda` choose. [`jobs/models_gpu.py`](../neural/jobs/models_gpu.py) checks both models on a GPU against the CPU (logits, greedy replies, and a session saved on the GPU that continues identically) and times them, on Hugging Face Jobs. On an L4, Qwen3-0.6B decodes at 27.8 tokens a second (7.1 on the CPU) and Qwen3.5-0.8B at 31.2 (5.6), with logits within 7e-5 of the CPU's and the same replies.

`shem run` unpacks a model image once, into `$SHEM_HOME/rootfs/<digest>`, and the driver loads the weights from there. The runtime computes in float32, so sessions are repeatable on the same device.

## Next

- **More compiled layers.** Other specifications compiled into blocks, and compiled layers that read and write the residual stream directly instead of through a tool call.
- **Speed.** Batching sessions of the same model in one forward pass, as the neural vCPU batches guests.
