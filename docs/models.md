# Model images

A model image is a language model packaged the way a container image packages a filesystem: the model is the base image, its blocks are layers, and `shem run` starts a session of it the way `docker run` starts a container. Images push to and pull from any OCI registry.

```sh
shem build -t qwen3 examples/qwen3          # Qwen3-0.6B from Hugging Face, one layer per block
shem run qwen3 "What is a golem?"           # a session: the prompt in, the reply out
shem push qwen3 ghcr.io/me/qwen3:0.6b       # to a registry
shem run ghcr.io/me/qwen3:0.6b "Olá"        # elsewhere: pulled on first use, then run
```

`docker build` builds model images too, through shem's BuildKit frontend: add `# syntax=ghcr.io/jgalego/shem/dockerfile:latest` as the first line ([shemfile.md](shemfile.md#with-docker-build)).

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

A `shem:` entrypoint names a driver, which the host runs, like a shell builtin; the prefix keeps it from ever being mistaken for a file in the image. `shem:generate` options: `--max-tokens N` (default 256) and `--system TEXT`. Decoding is greedy, so a prompt always gets the same reply.

## Sessions as layers

`WARM --until=prompt` primes a session at build time and freezes it into the image ([`examples/golem`](../examples/golem/Shemfile)):

```dockerfile
FROM hf.co/Qwen/Qwen3.5-0.8B
ENTRYPOINT ["shem:generate", "--system", "You are Shem, a golem made of clay and brought to life by a word."]
CMD ["Who are you?"]
WARM --until=prompt
```

The builder runs the model over the conversation up to where the user's prompt goes, here the system prompt, and saves every block's state in the image's state layer: the attention layers' key and value caches, and the Gated DeltaNet layers' convolution inputs and memory matrices. `shem run` continues that session, so the system prompt is never read again, and the reply is exactly what an unprimed session would give; a test checks this for both models.

What the state costs depends on the blocks. After the same 27-token prompt, Qwen3-0.6B's attention caches take 6 MB and grow with every token, while Qwen3.5-0.8B's state takes 20 MB, almost all of it Gated DeltaNet memories, and stays that size however long the session runs. A long document read at build time is where the fixed size pays off.

The session is saved as `session.safetensors`, every block's state by block number and name, and `session.json`, the tokens it has read. `shem run` unpacks them into `/.shem`, which is also where `shem save` puts them for `docker load`.

## The model as files

The importer (`shem_neural.model.importers`) writes the model into `/model`:

- `shem.json`, the manifest: the embedding, the blocks in order, and the head, each with its type, its weights file and its parameters, plus the tokenizer;
- `blocks/NNN-TYPE.safetensors`: one weights file per block, in the checkpoint's own type;
- `embed.safetensors`, `head.safetensors`, `tokenizer.json` and `chat_template.jinja`.

Each weights file is its own image layer, so images built on the same model share those blobs in stores and registries. A model with more than 95 weights files shares layers between neighbouring blocks, since overlay filesystems stack about 128 layers at most.

The format knows nothing about Qwen. A model is a residual stream: an embedding, then blocks that each read the stream and add to it, then a head. A Qwen layer is two blocks, a token mixer and an MLP.

| Block type | What it does | Its state in a session |
|---|---|---|
| `attention` | Grouped-query attention, with rotary positions on all or part of each head, optional RMSNorm on queries and keys, and an optional output gate | The key and value cache, which grows with the session |
| `gated_deltanet` | Gated DeltaNet: a causal depthwise convolution, then a matrix memory per head, written with the gated delta rule | The convolution's last inputs and the memory matrices, a fixed size however long the session |
| `mlp` | A SwiGLU feed-forward block | None |

Supporting a new model family means writing an importer that maps its tensors onto these types. A new kind of layer, such as Mamba or a mixture of experts, is a new block type in [`blocks.py`](../neural/src/shem_neural/model/blocks.py), registered by name. Neither changes the image format.

## Imported models

| Model | Blocks | Image | Verified |
|---|---|---|---|
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 28 attention, 28 MLP | 59 layers, 892 MB | Logits within 1e-3 of transformers, and the same 24 greedy tokens |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 18 Gated DeltaNet, 6 attention with output gates and partial rotary positions, 24 MLP | 51 layers | The same, against transformers' Qwen3.5 |

Qwen3.5's vision encoder and multi-token prediction layer are left out. [`tests/test_model.py`](../neural/tests/test_model.py) does the comparison; it downloads the checkpoints, so it runs with `SHEM_MODEL_TESTS=1`.

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

## Running

`shem run` unpacks a model image once, into `$SHEM_HOME/rootfs/<digest>`, and the driver loads the weights from there. The runtime computes in float32 on the CPU, so sessions are repeatable: Qwen3-0.6B replies at about five tokens a second.

## Next

- **Compiled layers.** Exact circuits, such as the RV32IM CPU of the [neural vCPU](neural.md), as block types spliced into a model, and tool calls that run programs on them.
- **Speed.** The model server holding loaded models and batching sessions, on GPUs.
