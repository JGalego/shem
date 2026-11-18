
## 0.4.0 (2026-11-18)

### Features

- Shem run -i has a conversation with a model, a turn to each line, with /clear and /exit
- --quantize int8 holds a model's projections as int8, so a model takes a quarter of float32's memory, its sessions and receipts recording it
- Seeded sampling with --temperature, --top-p, --top-k and --seed, in the driver, shem run and the API, with receipts that record the seed and replay by drawing again
- The API returns a receipt of a reply on request, shem replay checks it, and shem sign and verify hand receipts to cosign for Sigstore signatures
- The API takes tools and returns tool calls, gives log probabilities, and serves /v1/completions, refusing the response formats it cannot keep
- A LoRA adapter is a layer on a model's image, merged into the weights it names as the model loads, its name and digest recorded in sessions and receipts
- Processes in a container serve and call each other over Unix and loopback stream sockets, which never leave the container
- Docker exec starts another program in a running program container, and the shim reports the processes, CPU time and memory docker stats shows
- Static PIE executables load and relocate themselves, rv32-cc builds them with RV32_PIE, and Lua is an example interpreter built from source
- /v1/embeddings gives a unit vector of what the model made of a text, from its last token or averaged, as floats or base64
- Shem doctor says what is missing and what to do, and shem completions and shem man make shell completions and manual pages, which a release's archive carries
- /metrics on the API and the registry in Prometheus's format, and the registry keeps the digests of weights' layers in a file, so a restart does not read every large file through again
- --quantize int4 holds a model's projections in 4 bits, in groups of 64 with a scale each, an eighth of float32's memory
- Response_format json_object masks the model's choices so a reply is always a JSON object, checked against Python's parser
- Response_format json_schema masks the model's choices to an object that fits the schema, and logprobs come in a stream, a chunk at a time
- Docker exec -t gives the program a terminal of its own, and a container whose processes all wait for a client waits for docker exec instead of failing
- The reference vCPU runs compressed instructions for a program built with the C extension, checked against clang's encodings, the official rv32uc test and QEMU
- The build cache keeps what a program's or VM's WARM made, so a build that would run it again takes it from the cache
- The neural vCPU runs compressed instructions through a table of their expansions, checked against the reference and the rv32uc test, and holds register bits as bytes
- A Rust example, standard library and all, built for RV32 with a target of its own and run with the C extension, and sigaltstack is taken as Rust's runtime asks
- SmolVLM and Idefics3 read images: the encoder as part of the model, the pictures as data URLs in the API and --image, checked against transformers on a real checkpoint
- SmolVLM pictures are cut in tiles as the processor cuts them, with a detail of low to read one whole, checked against transformers on a real checkpoint, and --image-detail for shem run
- A program or VM run's files are an overlay of the image's unpacked copy, mounted without privileges, instead of every layer unpacked for the run
- A container's command line takes --receipt, written inside the container beneath its root, for a first reply or a later turn of a kept session, replayed from the tokens it read
- A build from an image already built has its base's files as an overlay of the unpacked copy, so the base is not unpacked for each build and the image made is the same
- LLaVA models read images: a CLIP or SigLIP encoder and a projector, the processor's pixels and tokens, checked against transformers on a real checkpoint
- Shem run --root-overlay runs an executable on an overlay of its root, and the model server's tools use it, so a program's writes never reach the image's kept copy
- Images with a layer in common keep its blob once, and the copies unpacked to run them keep a large file once, as hard links to a pool that rmi and prune sweep
- Shem builds for Windows, its host-file calls behind a sys layer, with sockets, overlays, pooled layers and raw terminals left out there, and a Windows leg in CI

### Fixes

- The install script installs ORAS when it is missing, so the quickstart's first pull works on a fresh machine
- Sessions are held to the model's context window: the importer records it, a conversation past it is refused, and a reply stops where it ends
- The context window's test stand-in has a room, and the release-path CI job starts from a clean directory, as a cached target/ may hold an earlier run's
- The model server's socket is per user, not a fixed path in /tmp, and shem and the shim refuse a socket another user listens at
- Shem serve takes an API key from SHEM_API_KEY, CORS only from origins it is given, bodies of up to 8 MiB, a limited queue and a client timeout, and a stream's failure is an event
- Shem takes a lock on the store while it changes it, and installs a built or pulled image whole, so concurrent builds, pulls and removals leave no half an image
- Opening a file in a container's root uses openat2 beneath it, so a link swapped in on the host after the path was resolved leads nowhere
- An image whose weights are not the ones its fingerprint names runs them in a model of its own, so images made by another version still run in the model server
- The Lua example passes hadolint, the mypy ratchet sorts as C and pins mypy, and a coverage file is not tracked
- Mkdir, unlink, symlink and rename act on the directory that holds their path, opened beneath the root, so a link swapped in on the host leads nowhere
- Making the root answers that it exists, which mkdir -p relies on, and mkdirat's mode is the platform's type
- Stat, readlink, utimensat and directory listings go through descriptors opened beneath the root, which closes the path race for every call that acts on a path
- The docker exec -t test runs on Linux, where the shim's terminals are
- The GPU benchmark resets the compiler's recompile count for each batch it compiles, as one sweep of sizes compiles more than its limit
- The Spike install takes its softfloat submodule
- Image names are listed with slashes and kept out of Windows-hostile directory names, and a tool that did not run says why
- An overlay's namespaces are tried in a child first, so a kernel that refuses the id maps leaves the run as it was, and Windows skips the tests of what it has not got
- A file's inode on Windows is the one its volume gives it, not its creation time, which two files made together share

### Performance

- Shem run asks the user's model server to reply, starting one that keeps models loaded for ten minutes if there is none, so the second run of a model skips reading its weights
- Attention appends to a cache that has room, uses PyTorch's fused attention and keeps its rotary frequencies, so a token's cost stops growing with the length of the conversation
- Containers of images with the same weights share one loaded model, verified against its fingerprint, and a model loading no longer holds up the others
- The API server answers requests together, up to sixteen replies reading the model's weights in one pass, with each reply what it would be alone
- The fused executor skips the shifter and memory when no guest in the batch needs them, and steps without gradient tracking
- A build cache keeps models imported at a pinned revision and the sessions primed on them, so rebuilding takes a fraction of a second; shem build --no-cache and shem prune --all

### Documentation

- Say what is in a model's weights and what is not, that the CPU is a separate network, and what joining them would take
- When to use shem and when not to, against Ollama, Docker Model Runner, RamaLama and KitOps, and what it does not do
- Issue and pull request templates, a list of places to start for new contributors, and what the adopters list is for
- The README says what works now, with sampling, adapters, int8, the fuller API, sockets, exec and the build cache
- The L4's throughput with byte registers, measured on a Hugging Face Job: 3.9 times the earlier best at 16,384 guests, every mode identical to the CPU
- How far apart devices' logits are, measured on an L4, an A100 and a CPU in float32 and bfloat16, for the tolerances receipts use
- Shem runs on Windows without WSL, with a page on what it has not got yet, and a failing Windows test now fails CI

### Tests

- The reference vCPU runs random RV32IM programs against QEMU's user-mode emulator in CI, and their registers must be the same
- Hypothesis property tests of the Python parsers: receipts, manifests, tool calls, messages, addresses, credentials, sizes and the digest file
- Hypothesis property tests of the vCPU server's frames and of image references
- A Hugging Face Job that measures how far apart devices' logits are along a reply's path, in float32 and bfloat16
- The reference vCPU against Spike on random programs, plain and compressed, as bare-metal programs that say what they computed in a signature

### CI

- Types without new mypy errors, Rust and Python coverage in the job summary, a benchmark that fails if batching stops paying, and tests of the Spaces' logic
- Releases carry an Intel Mac binary, built on Apple Silicon, for the installer and the Homebrew formula, and the README says Windows runs it in WSL 2
- A job that builds Spike at a pinned commit and compares the reference vCPU with it on random programs, plain and compressed

### Other

- Release 0.4.0
## 0.3.0 (2026-11-14)

### Features

- A landing page and an install script on the website, with the docs under /docs/
- Shem load stores the images of a docker save archive, to run what Docker built where its containerd cannot see the shim, as on Docker Desktop
- --dtype bfloat16 runs a model session in half the memory, and the model server keeps a model loaded in each type its images ask for
- Docker commit freezes a running program, and its image goes on where the program paused; a kill ends a program that waits for input
- Docker run -t gives a program a terminal of its own, with echo, line editing, Ctrl-C and the window's size
- Shem serve and docker run --serve serve a model image as OpenAI's chat completions API, in the container's own network under Docker
- Shem push, pull and run take hf.co/ORG/NAME, a Hugging Face repo that holds an image's OCI layout or a checkpoint to import, shem registry serves those repos to other OCI clients with pages for people, and shem-neural-verify compares an import with transformers
- The shem my repo Space turns a Hugging Face checkpoint into a shem image in the user's own repo, compared with transformers, with a ModelPack artifact on its modelpack branch
- Shem registry's pages look like model pages, with a search box, repo cards, badges, and the commands to use an image beside its details
- Shem registry serves a repo that holds a checkpoint as a ModelPack artifact of its files, its open pages update themselves, and its home page leads with search, counts and the repos pulled
- The Space's log shows each step with its state, time and outcome, its output folded away once done, progress as bars, and links to follow
- Shem images lists the store as a table, with a column each for the image, its short ID, whether it runs a program, VM or model, and its size, the header bold and tags dim on a terminal
- Shem images takes --format, as json or a Go template like docker's, after table for aligned columns, and -q for the IDs alone and --no-trunc for whole ones
- Shem rmi removes images by reference or ID, deleting the unpacked copy of each one no tag holds any more, and shem prune deletes the copies of images the store no longer holds
- Shem tag stores an image under another reference, linking its blobs, shem inspect shows an image's ID, settings and layers as JSON, and shem history the Shemfile line behind each layer, which builds now record
- Shem registry --public reads repos with each client's Hugging Face token, from docker login, or none, never the host's, so a registry anyone can reach serves each client only what the Hub would
- Shem run --receipt writes down a model run, its image's digest, prompt, reply and each token chosen with its margin, and shem replay checks one by running it again: verified, a mismatch, or undecided at a close call
- Shem registry serves a checkpoint's files as an image too, at files/ORG/NAME, or safetensors/ORG/NAME for one format of weights, for Docker to pull or copy from and Kubernetes to mount: each file of weights a tar layer sent as it is read from the Hub, and stored nowhere
- A Space that checks a receipt in a browser, replaying its prompt on its image, with each token marked by its margin, and shows what a changed prompt changes in the reply, a word at a time; its steps, space/replay.py, run from a terminal too

### Fixes

- Leave out the default ENTRYPOINT when a model stage sets its own, which made docker build warn about a line of the generated Dockerfile
- End LICENSE with a single newline, as the end-of-file fixer in CI wants
- A signal typed at the terminal goes to its foreground process group, init's included, and no longer returns from a system call init never made
- An image's layers and session files can no longer delete or write files outside the root they unpack into, by paths from /, paths that climb out, or links that lead out
- Counts a frozen container brings, of instructions and time, wrap or stop at their limits instead of overflowing
- Shem-neural lists a Hugging Face repo without requests, which it does not depend on, so a release's shem builds model images again, and CI builds one as a release does
- A release's shem imports GGUF files, fetching shem-neural with its gguf extra
- A guest's relative paths walk its working directory from the root, so a directory swapped for a symlink after the guest got there no longer leads out of it
- Device nodes and FIFOs in an image are not opened on the host, utimensat never follows a final symlink on the host, directories open in the guest are found through the root again, and a frozen working directory cannot climb out of it
- Sendfile into a pipe moves its input on only by what the pipe took, so BusyBox pipelines no longer lose data
- Utimensat refuses times past what 64 bits of nanoseconds hold instead of overflowing, as fuzzing frozen machines found
- Programs with program headers Linux refuses, more than a page of them or of another size, are refused instead of taking ages to load, as fuzzing found
- The shim reads and writes a container's /.shem state without following links the image or the guest left there, and reads only regular files of a bounded size
- A batch's CUDA graph capture no longer fails when the garbage collector frees another batch's graph during it, as it did on T4, A100, L40S and RTX PRO 6000
- Build the BuildKit frontend with golang.org/x/net 0.60.0 and Go 1.27.4, which fix vulnerabilities govulncheck finds on its path
- A RUN the model server cannot step fails alone, instead of stopping its executor and leaving every later RUN waiting
- The model server, which may be root, refuses --warm and --session on a container's command line, and the driver takes no abbreviated options
- The model server keeps each container's model apart, since two images may hold the same manifest and different weights, and unloads it when its last container ends
- The Space logs users in with Hugging Face, which Gradio does only when SYSTEM is spaces, as Docker Spaces don't set it
- The Space's user can write its own home, which uv's cache left to root, and the Space has the spaces package Gradio's reloader needs there
- The Space's log shows each step's output as it comes, with progress bars updated in place, instead of under a spinner until the end
- Shem registry says a repo that holds no image may hold a checkpoint for shem run, and its home page that names are lowercase
- Shem pull and run read a registry reference without a tag or digest as latest, as Docker does, instead of handing oras one it refuses
- The Space's image has every module of the app, and its raw log shows in monospace, each line after the time since the start
- A session published beside its model leaves the model's history without the step that made the session, so the model is the image before WARM byte for byte
- Shem registry answers what is no HTTP request, such as the TLS hello of a client that expects HTTPS, with an HTTP error at once, so Docker falls back to HTTP for a registry on localhost
- The blog's contents links use the anchors Hugging Face gives emoji headings, with explicit anchors so they resolve on GitHub too

### Documentation

- Update the blog post for 0.2.0
- Add CNCF governance files and a DCO check
- Start the model server from a release, as a systemd service or with sudo and full paths, as well as from a checkout
- Take terminals for program containers off the roadmap, and put docker exec into them on it
- The Space's upload command leaves out local caches, which git ignores and hf upload does not
- The Space's upload command gives each excluded pattern its own --exclude, as hf upload takes them
- GIFs of the shem my repo and shem registry Spaces in the models docs, and a blog update on both
- Tighten the blog's section on the shem my repo and shem registry Spaces, and fix its link from the updates list
- A table of contents in the blog
- An emoji on each section heading of the blog
- The blog links two shem images on the Hub as ready-made examples
- GIFs of a checkpoint pulled and mounted with Docker from the registry Space and of the shem replay Space, in the models docs, and blog sections on images of checkpoints' files and on receipts
- Shorten the blog's sections on images of checkpoints' files and on receipts

### Tests

- Fuzz what shem reads from images, archives for shem load, programs, frozen containers, layers and Shemfiles, every day, and try their seeds in CI

### CI

- Print the shim's trace, with the bytes it wrote to each stream, when the ctr run test loses the program's output
- Lint Dockerfiles and Shemfiles with hadolint, through a script that turns shem's own instructions into comments, and run the toolchain's pipes with pipefail
- The containerd test tries ctr run up to three times, as ctr run --rm can drop the output of a container that lives milliseconds before reading it, which a real fault would not

### Other

- Relicense from MIT to Apache 2.0
- Release 0.3.0
- Release 0.3.0
## 0.2.0 (2026-11-12)

### Features

- Llama, Mistral and Qwen2 imports, and model directories that load as Hugging Face checkpoints
- Gemma, Phi-3, OLMo and SmolLM3 imports, and chat templates that see the tokenizer's special tokens
- Run ModelPack artifacts as image twins that share their blobs
- Publish model images as ModelPack artifacts with shem push --modelpack
- Publish sessions as artifacts linked to their models, and resume them with shem run --state
- Keep block files at the top of the model directory, where vLLM looks for weights
- Run ModelPack layout directories as twins, and export uncompressed artifacts with their files at the root
- Mixtures of experts, Granite, Mamba and GGUF imports, docker build on ModelPack artifacts, and model images as Kubernetes image volumes
- Shemfile syntax highlighting and file icon for VS Code
- --system-file adds a file of the image to the system prompt, and --stop ends a reply at a text, as base models need
- Examples for SmolLM2, Granite MoE, Mamba, a Llama GGUF, Gemma, a handbook primed at build time, a ModelPack round trip and an image volume, tested in CI and on Hugging Face Jobs
- Version session state by block type and precision

### Fixes

- Leave out the second copy of the weights that Mistral's repos carry in their own format
- Keep a layout's blob that is already there, as a read-only link oras left, when an image built on a ModelPack twin exports uncompressed
- Leave out of main the session versions another change in progress records, which the driver doesn't write yet
- Commit new formulas to the Homebrew tap and skip unchanged ones

### Documentation

- A blog post on LLMs as containers
- Trim filler from the blog post
- Add missing newline to blog post
- Name ModelPack in the blog post's prior art
- Compare ModelPack and shem in the README
- The large checkpoints compared in bfloat16
- The checkpoints compared on a Hugging Face Job, up to Qwen3-30B-A3B and Mixtral-8x7B, the driver's new options and the examples

### Tests

- Compare same-weight logits within float rounding, as math libraries order a mixture of experts' sums by memory layout
- Compare checkpoints of up to 47 billion parameters with transformers on a Hugging Face Job with 256 GB, with just models-job

### CI

- Leave the saved test session out of the end-of-file fixer

### Other

- Release 0.2.0
- Drop the redundant homepage and inherit the readme from the workspace
## 0.1.0 (2026-11-11)

### Features

- Reference RV32IM vCPU and container runtime
- Run musl C programs and pass the official riscv-tests
- Run BusyBox with a filesystem, a virtual clock and more system calls
- Processes with fork, exec, wait, pipes and signals
- Neural vCPU compiled into network weights
- Shemfile builds, OCI layouts and WARM
- Paged memory with copy-on-write and continuous batching
- Boot a no-MMU Linux kernel in VM mode
- Containerd shim to run RV32 containers through ctr
- Shem push and shem pull
- Neural CPU on GPUs with flat memory and compiled steps
- Booted VM snapshots and VM images
- VM mode on the neural vCPU
- Compress image layers with zstd
- File mappings and file metadata on the virtual clock
- /proc, sysinfo and statfs
- Docker run on shem
- Run containers inside the model with docker run, and shem save for docker load
- Model images that build, run, push and pull language models
- Model images with primed sessions frozen into state layers
- Shem images are standard OCI images that docker pulls and loads
- Run model images in containers on the model server
- A BuildKit frontend that lets docker build Shemfiles
- Publish the compiled CPU's weights with a model card on Hugging Face
- A Kubernetes RuntimeClass for shem, tested on kind
- Live model sessions under docker run -i, docker exec and docker commit, and Kubernetes sandboxes
- Programs as compiled layers, run exactly when a model calls them as tools
- Shared writable file mappings, host terminals, and ppoll
- List images under node platforms for Kubernetes on released containerd
- Threads on any vCPU, futex, and a socket that appears only when it listens

### Fixes

- Open the shim's stdin FIFO without waiting for a writer
- Report a failed start on the container's stderr
- Give docker a network namespace to configure, and end tasks killed before they start
- Multi-turn sessions on every chat template, model sessions on GPUs, and the kubelet feature gate for riscv32 images

### Performance

- Integer shifter and register rows by gather and scatter

### Documentation

- What works by mode and what is next
- File mappings and deterministic file metadata
- Getting started without just and with cleanup
- Link the software shem is built with
- Move the built-with links to their own section
- Set the name's origin apart from the intro
- Development guide in the README
- Getting started runs a container inside the model with docker
- One Contribute section, with the install note under Release
- Changelog up to the release
- Add README visuals and Docker model examples
- Undo the Docker path in cleanup, one registry placeholder, and the platforms images are built for
- Compare Docker and LLM image bases

### Tests

- Report ctr's status and containerd's log on failure
- Give ctr run the program as well as its arguments
- Report Docker's state and logs when docker run fails
- Trace the shim's requests and time out the docker test's diagnostics

### CI

- Add CI workflow, README badges and tagline
- Pin setup-uv to v10.3.3
- Use actions/cache v6
- Release pipeline for crates.io, PyPI, binaries and the toolchain image
- Publish the runtime image and the example images on each release
- Cargo deny for licences and advisories, and Dependabot updates
- Cargo binstall metadata, a Homebrew formula and a docs site

### Other

- Add MIT license
- Format the shim's model test and the neural project lookup
- Let the model card keep one line per paragraph
- Licence notices for model images, SECURITY.md, a changelog, and a private model server socket
- Trim code and documentation clutter
<!-- generated by git-cliff -->
