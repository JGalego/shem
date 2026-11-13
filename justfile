set positional-arguments

toolchain_image := "shem-toolchain"

# List recipes
default:
    @just --list

# Build the shem binary
build:
    cargo build --release

# Run all tests
test:
    cargo test --workspace

# Check formatting and lints
lint:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings

# Format the code
fmt:
    cargo fmt --all

# Run a guest executable, e.g. `just run hello arg1`
[no-cd]
run *args: build
    "{{ justfile_directory() }}/target/release/shem" run "$@"

# Everything the hooks check
check: lint test

# Install the git hooks
hooks:
    pre-commit install --hook-type pre-commit --hook-type pre-push

# Build the guest toolchain image
toolchain:
    docker build -q --target toolchain -t {{ toolchain_image }} toolchain

# Compile C into a static shem guest, e.g. `just cc -O2 -o hello hello.c`
cc *args: toolchain
    mkdir -p "{{ justfile_directory() }}/target/zig-cache"
    docker run --rm -u "$(id -u):$(id -g)" \
        -v "{{ invocation_directory() }}:/src" \
        -v "{{ justfile_directory() }}/target/zig-cache:/zig-cache" \
        -e ZIG_GLOBAL_CACHE_DIR=/zig-cache \
        {{ toolchain_image }} rv32-cc "$@"

# Build the guest programs (riscv-tests, guests/, BusyBox, examples/) and run them; needs Docker
guest-tests: toolchain busybox-image
    docker build -q --target riscv-tests --output type=local,dest=target/riscv-tests toolchain
    docker build -q --target busybox --output type=local,dest=target/busybox toolchain
    mkdir -p target/guests
    just cc -O2 -o target/guests/hello guests/hello.c
    just cc -O2 -o target/guests/threads guests/threads.c
    cargo test --release --test guests -- --ignored

# Test the neural vCPU: its own unit tests, then against the reference vCPU; needs Docker and uv
neural-tests: toolchain linux-image
    mkdir -p target/guests
    just cc -O2 -o target/guests/threads guests/threads.c
    docker build -q --target riscv-tests --output type=local,dest=target/riscv-tests toolchain
    uv run --project neural pytest -q neural/tests
    cargo test --release --test neural -- --ignored

# Serve neural vCPUs on a Unix socket, for `shem run --vcpu neural`
neural-server socket="/tmp/shem-neural.sock":
    uv run --project neural python -m shem_neural.server --socket {{ socket }}

# Test and benchmark the neural vCPU on a GPU with Hugging Face Jobs, at the pushed HEAD; needs `hf auth login`
gpu-job flavor="l4x1":
    uvx --from huggingface_hub hf jobs uv run --flavor {{ flavor }} --timeout 1h neural/jobs/gpu.py --rev $(git rev-parse HEAD)

# Compare imported models with transformers on Hugging Face Jobs, at the pushed HEAD: gated checkpoints with your token, and those that need 256 GB; needs `hf auth login`
models-job flavor="cpu-performance" tests="tests/test_model.py":
    uvx --from huggingface_hub hf jobs run --flavor {{ flavor }} --timeout 6h --secrets HF_TOKEN \
        -e SHEM_MODEL_TESTS=1 -e SHEM_LARGE_MODEL_TESTS=1 -e SHEM_HUGE_MODEL_TESTS=1 -e HF_XET_HIGH_PERFORMANCE=1 \
        ghcr.io/astral-sh/uv:python3.12-bookworm bash -c \
        "curl -fsSL https://github.com/JGalego/shem/archive/$(git rev-parse HEAD).tar.gz | tar xz && cd shem-*/neural && uv sync -q --frozen && uv run pytest -v -rs --durations=0 {{ tests }}"

# Build the BusyBox rootfs as a Docker image, for Shemfiles to COPY --from
busybox-image:
    docker build -q --target busybox -t shem-busybox toolchain

# Download a prebuilt no-MMU Linux for VM mode (from cnlohr/mini-rv32ima-images, GPL)
linux-image:
    #!/bin/sh
    set -e
    [ -f target/linux/Image ] && exit 0
    mkdir -p target/linux
    commit=7da8c6e486b7d510426b5c2283cd338abcf04eb7
    curl -fsSL "https://raw.githubusercontent.com/cnlohr/mini-rv32ima-images/$commit/images/linux-6.8-rc1-rv32nommu-cnl-1.zip" -o target/linux/linux.zip
    echo "f6fa72e17ee810dfa42daec6d3836e427cadac3693069daa9a52d00055e4c056  target/linux/linux.zip" | sha256sum -c -
    python3 -m zipfile -e target/linux/linux.zip target/linux

# Boot Linux in VM mode, run commands in its shell, and power it off
vm-tests: linux-image
    cargo test --release --test vm -- --ignored

# Run an RV32 container through containerd with the shem shim; needs Docker, containerd and sudo
containerd-test: toolchain
    #!/bin/sh
    set -e
    cargo build --release -p containerd-shim-shem
    sudo install -m 0755 target/release/containerd-shim-shem-v1 /usr/local/bin/
    mkdir -p target/oci-hello
    just cc -O2 -o target/oci-hello/hello guests/hello.c
    printf 'FROM scratch\nCOPY hello /hello\nENTRYPOINT ["/hello"]\n' | docker build -q -t shem-hello:latest -f - target/oci-hello
    docker save shem-hello:latest | sudo ctr images import -
    # The shim traces every request here, and what it wrote to each stream
    sudo mkdir -p /run/shem
    status=0
    timeout 120 sudo ctr run --rm --runtime io.containerd.shem.v1 docker.io/library/shem-hello:latest shem-test /hello one two > target/oci-hello/out.txt || status=$?
    cat target/oci-hello/out.txt
    echo "ctr run exited with status $status (the program exits with its argument count, 3)"
    if ! grep -q "sum of squares below 100000" target/oci-hello/out.txt; then
        echo "--- the shim's trace"; sudo cat /run/shem/shim.log || true
        echo "--- containerd"; sudo journalctl -u containerd --no-pager | grep -iE 'shem|error|warn' | tail -30 || true
        exit 1
    fi

# Build an RV32 image with docker build and run it with docker run on the shem runtime; needs Docker and sudo
docker-test: toolchain
    #!/bin/sh
    set -e
    cargo build --release -p containerd-shim-shem -p shem
    sudo install -m 0755 target/release/containerd-shim-shem-v1 /usr/local/bin/
    mkdir -p target/oci-hello
    just cc -O2 -o target/oci-hello/hello guests/hello.c
    printf 'FROM scratch\nCOPY hello /hello\nENTRYPOINT ["/hello"]\n' | docker build -q --platform linux/riscv32 -t shem-hello:riscv32 -f - target/oci-hello
    docker info --format 'Docker {{{{.ServerVersion}}, storage {{{{.Driver}} {{{{json .DriverStatus}}'
    # The shim traces every request here, so a hang shows where it is
    sudo mkdir -p /run/shem
    status=0
    timeout -k 5 30 docker run --name shem-docker-test --runtime io.containerd.shem.v1 --platform linux/riscv32 shem-hello:riscv32 one two > target/oci-hello/docker.txt 2>&1 || status=$?
    cat target/oci-hello/docker.txt
    echo "docker run exited with status $status (the program exits with its argument count, 3)"
    if ! grep -q "sum of squares below 100000" target/oci-hello/docker.txt || [ "$status" != 3 ]; then
        # Each with a timeout: a stuck container also blocks Docker's commands about it
        echo "--- the shim's trace"; sudo cat /run/shem/shim.log || true
        echo "--- containers"; timeout 10 docker ps -a --no-trunc || true
        echo "--- containerd and Docker"; sudo timeout 20 journalctl -u containerd -u docker --since "-2min" --no-pager | tail -80 || true
        exit 1
    fi
    timeout 10 docker rm shem-docker-test >/dev/null

    echo "--- docker commit freezes a running program, and the image goes on from there"
    dir=target/oci-hello
    just cc -O2 -o $dir/tally guests/tally.c
    printf 'FROM scratch\nCOPY tally /tally\nENTRYPOINT ["/tally"]\n' | docker build -q --platform linux/riscv32 -t shem-tally:riscv32 -f - $dir
    run="docker run -i --runtime io.containerd.shem.v1 --platform linux/riscv32"
    rm -f $dir/tally.in && mkfifo $dir/tally.in
    timeout 120 $run --name shem-tally shem-tally:riscv32 < $dir/tally.in > $dir/tally.txt 2>&1 &
    exec 3> $dir/tally.in
    printf '1\n2\n' >&3
    seen() { for _ in $(seq 1 30); do grep -q "$1" $dir/tally.txt && return; sleep 1; done; return 1; }
    failed() {
        echo "$1"; cat $dir/tally.txt
        echo "--- the shim's trace"; sudo cat /run/shem/shim.log || true
        exit 1
    }
    seen "total 3" || failed "the program did not read its input"
    # The commit pauses the program as it waits for more, then it goes on
    timeout 60 docker commit shem-tally shem-tally:committed >/dev/null || failed "docker commit failed"
    printf '10\n' >&3
    seen "total 13" || failed "the program did not go on after the commit"
    exec 3>&-
    wait || failed "the program did not end with its input"
    timeout 10 docker rm shem-tally >/dev/null
    resumed=$(echo 4 | timeout 30 $run --rm shem-tally:committed)
    echo "the committed image read 4: $resumed"
    [ "$resumed" = "total 7" ] || failed "the committed image did not go on from total 3"
    # And with shem, as on Docker Desktop
    export SHEM_HOME=$PWD/$dir/home
    docker save shem-tally:committed | target/release/shem load
    resumed=$(echo 5 | target/release/shem run shem-tally:committed)
    echo "shem run read 5: $resumed"
    [ "$resumed" = "total 8" ] || failed "shem run did not go on from total 3"

# Model containers under Docker: run, run -i, exec into a live session, and commit it; needs Docker, uv and sudo
model-containers-test:
    #!/bin/sh
    set -e
    cargo build --release -p shem -p containerd-shim-shem
    sudo install -m 0755 target/release/containerd-shim-shem-v1 /usr/local/bin/
    uv sync -q --project neural
    # As root, to read the containers' root filesystems, which containerd mounts
    # SHEM_BIN runs tools' programs on the model's compiled layer
    sudo env SHEM_BIN="$PWD/target/release/shem" neural/.venv/bin/python -m shem_neural.server --socket /tmp/shem-neural.sock > target/model-server.log 2>&1 &
    trap 'sudo pkill -f "shem_neural[.]server" || true' EXIT
    until [ -S /tmp/shem-neural.sock ]; do sleep 1; done
    target/release/shem build -t qwen3 examples/qwen3
    target/release/shem save qwen3 | docker load
    run="docker run --runtime io.containerd.shem.v1 --platform linux/riscv32"
    check() { echo "$2" | tee -a target/model-containers.out; echo "$2" | grep -qi "$1" || { echo "expected $1"; tail -20 target/model-server.log; exit 1; }; }

    echo "--- a prompt"
    check "." "$(timeout 300 $run --rm qwen3 --max-tokens 16 'Say hello in Portuguese.')"
    # A system prompt keeps a 0.6B model on task; each check reads one reply
    sys="Answer in one short sentence."
    echo "--- turns on stdin"
    check "joana" "$(printf 'My name is Joana.\nWhat did I just tell you my name is?\n' | timeout 300 $run -i --rm qwen3 --system "$sys" --max-tokens 24 | tail -1)"
    echo "--- a live session, exec and commit"
    $run -d --name live qwen3 --keep --system "$sys" --max-tokens 24 'My name is Joana.'
    until [ -n "$(docker logs live 2>&1)" ]; do sleep 2; done
    sleep 8
    check "joana" "$(timeout 300 docker exec live shem:generate 'What did I just tell you my name is?')"
    docker commit live qwen3-remembers >/dev/null
    timeout 30 docker rm -f live >/dev/null
    check "joana" "$(timeout 300 $run --rm qwen3-remembers --max-tokens 24 'What did I tell you my name is?')"
    echo "--- programs on a compiled layer, with shem run and docker run"
    target/release/shem build -t tools examples/tools
    check "121932631112635269" "$(timeout 600 target/release/shem run tools 'What is 123456789 times 987654321?')"
    target/release/shem save tools | docker load
    check "8888888888711111111112" "$(timeout 600 $run --rm tools 'What is 99999999999 times 88888888888?')"

# Run an RV32 program in a Pod on shem, in a kind cluster; needs Docker, kind and kubectl
kubernetes-test:
    #!/bin/sh
    set -e
    # A static shim, since the kind node's C library may be older than this one
    cargo build --release --target x86_64-unknown-linux-musl -p containerd-shim-shem
    # Listed for the node's platform too: released containerd picks a pod's images by it
    docker build -q --platform linux/riscv32,linux/amd64 -t shem-k8s-hello:latest examples/hello
    kind delete cluster --name shem >/dev/null 2>&1 || true
    # The kubelet asks for images by runtime class, so it finds linux/riscv32 ones
    printf 'kind: Cluster\napiVersion: kind.x-k8s.io/v1alpha4\nfeatureGates:\n  RuntimeClassInImageCriApi: true\n' > target/kind.yaml
    kind create cluster --name shem --config target/kind.yaml --wait 120s
    node=shem-control-plane
    docker cp target/x86_64-unknown-linux-musl/release/containerd-shim-shem-v1 "$node:/usr/local/bin/"
    printf '%s\n' "" \
        "[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.shem]" \
        "  runtime_type = 'io.containerd.shem.v1'" \
        "[plugins.'io.containerd.cri.v1.images'.runtime_platforms.shem]" \
        "  platform = 'linux/riscv32'" | docker exec -i "$node" sh -c 'cat >> /etc/containerd/config.toml'
    docker exec "$node" systemctl restart containerd
    docker save shem-k8s-hello:latest | docker exec -i "$node" ctr -n k8s.io images import --all-platforms -
    kubectl apply -f examples/kubernetes/runtimeclass.yaml
    sed 's#image: .*#image: docker.io/library/shem-k8s-hello:latest#; s#imagePullPolicy: .*#imagePullPolicy: Never#' examples/kubernetes/hello.yaml | kubectl apply -f -
    if ! kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/shem-hello --timeout=120s; then
        kubectl describe pod shem-hello | tail -25
        docker exec "$node" journalctl -u containerd --no-pager | tail -40
        exit 1
    fi
    kubectl logs shem-hello | tee target/k8s.out
    grep -q "hello from a shem image" target/k8s.out
    kind delete cluster --name shem

# Push and pull an image through a local registry, and run ModelPack
# artifacts as image twins; needs Docker, oras and uv
registry-test:
    #!/bin/sh
    set -e
    docker run -d --rm --name shem-test-registry -p 5000:5000 registry:2 >/dev/null
    trap 'docker stop shem-test-registry >/dev/null' EXIT
    sleep 1
    SHEM_TEST_REGISTRY=localhost:5000 cargo test --release --test image registry -- --ignored
    SHEM_TEST_REGISTRY=localhost:5000 cargo test --release --test modelpack -- --ignored

# docker build model images through the frontend and the shem-neural image
# built from this checkout: a Hugging Face model primed with WARM, a
# ModelPack artifact as a base, and a GGUF file; needs Docker, oras, uv and
# a registry
frontend-models-test registry="localhost:5000":
    #!/bin/sh
    set -e
    reg={{registry}}
    cargo build --release -p shem
    shem=target/release/shem
    docker build -q -t $reg/shem-neural:ci neural >/dev/null
    docker push -q $reg/shem-neural:ci >/dev/null
    docker build -q -t $reg/shem-dockerfile:ci frontend >/dev/null
    docker push -q $reg/shem-dockerfile:ci >/dev/null
    dir=target/frontend-models
    rm -rf $dir && mkdir -p $dir/warm $dir/model $dir/artifact $dir/gguf
    export SHEM_HOME=$PWD/$dir/home
    model=hf.co/trl-internal-testing/tiny-Qwen3ForCausalLM
    # The registry is on localhost, which a build's fetch reaches on the host's network
    build() { docker build -q --platform linux/riscv32 --allow network.host --build-arg SHEM_NEURAL_IMAGE=$reg/shem-neural:ci -t "$1" "$2" >/dev/null; }
    # Images Docker built run from shem's store, as on Docker Desktop
    load() { docker save "$1" | $shem load; }
    same() { [ "$1" = "$2" ] || { echo "expected the same reply:"; echo "$1"; echo "$2"; exit 1; }; }

    echo "--- WARM through docker build, as shem build primes it, from a file of the image too"
    echo "You are a golem." > $dir/warm/golem.txt
    shemfile="FROM $model\nCOPY golem.txt /etc/golem.txt\nENTRYPOINT [\"shem:generate\", \"--max-tokens\", \"8\", \"--system\", \"Stay in character.\", \"--system-file\", \"/etc/golem.txt\"]\nWARM --until=prompt\n"
    printf "# syntax=$reg/shem-dockerfile:ci\n$shemfile" > $dir/warm/Dockerfile
    printf "$shemfile" > $dir/warm/Shemfile
    build shem-warm $dir/warm
    load shem-warm
    $shem build -t warm $dir/warm
    same "$($shem run shem-warm 'Who are you?')" "$($shem run warm 'Who are you?')"

    echo "--- a ModelPack artifact as a base"
    printf "FROM $model\n" > $dir/model/Shemfile
    $shem build -t tiny $dir/model
    $shem push --plain-http --modelpack --uncompressed tiny $reg/shem-tests/tiny:modelpack
    printf "# syntax=$reg/shem-dockerfile:ci\nFROM $reg/shem-tests/tiny:modelpack\n" > $dir/artifact/Dockerfile
    build shem-artifact $dir/artifact
    load shem-artifact
    same "$($shem run shem-artifact 'Hello')" "$($shem run $reg/shem-tests/tiny:modelpack 'Hello')"

    echo "--- a GGUF file"
    printf "# syntax=$reg/shem-dockerfile:ci\nFROM hf.co/Qwen/Qwen3-0.6B-GGUF:Q8_0\nENTRYPOINT [\"shem:generate\", \"--max-tokens\", \"8\"]\n" > $dir/gguf/Dockerfile
    build shem-gguf $dir/gguf
    load shem-gguf
    $shem run shem-gguf 'Say hi.' | tee $dir/gguf.out
    test -s $dir/gguf.out

# Build and run the model examples, each in a store of its own, then the
# ModelPack round trip through a registry on localhost:5000; needs uv, modctl,
# oras and the hf CLI
examples-test:
    #!/bin/sh
    set -e
    cargo build --release -p shem
    export PATH="$PWD/target/release:$PATH"
    check() { echo "$2"; echo "$2" | grep -qi "$1" || { echo "--- expected $1"; exit 1; }; }
    run() {
        echo "--- $1"
        export SHEM_HOME=$PWD/target/examples/$1
        rm -rf $SHEM_HOME
        shem build -t $1 examples/$1
        check "$2" "$(timeout 900 shem run $1)"
        rm -rf $SHEM_HOME
    }
    run smollm2 golem
    run granite-moe expert
    run mamba lisbon
    run llama-gguf golem
    run handbook "shem push"
    echo "--- modelpack"
    export SHEM_HOME=$PWD/target/examples/modelpack
    rm -rf $SHEM_HOME
    examples/modelpack/run.sh
    rm -rf $SHEM_HOME

# Run examples-test on Hugging Face Jobs, at the pushed HEAD, with the gated Gemma example and the image volume Pod's command too; needs `hf auth login`
examples-job flavor="cpu-upgrade":
    uvx --from huggingface_hub hf jobs run --flavor {{ flavor }} --timeout 2h --secrets HF_TOKEN -e HF_XET_HIGH_PERFORMANCE=1 \
        ghcr.io/astral-sh/uv:python3.12-bookworm bash -c \
        "curl -fsSL https://github.com/JGalego/shem/archive/$(git rev-parse HEAD).tar.gz | tar xz && sh shem-*/neural/jobs/examples.sh"

# Mount a model image as a Kubernetes image volume, as a model server would
# read it: its /model is a Hugging Face checkpoint; needs Docker, kind and
# kubectl, and uv for the import
kubernetes-model-volume-test:
    #!/bin/sh
    set -e
    cargo build --release -p shem
    dir=target/model-volume && rm -rf $dir && mkdir -p $dir/model
    export SHEM_HOME=$PWD/$dir/home
    printf 'FROM hf.co/trl-internal-testing/tiny-Qwen3ForCausalLM\n' > $dir/model/Shemfile
    target/release/shem build -t shem-model-volume:latest $dir/model
    # Listed for the node's platform too, which image volumes are pulled for
    target/release/shem save --platform linux/amd64 shem-model-volume:latest -o $dir/image.tar
    kind delete cluster --name shem-volume >/dev/null 2>&1 || true
    printf 'kind: Cluster\napiVersion: kind.x-k8s.io/v1alpha4\nfeatureGates:\n  ImageVolume: true\n' > $dir/kind.yaml
    kind create cluster --name shem-volume --config $dir/kind.yaml --wait 120s
    trap 'kind delete cluster --name shem-volume >/dev/null 2>&1' EXIT
    node=shem-volume-control-plane
    docker exec -i "$node" ctr -n k8s.io images import --all-platforms - < $dir/image.tar
    kubectl apply -f - <<YAML
    apiVersion: v1
    kind: Pod
    metadata: {name: shem-model-volume}
    spec:
      restartPolicy: Never
      containers:
      - name: reader
        image: busybox:1.37
        command: ["sh", "-c", "ls /models/model && sha256sum /models/model/block-000-attention.safetensors && head -c 120 /models/model/shem.json"]
        volumeMounts: [{name: model, mountPath: /models}]
      volumes:
      - name: model
        image: {reference: docker.io/library/shem-model-volume:latest, pullPolicy: Never}
    YAML
    kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/shem-model-volume --timeout=120s || { kubectl describe pod shem-model-volume | tail -20; exit 1; }
    kubectl logs shem-model-volume | tee $dir/out
    grep -q "model.safetensors.index.json" $dir/out
    grep -q '"format": "shem.model/v2"' $dir/out

# Build the website (landing page, install script and docs) as the docs workflow does, and serve it at http://localhost:8000
site:
    rm -rf _site
    mdbook build -d _site/docs
    cp -r site/. install.sh _site/
    python3 -m http.server -d _site 8000
