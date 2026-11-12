#!/bin/sh
# The examples on a Hugging Face Job, at the commit it downloads: shem and the
# examples' tools installed in the job's container, a registry on
# localhost:5000, then `just examples-test`, the gated Gemma example with the
# job's token, and the Kubernetes image volume Pod's command on a model
# image's files.
#
#   just examples-job
set -eu
cd "$(dirname "$0")/../.."
bin=/usr/local/bin

echo "--- tools"
curl -fsSL https://sh.rustup.rs | sh -s -- -y -q --profile minimal
. "$HOME/.cargo/env"
curl -fsSL https://github.com/modelpack/modctl/releases/download/v0.2.2/modctl-0.2.2-linux-amd64.tar.gz | tar xz -C $bin modctl
curl -fsSL https://github.com/oras-project/oras/releases/download/v1.3.6/oras_1.3.6_linux_amd64.tar.gz | tar xz -C $bin oras
curl -fsSL https://github.com/distribution/distribution/releases/download/v3.1.2/registry_3.1.2_linux_amd64.tar.gz | tar xz -C $bin registry
uv tool install -q rust-just
uv tool install -q huggingface_hub
export PATH="$HOME/.local/bin:$PATH"
printf 'version: 0.1\nstorage:\n  filesystem:\n    rootdirectory: /tmp/registry\nhttp:\n  addr: 127.0.0.1:5000\n' > /tmp/registry.yml
registry serve /tmp/registry.yml > /tmp/registry.log 2>&1 &
until curl -fs http://127.0.0.1:5000/v2/ > /dev/null; do sleep 1; done

just examples-test

echo "--- gemma3, a gated model, with the job's token"
export SHEM_HOME="$PWD/target/examples/gemma3"
target/release/shem build -t gemma3 examples/gemma3
target/release/shem run gemma3 | tee /tmp/gemma3.out
grep -qi golem /tmp/gemma3.out
rm -rf "$SHEM_HOME"

echo "--- the image volume Pod's command, on a model image's /model mounted where the Pod mounts it"
mkdir -p /models/smollm2
uv run -q --project neural python -m shem_neural.model.importers HuggingFaceTB/SmolLM2-360M-Instruct /models/smollm2/model
uv run -q --no-project --with pyyaml python - examples/kubernetes/model-volume.yaml <<'EOF' | tee /tmp/pod.out
import subprocess
import sys

import yaml

pod = yaml.safe_load(open(sys.argv[1]))
sys.exit(subprocess.run(pod["spec"]["containers"][0]["command"]).returncode)
EOF
grep -qi golem /tmp/pod.out
echo "--- every example ran"
