#!/bin/sh
# A model's round trip between ModelPack and shem, through a registry on
# localhost:5000. It needs shem, modctl, oras and the Hugging Face CLI:
#
#   docker run -d --rm -p 127.0.0.1:5000:5000 --name registry registry:2
#   examples/modelpack/run.sh
set -eu
cd "$(dirname "$0")"
registry=localhost:5000
work=$(mktemp -d)
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }

step "1. Pack a Hugging Face checkpoint as a ModelPack artifact with modctl, in tar layers"
hf download HuggingFaceTB/SmolLM2-360M-Instruct --local-dir "$work/smollm2" \
    config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json \
    special_tokens_map.json vocab.json merges.txt README.md >/dev/null
modctl modelfile generate --name smollm2-360m-instruct --family llama --output "$work" "$work/smollm2"
modctl build --raw=false --plain-http --output-remote -f "$work/Modelfile" -t $registry/smollm2:modelpack "$work/smollm2"

step "2. Run the artifact as it is: shem pulls it once and keeps its image twin, over the same blobs"
shem run $registry/smollm2:modelpack --max-tokens 48 "What is a golem? Answer in one sentence."

step "3. Build on it: a golem, its system prompt read at build time"
shem build -t smollm2-golem .
shem run smollm2-golem

step "4. Publish the golem's session alone, linked to the model, and continue it on the plain artifact"
shem push --plain-http --state smollm2-golem $registry/smollm2:golem-session
shem run --state $registry/smollm2:golem-session $registry/smollm2:modelpack "What are you made of?"
# The registry lists the session among the referrers of the image it was primed on
subject=$(oras manifest fetch --plain-http $registry/smollm2:golem-session | python3 -c 'import json, sys; print(json.load(sys.stdin)["subject"]["digest"])')
oras discover --plain-http --format tree $registry/smollm2@$subject

step "5. Publish the golem as a ModelPack artifact, which modctl extracts as a Hugging Face checkpoint"
shem push --plain-http --modelpack --uncompressed smollm2-golem $registry/smollm2:golem-modelpack
modctl pull --plain-http --extract-from-remote --extract-dir "$work/extracted" $registry/smollm2:golem-modelpack
ls "$work/extracted"
test -f "$work/extracted/model.safetensors.index.json"
