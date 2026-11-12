# syntax=ghcr.io/jgalego/shem/dockerfile:latest
# Gemma is gated: pass your Hugging Face token as a build secret
#   docker build --platform linux/riscv32 --secret id=HF_TOKEN,env=HF_TOKEN -t gemma3 examples/gemma3
FROM hf.co/google/gemma-3-1b-it
ENTRYPOINT ["shem:generate", "--max-tokens", "96"]
CMD ["What is a golem? Answer in one sentence."]
