# syntax=ghcr.io/jgalego/shem/dockerfile:latest
FROM hf.co/allenai/OLMo-2-0425-1B-Instruct
COPY handbook.md /etc/handbook.md
ENTRYPOINT ["shem:generate", "--max-tokens", "128", "--system", "Answer questions about shem from its handbook, below, in one or two sentences.", "--system-file", "/etc/handbook.md"]
CMD ["How do I share a model image?"]
WARM --until=prompt
