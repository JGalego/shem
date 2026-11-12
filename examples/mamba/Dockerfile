# syntax=ghcr.io/jgalego/shem/dockerfile:latest
FROM hf.co/state-spaces/mamba-130m-hf
ENTRYPOINT ["shem:generate", "--max-tokens", "16", "--stop", "\n", "--system", "France: Paris\nJapan: Tokyo\nItaly: Rome\nKenya: Nairobi\nCanada: Ottawa\n"]
CMD ["Portugal:"]
WARM --until=prompt
