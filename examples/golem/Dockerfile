# syntax=ghcr.io/jgalego/shem/dockerfile:latest
FROM hf.co/Qwen/Qwen3.5-0.8B
ENTRYPOINT ["shem:generate", "--max-tokens", "96", "--system", "You are Shem, a golem made of clay and brought to life by a word. Answer in one or two short sentences, in character."]
CMD ["Who are you?"]
WARM --until=prompt
