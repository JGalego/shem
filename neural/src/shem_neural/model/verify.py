"""Compare an imported model with transformers' own implementation.

    shem-neural-verify SOURCE MODEL [--revision REVISION] [--tokens N] [--report FILE]

SOURCE is the checkpoint the model was imported from: a Hugging Face model,
such as Qwen/Qwen3-0.6B, or a checkpoint's directory. MODEL is the imported
model's directory. Both run in float32 on a chat prompt: the prompt's logits
must agree within 1e-3, and greedy decoding, one token at a time from the
session's state, must give transformers' reply. Prints the comparison as JSON,
or writes it to FILE, and exits with status 1 if they disagree.

transformers holds the checkpoint in memory, and then shem the model, one at
a time: a 1B model's comparison fits in 16 GB.
"""

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from shem_neural.model.blocks import Model
from shem_neural.model.manifest import Manifest
from shem_neural.model.tokenizer import Tokenizer

MESSAGES = [{"role": "user", "content": "Name three uses of clay."}]
#: How far apart the prompt's logits may be, absolutely and relatively
TOLERANCE = 1e-3


@dataclass
class Comparison:
    #: The largest difference between the prompt's logits
    difference: float
    #: Whether the prompt's logits agree within the tolerance
    logits: bool
    #: The greedy replies, as text
    reply: str
    expected: str
    #: Whether the greedy replies are the same tokens
    tokens: bool
    #: How many tokens the replies have
    length: int

    @property
    def passed(self) -> bool:
        return self.logits and self.tokens


def reference(source: str, revision: str | None = None):
    """transformers' model of the checkpoint, in float32. A vision-language
    checkpoint loads as one, of which shem imports the language model."""
    import transformers

    config = transformers.AutoConfig.from_pretrained(source, revision=revision)
    vlm = any(a.endswith("ForConditionalGeneration") for a in getattr(config, "architectures", None) or [])
    model = transformers.AutoModelForImageTextToText if vlm else transformers.AutoModelForCausalLM
    return model.from_pretrained(source, revision=revision, dtype=torch.float32).eval()


def verify(source: str, directory: Path, revision: str | None = None, length: int = 24) -> Comparison:
    """Compare the model imported into `directory` with transformers' model
    of `source`, the checkpoint it was imported from."""
    import transformers

    tokenizer = Tokenizer(directory, Manifest.load(directory).tokenizer)
    prompt = tokenizer.chat(MESSAGES)
    tokens = torch.tensor([prompt])

    # transformers' logits for the prompt, and its greedy reply. The
    # checkpoint's generation defaults are replaced, not overridden, since
    # generate merges them into a config it is given: Qwen2.5's repetition
    # penalty would apply even without sampling
    ref = reference(source, revision)
    with torch.no_grad():
        expected = ref(input_ids=tokens).logits
    ref.generation_config = transformers.GenerationConfig()
    want = ref.generate(input_ids=tokens, do_sample=False, max_new_tokens=length)[0, len(prompt) :].tolist()
    del ref
    gc.collect()

    # The prompt's logits, in one pass
    model = Model(directory)
    got = model.forward(tokens, model.init_state(), 0)
    close = torch.allclose(got, expected, atol=TOLERANCE, rtol=TOLERANCE)
    difference = float((got - expected).abs().max())

    # Greedy decoding, one token at a time from the session's state
    state = model.init_state()
    logits = model.forward(tokens, state, 0)
    reply = []
    for position in range(len(prompt), len(prompt) + len(want)):
        token = int(logits[0, -1].argmax())
        reply.append(token)
        logits = model.forward(torch.tensor([[token]]), state, position)
    return Comparison(
        difference=difference,
        logits=close,
        reply=tokenizer.decode(reply),
        expected=tokenizer.decode(want),
        tokens=reply == want,
        length=len(want),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="the checkpoint: a Hugging Face model, or a checkpoint's directory")
    parser.add_argument("model", type=Path, help="the imported model's directory")
    parser.add_argument("--revision", help="a branch, tag or commit of the Hugging Face model")
    parser.add_argument("--tokens", type=int, default=24, help="how many tokens to decode greedily (default 24)")
    parser.add_argument("--report", type=Path, help="write the comparison to this file instead of stdout")
    args = parser.parse_args()
    comparison = verify(args.source, args.model, args.revision, args.tokens)
    report = json.dumps(asdict(comparison) | {"passed": comparison.passed}, indent=2)
    if args.report:
        args.report.write_text(report)
    else:
        print(report)
    sys.exit(0 if comparison.passed else 1)


if __name__ == "__main__":
    main()
