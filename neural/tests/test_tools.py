"""Tools and compiled layers, without a model: parsing tool calls, holding
them back from the output, and vCPUs that run on an image's compiled layer."""

import pytest
import torch

from shem_neural.compile import compile_weights, save
from shem_neural.export import export
from shem_neural.model.generate import _partial, reply, system_file
from shem_neural.model.tools import compiled_layers, tool_calls
from shem_neural.server import Registry


def test_tool_calls_parse_in_either_format():
    as_json = 'Sure.<tool_call>\n{"name": "mul", "arguments": {"a": "3", "b": "4"}}\n</tool_call>'
    assert tool_calls(as_json) == [("mul", {"a": "3", "b": "4"})]
    as_tags = (
        "<tool_call>\n<function=mul>\n<parameter=a>\n3\n</parameter>\n"
        "<parameter=b>\n4\n</parameter>\n</function>\n</tool_call>"
    )
    assert tool_calls(as_tags) == [("mul", {"a": "3", "b": "4"})]
    stringified = '<tool_call>{"name": "mul", "arguments": "{\\"a\\": \\"3\\"}"}</tool_call>'
    assert tool_calls(stringified) == [("mul", {"a": "3"})]
    two = as_json + "\n" + as_tags
    assert len(tool_calls(two)) == 2
    for broken in ["<tool_call>{not json}</tool_call>", '<tool_call>\n{"name": "mul"', "no call at all"]:
        assert tool_calls(broken) == []


def test_partial_tags_are_held_back():
    assert _partial("The answer is <tool", "<tool_call>") == 5
    assert _partial("The answer is 12", "<tool_call>") == 0
    assert _partial("<", "<tool_call>") == 1


def test_compiled_layers_are_found_by_their_format(tmp_path):
    export(tmp_path / "compiled" / "rv32im", "0.0.0")
    (tmp_path / "compiled" / "other").mkdir()
    (tmp_path / "compiled" / "other" / "config.json").write_text('{"format": "something else"}')
    assert list(compiled_layers(tmp_path)) == ["rv32im"]


def test_vcpus_run_on_the_compiled_layer_they_name(tmp_path):
    registry = Registry(dense=False)
    export(tmp_path / "layer", "0.0.0")
    # A layer exported by this shem is this server's own CPU, so it batches with it
    assert registry.cpu_for(tmp_path / "layer") is registry.cpu
    # Other weights get a CPU of their own
    other = compile_weights()
    other["byte_bits"] = torch.flip(other["byte_bits"], [0])
    save(other, tmp_path / "other.safetensors")
    assert registry.cpu_for(tmp_path / "other.safetensors") is not registry.cpu
    # And weights that are no CPU are refused
    save({"x": torch.zeros(2)}, tmp_path / "junk.safetensors")
    with pytest.raises(ValueError):
        registry.cpu_for(tmp_path / "junk.safetensors")


def test_system_files_are_read_from_the_image_alone(tmp_path):
    root = tmp_path / "root"
    (root / "etc").mkdir(parents=True)
    (root / "etc" / "handbook.md").write_text("# Handbook\n\nGolems are made of clay.\n")
    (tmp_path / "secret").write_text("the host's")
    (root / "etc" / "link").symlink_to(tmp_path / "secret")
    assert system_file(root, "/etc/handbook.md") == "# Handbook\n\nGolems are made of clay."
    for path in ("/../secret", "/etc/link"):
        with pytest.raises(ValueError, match="out of the image"):
            system_file(root, path)


class Scripted:
    """A session whose model says `words` in turn, a token each, whatever it
    reads, with a tokenizer whose tokens are those words."""

    def __init__(self, words: list[str]):
        self.words, self.read = words, []
        self.eos: set[int] = set()
        self.logits = self.say(0)

    def say(self, i: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(torch.tensor(min(i, len(self.words) - 1)), len(self.words)).float()

    def feed(self, tokens: list[int]) -> None:
        self.read += tokens
        self.logits = self.say(len(self.read))

    def decode(self, ids: list[int]) -> str:
        return "".join(self.words[i] for i in ids)


def test_replies_end_before_a_stop_text():
    import io

    def run(words, stops, max_tokens=8):
        session, out = Scripted(words), io.StringIO()
        tokens = reply(session, session, max_tokens, out, stops=stops)
        return out.getvalue(), tokens, session.read

    # The token that reaches the stop is neither kept nor read
    assert run([" Lisbon", "\n", "Spain", ":"], ["\n"]) == (" Lisbon\n", [0], [0])
    # A stop across tokens: its start is held back, then left out
    assert run([" Lis", "bon", "\n", "\n", "x"], ["\n\n"]) == (" Lisbon\n", [0, 1, 2], [0, 1, 2])
    # Text that only began like a stop is written after all
    assert run([" a", "\n", "b"], ["\nX"], max_tokens=3) == (" a\nb\n", [0, 1, 2], [0, 1, 2])
    assert run([" a", "b"], [], max_tokens=2) == (" ab\n", [0, 1], [0, 1])
