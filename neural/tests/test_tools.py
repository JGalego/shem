"""Tools and compiled layers, without a model: parsing tool calls, holding
them back from the output, and vCPUs that run on an image's compiled layer."""

import pytest
import torch

from shem_neural.compile import compile_weights, save
from shem_neural.export import export
from shem_neural.model.generate import _partial
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
