"""What shem reads from outside, on arbitrary input: it may refuse, in the way it says it does, and nothing else.

These are property tests, with Hypothesis, of the Python parsers that the Rust fuzz targets
(`fuzz/`) have no counterpart for: receipts, manifests, tool calls, chat messages, addresses,
the registry's headers and sizes, and the registry's digest file.
"""

import argparse
import json
import string

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shem_neural.model import receipt, serve
from shem_neural.model.manifest import Manifest
from shem_neural.model.tools import tool_calls
from shem_neural.registry import Digests, byte_size, credential

FUZZ = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])

#: Any JSON value, as a parser of JSON files meets one
json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(),
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=8), inner, max_size=4),
    max_leaves=20,
)


@FUZZ
@given(st.text())
def test_a_receipt_file_is_a_receipt_or_refused_in_words(tmp_path_factory, text):
    path = tmp_path_factory.mktemp("r") / "receipt.json"
    path.write_text(text)
    try:
        loaded = receipt.load(path)
    except ValueError as error:
        assert "receipt" in str(error)
    else:
        assert loaded["format"] == receipt.FORMAT and len(loaded["tokens"]) == len(loaded["margins"])


@FUZZ
@given(json_values)
def test_a_receipt_that_is_any_json_is_a_receipt_or_refused(tmp_path_factory, value):
    path = tmp_path_factory.mktemp("r") / "receipt.json"
    path.write_text(json.dumps(value))
    try:
        receipt.load(path)
    except ValueError:
        pass


@FUZZ
@given(json_values)
def test_a_manifest_is_loaded_or_refused_never_crashed_on(tmp_path_factory, value):
    directory = tmp_path_factory.mktemp("m")
    (directory / "shem.json").write_text(json.dumps(value))
    try:
        Manifest.load(directory)
    except (ValueError, KeyError, TypeError, AttributeError):
        pass  # a file that is not a manifest: refused with what is missing


@FUZZ
@given(st.text())
def test_tool_calls_in_any_reply_are_found_or_there_are_none(text):
    for name, arguments in tool_calls(text):
        assert isinstance(name, str) and isinstance(arguments, dict)


@FUZZ
@given(st.text(alphabet=string.printable + "é中"), st.integers(-5, 70000))
def test_an_api_address_is_a_port_or_refused(text, port):
    for candidate in (text, f"{text}:{port}", str(port)):
        try:
            host, number = serve.address(candidate, "127.0.0.1")
        except ValueError:
            continue
        assert 0 <= number < 65536 and host


@FUZZ
@given(json_values)
def test_a_chat_message_is_a_message_or_a_refusal(raw):
    try:
        made = serve.message(raw)
    except (ValueError, KeyError, TypeError, AttributeError):
        return
    assert made["role"] in ("system", "user", "assistant", "tool") and isinstance(made["content"], str)


@FUZZ
@given(st.text())
def test_a_registry_credential_is_a_token_or_none(header):
    found = credential(header)
    assert found is None or isinstance(found, str)


@FUZZ
@given(st.text(max_size=12))
def test_a_size_is_a_number_of_bytes_or_refused(text):
    try:
        size = byte_size(text)
    except (ValueError, argparse.ArgumentTypeError):
        pass
    else:
        assert size >= 0


@FUZZ
@given(st.binary(max_size=200))
def test_a_digest_file_that_is_garbage_is_forgotten(tmp_path_factory, data):
    path = tmp_path_factory.mktemp("d") / "digests.json"
    path.write_bytes(data)
    assert isinstance(dict(Digests(path)), dict)
