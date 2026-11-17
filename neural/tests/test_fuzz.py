"""What shem reads from outside, on arbitrary input: it may refuse, in the way it says it does, and nothing else.

These are property tests, with Hypothesis, of the Python parsers that the Rust fuzz targets
(`fuzz/`) have no counterpart for: receipts, manifests, tool calls, chat messages, addresses,
the registry's headers and sizes, the registry's digest file, the vCPU server's frames, and image references.
"""

import argparse
import io
import json
import string

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shem_neural import server
from shem_neural.model import receipt, serve
from shem_neural.model.manifest import Manifest
from shem_neural.model.tools import tool_calls
from shem_neural.modelpack import parse_reference
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


@FUZZ
@given(st.lists(st.binary(max_size=300), max_size=5))
def test_frames_come_back_whole_and_in_order(payloads):
    wire = io.BytesIO()
    for payload in payloads:
        server.write_frame(wire, payload)
    wire.seek(0)
    assert [server.read_frame(wire) for _ in payloads] == payloads
    assert server.read_frame(wire) is None, "the end of the stream is no frame"


@FUZZ
@given(st.binary(max_size=40))
def test_a_stream_that_is_cut_short_or_garbage_gives_a_frame_a_short_one_or_none(data):
    stream = io.BytesIO(data)
    frame = server.read_frame(stream)  # never an exception, whatever the bytes
    assert frame is None or isinstance(frame, bytes)


names = st.from_regex(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", fullmatch=True)


@FUZZ
@given(
    st.one_of(st.none(), st.sampled_from(["localhost", "ghcr.io", "example.com:5000", "hf.co"])),
    st.lists(names, min_size=1, max_size=3),
    st.one_of(st.none(), names),
    st.one_of(st.none(), st.just("sha256:" + "a" * 64)),
)
def test_a_reference_splits_into_where_what_and_which(host, path, tag, digest):
    name = "/".join(path)
    text = (f"{host}/" if host else "") + name
    if digest:
        text += "@" + digest
    elif tag:
        text += ":" + tag
    got_host, got_path, got_which = parse_reference(text)
    assert got_which == (digest or tag or "latest")
    if host:
        assert (got_host, got_path) == (host, name)
    elif len(path) == 1:
        assert (got_host, got_path) == ("registry-1.docker.io", f"library/{name}")
    # Whatever it was made of, a reference that names no registry never loses its path
    assert got_path.endswith(path[-1]) and isinstance(got_host, str)


@FUZZ
@given(st.text(max_size=60))
def test_any_text_is_a_reference_and_never_crashes_the_parser(text):
    host, path, which = parse_reference(text)
    assert isinstance(host, str) and isinstance(path, str) and isinstance(which, str)
