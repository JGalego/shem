"""OpenAI's chat completions API, as `--serve` serves it: the HTTP side with a
stand-in for the model, then the model's side with the tiny saved model and
a tokenizer of one token per character."""

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from shem_neural.model import serve
from shem_neural.model.blocks import Model
from shem_neural.model.generate import Session
from shem_neural.model.tokenizer import Tokenizer

TINY = Path(__file__).parent / "sessions" / "model"
HI = {"messages": [{"role": "user", "content": "Hi"}]}


class Replies:
    """A stand-in for a model: says "Hello", in two pieces, to anything."""

    name = "tiny/model"

    def __init__(self):
        self.asked = []

    def complete(self, messages, max_tokens=None, stops=(), out=None):
        self.asked.append((messages, max_tokens, list(stops)))
        for piece in ("Hel", "lo"):
            if out is not None:
                out(piece)
        return {"text": "Hello", "finish": "stop", "prompt_tokens": 3, "completion_tokens": 2}


@pytest.fixture
def api():
    replies = Replies()
    server = serve.start(replies, serve.listen("127.0.0.1", 0))
    host, port = server.server_address
    yield replies, f"http://{host}:{port}"
    server.shutdown()
    server.server_close()


def call(url: str, body: dict | None = None) -> tuple[int, bytes]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_completions_answer_as_openai_does(api):
    replies, url = api
    content = [{"type": "text", "text": "Hi "}, {"type": "text", "text": "there"}]
    body = {
        "model": "anything",
        "messages": [{"role": "developer", "content": "Be brief."}, {"role": "user", "content": content}],
        "max_tokens": 5,
        "stop": "\n\n",
        "temperature": 0.7,
    }
    status, data = call(f"{url}/v1/chat/completions", body)
    assert status == 200
    reply = json.loads(data)
    assert reply["object"] == "chat.completion" and reply["model"] == "tiny/model"
    assert reply["choices"] == [
        {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
    ]
    assert reply["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    # A developer's message is a system one, and parts of text are one text
    messages, max_tokens, stops = replies.asked[0]
    assert messages == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi there"}]
    assert (max_tokens, stops) == (5, ["\n\n"])


def test_streams_send_the_reply_as_it_comes(api):
    _, url = api
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": True, "stream_options": {"include_usage": True}}
    status, data = call(f"{url}/v1/chat/completions", body)
    assert status == 200
    events = [line.removeprefix("data: ") for line in data.decode().split("\n\n") if line]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert len({chunk["id"] for chunk in chunks}) == 1
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    assert deltas == [{"role": "assistant", "content": ""}, {"content": "Hel"}, {"content": "lo"}, {}]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["total_tokens"] == 5


def test_the_model_and_health_are_listed(api):
    _, url = api
    status, data = call(f"{url}/v1/models")
    assert status == 200 and json.loads(data)["data"][0]["id"] == "tiny/model"
    assert call(f"{url}/health")[0] == 200


@pytest.mark.parametrize(
    "body",
    [
        {"messages": "Hi"},
        {"messages": []},
        {"messages": [{"role": "user", "content": "Hi"}], "n": 2},
        {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 0},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]},
        {"messages": [{"role": "tool", "content": "42"}]},
    ],
)
def test_requests_it_cannot_answer_say_why(api, body):
    _, url = api
    status, data = call(f"{url}/v1/chat/completions", body)
    assert status == 400
    assert json.loads(data)["error"]["type"] == "invalid_request_error"


def test_other_paths_are_not_found(api):
    _, url = api
    assert call(f"{url}/v1/completions", {"prompt": "Hi"})[0] == 404
    assert call(f"{url}/")[0] == 404


@pytest.mark.parametrize(
    "text, where", [("8000", ("0.0.0.0", 8000)), ("127.0.0.1:9000", ("127.0.0.1", 9000)), ("[::1]:80", ("::1", 80))]
)
def test_addresses_name_a_port_and_maybe_a_host(text, where):
    assert serve.address(text, "0.0.0.0") == where


@pytest.mark.parametrize("text", ["", "http", "1:2:x", "70000", "-1"])
def test_addresses_without_a_port_are_refused(text):
    with pytest.raises(ValueError):
        serve.address(text, "0.0.0.0")


class Characters:
    """A tokenizer of one token per character, as the tiny model's vocabulary
    allows, with a token that starts each message by its role."""

    eos: set = set()
    ROLES = {"system": 1, "user": 2, "assistant": 3}

    def chat(self, messages, tools=None):
        return self.prefix(messages) + [self.ROLES["assistant"]]

    def prefix(self, messages):
        tokens = []
        for message in messages:
            tokens += [self.ROLES[message["role"]]] + [4 + ord(c) % 60 for c in message["content"]]
        return tokens

    def decode(self, ids):
        return "".join(chr(ord("a") + i % 26) for i in ids)


def chat_of(primed=None, max_tokens=8, system="You are tiny."):
    model = Model(TINY)
    args = SimpleNamespace(max_tokens=max_tokens, stop=[], system=system)
    return serve.Chat(model, Characters(), args, primed)


def test_replies_take_the_image_system_prompt_and_stop_at_the_length():
    chat = chat_of()
    done = chat.complete([{"role": "user", "content": "Hi"}], max_tokens=4)
    assert done["prompt_tokens"] == len(
        Characters().chat([{"role": "system", "content": "You are tiny."}, {"role": "user", "content": "Hi"}])
    )
    assert (done["completion_tokens"], done["finish"]) == (4, "length")
    assert len(done["text"]) == 4
    # A request's own system prompt replaces the image's
    own = chat.complete([{"role": "system", "content": "Hush."}, {"role": "user", "content": "Hi"}], max_tokens=4)
    assert own["prompt_tokens"] == len(
        Characters().chat([{"role": "system", "content": "Hush."}, {"role": "user", "content": "Hi"}])
    )


def test_conversations_go_on_from_the_primed_session():
    tokenizer = Characters()
    system = [{"role": "system", "content": "You are tiny."}]
    cold = chat_of()
    primed = Session(cold.model)
    primed.feed(tokenizer.prefix(system))
    chat = chat_of(primed)
    prompt = tokenizer.chat(system + [{"role": "user", "content": "Hi"}])

    session = chat._start(prompt)
    assert session.tokens == primed.tokens, "the primed tokens are not read again"
    session.feed(prompt[len(session.tokens) :])
    fresh = cold._start(prompt)
    assert fresh.tokens == []
    fresh.feed(prompt)
    assert torch.allclose(session.logits, fresh.logits, atol=1e-5, rtol=1e-5)
    # Its own state is a copy: replies do not change the primed session
    assert primed.tokens == tokenizer.prefix(system)

    other = tokenizer.chat([{"role": "system", "content": "Hush."}, {"role": "user", "content": "Hi"}])
    assert chat._start(other).tokens == [], "another conversation starts afresh"


def test_the_tiny_model_answers_through_its_own_tokenizer():
    model = Model(TINY)
    args = SimpleNamespace(max_tokens=6, stop=[], system=None)
    chat = serve.Chat(model, Tokenizer(TINY, model.manifest.tokenizer), args)
    server = serve.start(chat, serve.listen("127.0.0.1", 0))
    host, port = server.server_address
    try:
        status, data = call(f"http://{host}:{port}/v1/chat/completions", HI)
    finally:
        server.shutdown()
        server.server_close()
    assert status == 200
    reply = json.loads(data)
    assert reply["usage"]["completion_tokens"] == 6 and reply["choices"][0]["finish_reason"] == "length"
    assert reply["choices"][0]["message"]["content"]


def test_the_model_server_serves_a_container_api_until_it_ends():
    from shem_neural.server import Models

    models = Models(device="cpu")
    said = []
    request = {"model": str(TINY), "argv": ["--serve", "127.0.0.1:0", "--max-tokens", "3"], "session_id": "c1"}
    models.generate(request, said.append)
    port = int(re.search(r"at port (\d+)", b"".join(said).decode()).group(1))
    status, data = call(f"http://127.0.0.1:{port}/v1/chat/completions", HI)
    assert status == 200 and json.loads(data)["usage"]["completion_tokens"] == 3
    models.close({"session_id": "c1"})
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)


def test_the_model_server_takes_no_session_paths_from_a_container(tmp_path):
    """The server may be root; a container's command line is its image's."""
    from shem_neural.server import Models

    models = Models(device="cpu")
    for argv in (["--warm", str(tmp_path / "w")], [f"--warm={tmp_path / 'w'}"], ["--session", str(tmp_path)]):
        with pytest.raises(ValueError, match="--warm or --session"):
            models.generate({"model": str(TINY), "argv": [*argv, "--system", "x", "hi"]}, lambda _: None)
    assert not (tmp_path / "w").exists()
    assert not models.loaded
