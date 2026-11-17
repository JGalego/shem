"""OpenAI's chat completions API, as `--serve` serves it: the HTTP side with a
stand-in for the model, then the model's side with the tiny saved model and
a tokenizer of one token per character."""

import json
import re
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from shem_neural.model import serve
from shem_neural.model.blocks import Model
from shem_neural.model.generate import ContextFull, Session
from shem_neural.model.tokenizer import Tokenizer

TINY = Path(__file__).parent / "sessions" / "model"
HI = {"messages": [{"role": "user", "content": "Hi"}]}


class Replies:
    """A stand-in for a model: says "Hello", in two pieces, to anything."""

    name = "tiny/model"

    def __init__(self, api_key=None, cors=(), fail=None):
        self.asked = []
        self.api_key, self.cors, self.fail = api_key, list(cors), fail
        self.waiting = threading.BoundedSemaphore(serve.MAX_WAITING)

    def prompt(self, messages):
        return [0] * 3

    def complete(self, messages, max_tokens=None, stops=(), out=None, draw=None, receipt=False):
        self.asked.append((messages, max_tokens, list(stops)))
        self.drew = draw
        if self.fail:
            raise RuntimeError(self.fail)
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


def call(url: str, body: dict | None = None, headers: dict | None = None, method: str | None = None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data, {"Content-Type": "application/json", **(headers or {})}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            call.headers = response.headers
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        call.headers = error.headers
        return error.code, error.read()


def serving(replies):
    server = serve.start(replies, serve.listen("127.0.0.1", 0))
    host, port = server.server_address
    return server, f"http://{host}:{port}"


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
    for argv in (
        ["--warm", str(tmp_path / "w")],
        [f"--warm={tmp_path / 'w'}"],
        ["--session", str(tmp_path)],
        ["--receipt", str(tmp_path / "w")],
        ["--replay", str(tmp_path / "w")],
    ):
        with pytest.raises(ValueError, match="--warm, --session, --receipt or --replay"):
            models.generate({"model": str(TINY), "argv": [*argv, "--system", "x", "hi"]}, lambda _: None)
    assert not (tmp_path / "w").exists()
    assert not models.loaded


def test_models_with_no_fingerprint_are_kept_per_directory_and_released_with_their_last_container(tmp_path):
    """A model that names no fingerprint cannot be shown to be another's:
    each container's is its own directory's."""
    import shutil

    from shem_neural.server import Models

    copy = tmp_path / "copy"
    shutil.copytree(TINY, copy)
    models = Models(device="cpu")
    first = models.get(TINY, "float32", "c1")[0]
    assert models.get(copy, "float32", "c2")[0] is not first
    assert models.get(TINY, "float32", "c3")[0] is first
    models.release("c1")
    assert len(models.loaded) == 2
    models.release("c3")
    models.release("c2")
    assert not models.loaded


def test_a_conversation_past_the_context_window_is_refused_and_a_reply_stops_at_its_end():
    chat = chat_of(max_tokens=50)
    chat.model.manifest.context = 20
    long = [{"role": "user", "content": "x" * 30}]
    with pytest.raises(ContextFull, match="context window is 20"):
        chat.complete(long)
    # A reply ends where the window does, saying it ran out of length
    done = chat.complete([{"role": "user", "content": "Hi"}])
    assert done["finish"] == "length"
    assert done["prompt_tokens"] + done["completion_tokens"] <= 20 and done["completion_tokens"] > 0

    server = serve.start(chat, serve.listen("127.0.0.1", 0))
    host, port = server.server_address
    try:
        for stream in (False, True):
            status, data = call(f"http://{host}:{port}/v1/chat/completions", {"messages": long, "stream": stream})
            assert status == 400
            assert json.loads(data)["error"]["code"] == "context_length_exceeded"
    finally:
        server.shutdown()
        server.server_close()


def test_a_session_refuses_tokens_past_the_window():
    session = Session(Model(TINY))
    assert session.room() is None, "the tiny model states no window"
    session.model.manifest.context = 5
    session.feed([1, 2, 3])
    assert session.room() == 2
    with pytest.raises(ContextFull):
        session.feed([4, 5, 6])
    assert session.tokens == [1, 2, 3], "a refused feed changes nothing"


def test_a_key_is_required_when_there_is_one():
    server, url = serving(Replies(api_key="s3cret"))
    try:
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic s3cret"}):
            status, data = call(f"{url}/v1/chat/completions", HI, headers)
            assert status == 401 and json.loads(data)["error"]["code"] == "invalid_api_key"
        assert call(f"{url}/v1/models")[0] == 401
        assert call(f"{url}/health")[0] == 200, "probes need no key"
        good = {"Authorization": "Bearer s3cret"}
        assert call(f"{url}/v1/chat/completions", HI, good)[0] == 200
        assert call(f"{url}/v1/models", headers=good)[0] == 200
    finally:
        server.shutdown()
        server.server_close()


def test_pages_call_the_api_only_from_the_origins_allowed():
    server, url = serving(Replies(cors=["https://app.example"]))
    try:
        ask = {"Origin": "https://app.example"}
        assert call(f"{url}/v1/chat/completions", HI, ask)[0] == 200
        assert call.headers["Access-Control-Allow-Origin"] == "https://app.example"
        assert call(f"{url}/v1/chat/completions", HI, {"Origin": "https://evil.example"})[0] == 200
        assert call.headers["Access-Control-Allow-Origin"] is None, "the reply is not readable there"
        status, _ = call(
            f"{url}/v1/chat/completions", headers=ask | {"Access-Control-Request-Method": "POST"}, method="OPTIONS"
        )
        assert status == 204 and "Authorization" in call.headers["Access-Control-Allow-Headers"]
        assert (
            call(f"{url}/v1/chat/completions", headers={"Origin": "https://evil.example"}, method="OPTIONS")[0] == 403
        )
    finally:
        server.shutdown()
        server.server_close()
    server, url = serving(Replies(cors=["*"]))
    try:
        assert call(f"{url}/v1/models", headers={"Origin": "https://any.example"})[0] == 200
        assert call.headers["Access-Control-Allow-Origin"] == "*"
    finally:
        server.shutdown()
        server.server_close()


def test_a_body_is_a_size_the_api_takes(api):
    _, url = api
    host, port = url.removeprefix("http://").split(":")
    for length, status in (("-5", 400), ("abc", 400), (str(serve.MAX_BODY + 1), 413)):
        with socket.create_connection((host, int(port)), timeout=10) as raw:
            raw.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: {length}\r\n\r\n".encode())
            assert raw.recv(4096).startswith(f"HTTP/1.1 {status}".encode()), length


def test_a_failure_after_the_headers_of_a_stream_is_an_event():
    server, url = serving(Replies(fail="the template refused"))
    try:
        status, data = call(f"{url}/v1/chat/completions", HI | {"stream": True})
        assert status == 200
        events = [line.removeprefix("data: ") for line in data.decode().split("\n\n") if line]
        assert events[-1] == "[DONE]"
        assert "the template refused" in json.loads(events[-2])["error"]["message"]
        assert call(f"{url}/v1/chat/completions", HI)[0] == 500
    finally:
        server.shutdown()
        server.server_close()


def test_requests_past_the_queue_are_told_to_retry(monkeypatch):
    monkeypatch.setattr(serve, "MAX_WAITING", 1)
    replies = Replies()
    replies.waiting = threading.BoundedSemaphore(1)
    server, url = serving(replies)
    try:
        replies.waiting.acquire()  # the model's turn is taken, and its queue is full
        status, data = call(f"{url}/v1/chat/completions", HI)
        assert status == 429 and json.loads(data)["error"]["code"] == "busy"
        assert call.headers["Retry-After"] == "5"
        replies.waiting.release()
        assert call(f"{url}/v1/chat/completions", HI)[0] == 200
    finally:
        server.shutdown()
        server.server_close()


def test_a_model_stays_loaded_while_it_lingers_and_is_evicted_after():
    from shem_neural.server import Models

    models = Models(device="cpu", linger=60)
    first = models.get(TINY, "float32", "run-1")[0]
    models.release("run-1")
    assert len(models.loaded) == 1, "kept for the next run"
    assert models.get(TINY, "float32", "run-2")[0] is first, "which finds it loaded"
    models.release("run-2")
    key = next(iter(models.idle))
    assert models.evict(models.idle[key] + 30) == []
    assert models.evict(models.idle[key] + 61) == [key]
    assert not models.loaded and not models.idle


def test_a_model_in_use_is_not_evicted_however_long_it_lingers():
    from shem_neural.server import Models

    models = Models(device="cpu", linger=1)
    models.get(TINY, "float32", "run-1")
    models.get(TINY, "float32", "run-2")
    models.release("run-1")
    assert models.evict(1e9) == [] and len(models.loaded) == 1


def test_a_server_that_nothing_uses_exits_and_leaves_no_socket(tmp_path):
    import time

    from shem_neural.server import serve

    socket_path = tmp_path / "idle.sock"
    started = time.monotonic()
    serve(socket_path, keep_warm=0.5, idle_exit=1.0)  # returns when the server exits by itself
    assert 1.0 <= time.monotonic() - started < 15
    assert not socket_path.exists()


def test_a_slow_load_holds_up_only_those_who_want_that_model(tmp_path, monkeypatch):
    import shutil

    from shem_neural.model import blocks
    from shem_neural.server import Models

    slow = tmp_path / "slow"
    shutil.copytree(TINY, slow)
    release, started = threading.Event(), threading.Event()
    real = blocks.Model

    def model(directory, **kwargs):
        if Path(directory) == slow:
            started.set()
            assert release.wait(30)
        return real(directory, **kwargs)

    monkeypatch.setattr(blocks, "Model", model)
    models = Models(device="cpu")
    loaded = []
    loader = threading.Thread(target=lambda: loaded.append(models.get(slow, "float32", "c1")))
    loader.start()
    assert started.wait(30)
    quick = models.get(TINY, "float32", "c2")[0]  # not held up by the other's load
    assert quick is not None and not loaded
    release.set()
    loader.join(30)
    assert loaded and len(models.loaded) == 2


def prompts():
    tokenizer = Characters()
    return [tokenizer.chat([{"role": "user", "content": text}]) for text in ("Hi", "Who are you, golem?", "Say it")]


def test_sessions_read_a_token_each_in_one_pass_as_they_do_alone():
    model = Model(TINY)
    alone, together = [], []
    for prompt in prompts():
        for sessions in (alone, together):
            session = Session(model)
            session.feed(prompt)
            sessions.append(session)
    tokens = [7, 20, 33]
    for _ in range(4):
        for session, token in zip(alone, tokens, strict=True):
            session.feed([token])
        Session.feed_batch(together, tokens)
        tokens = [(t * 5 + 1) % 60 for t in tokens]
        for a, b in zip(alone, together, strict=True):
            assert a.tokens == b.tokens
            assert torch.allclose(a.logits, b.logits, atol=1e-5, rtol=1e-5)


def test_replies_made_together_are_the_replies_made_alone():
    chat = chat_of(max_tokens=10)
    tokenizer, model = Characters(), chat.model
    sessions, replies = [], []
    for prompt in prompts():
        session = Session(model)
        session.feed(prompt)
        sessions.append(session)
        replies.append(serve.Job())
        replies[-1].made = serve.Reply(session, tokenizer, 10, serve._Text(replies[-1].pieces.put))
    active = list(replies)
    sizes = []
    while active:
        active = serve.Batcher._step(active)
        sizes.append(len(active))
    assert max(sizes) == 3, "all three read in one pass"
    alone = []
    for prompt in prompts():
        session = Session(model)
        session.feed(prompt)
        alone.append(generate_reply(session, tokenizer, 10))
    assert [job.made.tokens for job in replies] == alone


def generate_reply(session, tokenizer, max_tokens):
    import io

    from shem_neural.model.generate import reply

    return reply(session, tokenizer, max_tokens, io.StringIO())


def test_requests_answered_at_once_get_the_replies_they_would_alone():
    chat = chat_of(max_tokens=8)
    asked = [[{"role": "user", "content": text}] for text in ("Hi", "Who are you, golem?", "Say it", "Hello there")]
    alone = [chat.complete(messages) for messages in asked]
    together = [None] * len(asked)

    def ask(i):
        together[i] = chat.complete(asked[i])

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(len(asked))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert together == alone


def test_a_client_that_goes_away_mid_reply_is_dropped_and_the_others_go_on():
    chat = chat_of(max_tokens=30)
    gone = []

    def leaves(text):
        gone.append(text)
        raise BrokenPipeError

    with pytest.raises(BrokenPipeError):
        chat.complete([{"role": "user", "content": "Hi"}], out=leaves)
    done = chat.complete([{"role": "user", "content": "Hi again"}])
    assert done["completion_tokens"] == 30 and len(gone) == 1


def test_a_reply_that_fails_ends_its_request_with_the_error(monkeypatch):
    chat = chat_of()

    def broken(*_):
        raise RuntimeError("no memory")

    monkeypatch.setattr(serve.Session, "feed_batch", staticmethod(broken))
    with pytest.raises(RuntimeError, match="no memory"):
        chat.complete([{"role": "user", "content": "Hi"}])


def test_a_reply_can_come_with_a_receipt_that_replays_it():
    import io

    from shem_neural.model import receipt

    model = Model(TINY)
    args = SimpleNamespace(max_tokens=6, stop=[], system="You are tiny.", subject=["image=tiny@sha256:ab"])
    tokenizer = Characters()
    chat = serve.Chat(model, tokenizer, args)
    done = chat.complete([{"role": "user", "content": "Hi"}], draw={"temperature": 1.2, "seed": 3}, receipt=True)
    written = done["receipt"]
    assert written["image"] == "tiny@sha256:ab" and written["sampling"]["seed"] == 3
    assert written["messages"][0] == {"role": "system", "content": "You are tiny."}
    assert written["prompt"] == "Hi" and written["reply"] == done["text"]
    assert len(written["tokens"]) == len(written["margins"]) == done["completion_tokens"]
    status, why = receipt.replay_messages(model, tokenizer, written, io.StringIO())
    assert status == receipt.SAME, why
    forged = {**written, "tokens": [(t + 1) % 60 for t in written["tokens"]]}
    assert receipt.replay_messages(model, tokenizer, forged)[0] in (receipt.DIFFERENT, receipt.UNDECIDED)
    assert "receipt" not in chat.complete([{"role": "user", "content": "Hi"}])


def test_the_api_returns_a_receipt_when_asked_to():
    chat = chat_of()
    server = serve.start(chat, serve.listen("127.0.0.1", 0))
    host, port = server.server_address
    try:
        body = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3, "shem_receipt": True}
        status, data = call(f"http://{host}:{port}/v1/chat/completions", body)
        assert status == 200 and json.loads(data)["shem_receipt"]["format"] == "shem.receipt/v1"
        status, data = call(f"http://{host}:{port}/v1/chat/completions", body | {"stream": True})
        events = [json.loads(e.removeprefix("data: ")) for e in data.decode().split("\n\n") if e and "[DONE]" not in e]
        assert events[-1]["shem_receipt"]["reply"]
        status, data = call(f"http://{host}:{port}/v1/chat/completions", {"messages": body["messages"]})
        assert "shem_receipt" not in json.loads(data)
    finally:
        server.shutdown()
        server.server_close()
