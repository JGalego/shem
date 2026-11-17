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

from shem_neural.metrics import Metrics
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
        self.metrics = Metrics()
        self.tokenizer = SimpleNamespace(
            encode=lambda text: [ord(c) % 60 for c in text], decode=lambda ids: "".join(f"<{i}>" for i in ids)
        )
        self.calls = []

    def prompt(self, messages, tools=None):
        return [0] * 3

    def check(self, tokens):
        return tokens

    def complete(self, messages, max_tokens=None, stops=(), out=None, draw=None, receipt=False, tools=None, **more):
        self.asked.append((messages, max_tokens, list(stops)))
        self.drew, self.more, self.tools = draw, more, tools
        if self.fail:
            raise RuntimeError(self.fail)
        entries = [(7, -0.5, [(7, -0.5), (8, -1.5)][: more.get("logprobs") or 0]), (9, -2.0, [])]
        for piece, entry in zip(("Hel", "lo"), entries, strict=True):
            if out is not None and more.get("out_logprobs") and more.get("logprobs") is not None:
                out(piece, [entry])
            elif out is not None:
                out(piece)
        done = {
            "text": "Hello",
            "calls": self.calls if tools else [],
            "finish": "tool_calls" if self.calls and tools else "stop",
            "prompt_tokens": 3,
            "completion_tokens": 2,
        }
        if more.get("logprobs") is not None:
            done["logprobs"] = entries
        return done


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
        {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "logprobs": None, "finish_reason": "stop"}
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
        {"messages": [{"role": "tool", "content": 42}]},
    ],
)
def test_requests_it_cannot_answer_say_why(api, body):
    _, url = api
    status, data = call(f"{url}/v1/chat/completions", body)
    assert status == 400
    assert json.loads(data)["error"]["type"] == "invalid_request_error"


def test_other_paths_are_not_found(api):
    _, url = api
    assert call(f"{url}/v1/images", {"prompt": "Hi"})[0] == 404
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

    def encode(self, text):
        return [4 + ord(c) % 60 for c in text]

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
        ["--replay", str(tmp_path / "w")],
        ["--image", "/etc/passwd"],
    ):
        with pytest.raises(ValueError, match="--warm, --session, --replay or --image"):
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


def test_text_completions_continue_a_text_without_a_chat_template(api):
    replies, url = api
    status, data = call(f"{url}/v1/completions", {"prompt": "Once", "max_tokens": 5, "stop": ["\n"]})
    assert status == 200
    reply = json.loads(data)
    assert reply["object"] == "text_completion" and reply["choices"][0]["text"] == "Hello"
    assert reply["choices"][0]["finish_reason"] == "stop" and reply["usage"]["total_tokens"] == 5
    assert replies.more["prompt_tokens"] == [ord(c) % 60 for c in "Once"]
    assert replies.asked[-1][0] is None, "no messages: the prompt is the text itself"
    status, data = call(f"{url}/v1/completions", {"prompt": "Once", "stream": True})
    events = [json.loads(e.removeprefix("data: ")) for e in data.decode().split("\n\n") if e and "[DONE]" not in e]
    assert [e["choices"][0]["text"] for e in events] == ["Hel", "lo", ""] and events[0]["object"] == "text_completion"
    assert call(f"{url}/v1/completions", {"prompt": [1, 2, 3]})[0] == 400
    assert call(f"{url}/v1/completions", {})[0] == 400


def test_log_probabilities_come_back_in_the_endpoints_shapes(api):
    replies, url = api
    status, data = call(f"{url}/v1/chat/completions", HI | {"logprobs": True, "top_logprobs": 2})
    assert status == 200
    content = json.loads(data)["choices"][0]["logprobs"]["content"]
    assert content[0]["token"] == "<7>" and content[0]["logprob"] == -0.5
    assert [t["logprob"] for t in content[0]["top_logprobs"]] == [-0.5, -1.5] and content[1]["top_logprobs"] == []
    status, data = call(f"{url}/v1/completions", {"prompt": "Hi", "logprobs": 1})
    shown = json.loads(data)["choices"][0]["logprobs"]
    assert shown["tokens"] == ["<7>", "<9>"] and shown["token_logprobs"] == [-0.5, -2.0]
    assert shown["text_offset"] == [0, 3] and shown["top_logprobs"][0] == {"<7>": -0.5}
    assert json.loads(call(f"{url}/v1/chat/completions", HI)[1])["choices"][0]["logprobs"] is None
    # In a stream, each chunk brings the log probabilities of its tokens
    status, data = call(f"{url}/v1/chat/completions", HI | {"logprobs": True, "top_logprobs": 1, "stream": True})
    events = [json.loads(e.removeprefix("data: ")) for e in data.decode().split("\n\n") if e and "[DONE]" not in e]
    shown = [e["choices"][0].get("logprobs") for e in events]
    assert status == 200 and shown[0] is None and shown[-1] is None
    assert [c["token"] for e in shown if e for c in e["content"]] == ["<7>", "<9>"]
    status, data = call(f"{url}/v1/completions", {"prompt": "Hi", "logprobs": 1, "stream": True})
    events = [json.loads(e.removeprefix("data: ")) for e in data.decode().split("\n\n") if e and "[DONE]" not in e]
    assert [t for e in events if e["choices"][0]["logprobs"] for t in e["choices"][0]["logprobs"]["tokens"]] == [
        "<7>",
        "<9>",
    ]
    assert call(f"{url}/v1/chat/completions", HI | {"logprobs": True, "top_logprobs": 99})[0] == 400


def test_tools_are_offered_and_their_calls_come_back(api):
    replies, url = api
    replies.calls = [("mul", {"a": 6, "b": 7})]
    tool = {
        "type": "function",
        "function": {"name": "mul", "description": "Multiply", "parameters": {"type": "object"}},
    }
    status, data = call(f"{url}/v1/chat/completions", HI | {"tools": [tool]})
    assert status == 200 and replies.tools == [tool]
    choice = json.loads(data)["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    made = choice["message"]["tool_calls"][0]
    assert made["type"] == "function" and made["function"] == {"name": "mul", "arguments": '{"a": 6, "b": 7}'}
    status, data = call(f"{url}/v1/chat/completions", HI | {"tools": [tool], "stream": True})
    events = [json.loads(e.removeprefix("data: ")) for e in data.decode().split("\n\n") if e and "[DONE]" not in e]
    calls = next(e for e in events if "tool_calls" in e["choices"][0]["delta"])["choices"][0]["delta"]["tool_calls"]
    assert calls[0]["index"] == 0 and calls[0]["function"]["name"] == "mul"
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    # Not offered, or forbidden, they are not called
    assert json.loads(call(f"{url}/v1/chat/completions", HI)[1])["choices"][0]["finish_reason"] == "stop"
    forbidden = HI | {"tools": [tool], "tool_choice": "none"}
    assert json.loads(call(f"{url}/v1/chat/completions", forbidden)[1])["choices"][0]["finish_reason"] == "stop"
    assert call(f"{url}/v1/chat/completions", HI | {"tools": [tool], "tool_choice": "required"})[0] == 400
    assert call(f"{url}/v1/chat/completions", HI | {"tools": [{"type": "retrieval"}]})[0] == 400


def test_a_conversation_with_tool_calls_and_replies_is_what_the_template_takes():
    made = serve.message(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "mul", "arguments": '{"a": 6}'}}],
        }
    )
    assert made == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "mul", "arguments": {"a": 6}}}],
    }
    assert serve.message({"role": "tool", "tool_call_id": "c1", "content": "42"}) == {"role": "tool", "content": "42"}
    with pytest.raises(ValueError):
        serve.message({"role": "assistant", "tool_calls": [{"function": {"name": "m", "arguments": "[1]"}}]})


def test_json_object_replies_are_asked_for_and_other_formats_are_refused(api):
    replies, url = api
    assert call(f"{url}/v1/chat/completions", HI | {"response_format": {"type": "text"}})[0] == 200
    assert replies.more["json_object"] is False
    assert call(f"{url}/v1/chat/completions", HI | {"response_format": {"type": "json_object"}})[0] == 200
    assert replies.more["json_object"] is True
    mine = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    named = {"type": "json_schema", "json_schema": {"name": "x", "schema": mine, "strict": True}}
    assert call(f"{url}/v1/chat/completions", HI | {"response_format": named})[0] == 200
    assert replies.more["schema"] == mine
    for bad in ({"name": "x", "schema": {"type": "object", "pattern": "a"}}, {"name": "x"}, {"schema": {}}):
        status, data = call(
            f"{url}/v1/chat/completions", HI | {"response_format": {"type": "json_schema", "json_schema": bad}}
        )
        assert status == 400 and json.loads(data)["error"]["message"], bad
    json_mode = {"response_format": {"type": "json_object"}}
    assert call(f"{url}/v1/completions", {"prompt": "x"} | json_mode)[0] == 400
    assert call(f"{url}/v1/chat/completions", HI | json_mode | {"shem_receipt": True})[0] == 400


class JsonCharacters(Characters):
    """Characters that can write JSON, and a token that ends the reply."""

    ALPHABET = '{}[]":,. -0123456789abcdefghijklmnopqrstuvwxyz'
    eos = {0}

    def decode(self, ids):
        return "".join(self.ALPHABET[i % len(self.ALPHABET)] for i in ids)


def test_a_real_model_writes_an_object_that_fits_a_schema():
    args = SimpleNamespace(max_tokens=300, stop=[], system=None)
    chat = serve.Chat(Model(TINY), JsonCharacters(), args)
    plain = chat.complete([{"role": "user", "content": "Hi"}], json_object=True)
    assert isinstance(json.loads(plain["text"]), dict)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}, "n": {"type": "integer"}, "kind": {"enum": ["a", "b"]}},
        "required": ["ok", "n", "kind"],
    }
    done = chat.complete([{"role": "user", "content": "Hi"}], schema=schema)
    value = json.loads(done["text"])
    assert set(value) == {"ok", "n", "kind"} and isinstance(value["ok"], bool) and isinstance(value["n"], int)
    assert value["kind"] in ("a", "b")


def test_a_real_model_gives_log_probabilities_and_continues_raw_text():
    chat = chat_of(max_tokens=5)
    done = chat.complete([{"role": "user", "content": "Hi"}], logprobs=2)
    assert len(done["logprobs"]) == done["completion_tokens"] == 5
    for token, logprob, top in done["logprobs"]:
        assert logprob <= 0 and len(top) == 2
        assert top[0][0] == token and abs(top[0][1] - logprob) < 1e-6, "greedy chooses the likeliest"
        assert top[0][1] >= top[1][1]
    streamed, pieces = [], []
    again = chat.complete(
        [{"role": "user", "content": "Hi"}],
        logprobs=2,
        out=lambda piece, entries: (pieces.append(piece), streamed.extend(entries)),
        out_logprobs=True,
    )
    assert streamed == again["logprobs"] == done["logprobs"], "a stream carries every token's, once, in order"
    assert "logprobs" not in chat.complete([{"role": "user", "content": "Hi"}])
    raw = chat.complete(None, prompt_tokens=Characters().encode("Once upon"), max_tokens=4)
    assert raw["completion_tokens"] == 4 and raw["prompt_tokens"] == len("Once upon")
    assert not raw["text"].startswith("\n") and raw["calls"] == []


def test_embeddings_are_unit_vectors_of_what_the_model_made_of_a_text():
    import base64
    import struct

    model = Model(TINY)
    args = SimpleNamespace(max_tokens=4, stop=[], system=None)
    chat = serve.Chat(model, Characters(), args)
    server = serve.start(chat, serve.listen("127.0.0.1", 0))
    host, port = server.server_address
    url = f"http://{host}:{port}/v1/embeddings"
    try:
        status, data = call(url, {"input": ["Hi there", "Another text"]})
        assert status == 200
        reply = json.loads(data)
        assert reply["object"] == "list" and [d["index"] for d in reply["data"]] == [0, 1]
        one, two = (d["embedding"] for d in reply["data"])
        assert len(one) == model.manifest.hidden_size
        assert abs(sum(x * x for x in one) - 1) < 1e-4 and one != two
        assert reply["usage"]["prompt_tokens"] == len("Hi there") + len("Another text")
        # Deterministic, a text alone is what it is among others, and mean pooling is another vector
        alone = json.loads(call(url, {"input": "Hi there"})[1])["data"][0]["embedding"]
        assert alone == one
        mean = json.loads(call(url, {"input": "Hi there", "shem_pooling": "mean"})[1])["data"][0]["embedding"]
        assert mean != one and abs(sum(x * x for x in mean) - 1) < 1e-4
        # The same numbers, packed as base64 floats
        packed = json.loads(call(url, {"input": "Hi there", "encoding_format": "base64"})[1])["data"][0]["embedding"]
        floats = struct.unpack(f"<{len(one)}f", base64.b64decode(packed))
        assert all(abs(a - b) < 1e-6 for a, b in zip(floats, one, strict=True))
        # And what it cannot do is said
        for bad in (
            {"input": []},
            {"input": [1, 2]},
            {"input": "x", "dimensions": 8},
            {"input": "x", "shem_pooling": "max"},
        ):
            assert call(url, bad)[0] == 400
        assert call(url, {"input": ""})[0] == 400
    finally:
        server.shutdown()
        server.server_close()


def test_the_hidden_states_are_what_the_head_reads():
    model = Model(TINY)
    tokens = torch.tensor([[3, 17, 42, 8]])
    hidden = model.hidden(tokens, model.init_state(), 0)
    logits = model.forward(tokens, model.init_state(), 0)
    assert hidden.shape == (1, 4, model.manifest.hidden_size)
    assert torch.allclose(hidden @ model.head_weight.T, logits, atol=1e-5)


def test_the_api_serves_prometheus_metrics_of_what_it_has_done():
    chat = chat_of(max_tokens=4)
    server = serve.start(chat, serve.listen("127.0.0.1", 0))
    host, port = server.server_address
    url = f"http://{host}:{port}"
    try:
        assert call(f"{url}/v1/chat/completions", HI)[0] == 200
        assert call(f"{url}/v1/chat/completions", {"messages": []})[0] == 400
        assert call(f"{url}/not/a/path")[0] == 404
        status, data = call(f"{url}/metrics")
        assert status == 200 and call.headers["Content-Type"].startswith("text/plain")
        text = data.decode()
        assert "# TYPE shem_requests_total counter" in text
        assert 'shem_requests_total{path="/v1/chat/completions",status="200"} 1' in text
        assert 'shem_requests_total{path="/v1/chat/completions",status="400"} 1' in text
        assert 'shem_requests_total{path="other",status="404"} 1' in text, "unknown paths are one label, not a flood"
        assert 'shem_tokens_total{kind="completion"} 4' in text
        assert 'shem_request_seconds_count{path="/v1/chat/completions"} 2' in text
        assert 'shem_batch_size_bucket{le="1"}' in text and "shem_requests_waiting 0" in text
        assert 'shem_model_info{model="' in text
    finally:
        server.shutdown()
        server.server_close()
    # A key is asked for, as for the rest, when there is one
    secured = Replies(api_key="s3cret")
    server, url = serving(secured)
    try:
        assert call(f"{url}/metrics")[0] == 401
        assert call(f"{url}/metrics", headers={"Authorization": "Bearer s3cret"})[0] == 200
    finally:
        server.shutdown()
        server.server_close()
