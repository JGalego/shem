"""Serve a model over HTTP as OpenAI's chat completions API does, for the
clients and tools written for that API:

    POST /v1/chat/completions   a reply to the messages, streamed with "stream": true
    GET  /v1/models             the model, by the repository it came from
    GET  /health                200 once the model is loaded

Replies are greedy, as the driver's are, so sampling settings are accepted
and ignored. Requests take turns on the model. The image's system prompt
comes first unless a request gives its own, and a conversation that starts
with the image's primed session, token for token, goes on from it rather
than reading those tokens again.

`shem serve IMAGE` serves on the host. In a container, `--serve PORT`
serves in the container's network, where `docker run -p` publishes it.
"""

import contextlib
import copy
import json
import os
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .generate import Session, reply


class Chat:
    """A model and its tokenizer, the session the image primed, if it has
    one, and the driver's options: what requests take turns on."""

    def __init__(self, model, tokenizer, args, primed: Session | None = None, compute=contextlib.nullcontext):
        """`compute` makes the context a reply is computed in, as the model
        server's, which gives a model every core."""
        self.model, self.tokenizer, self.compute = model, tokenizer, compute
        self.name = model.manifest.source.get("repo") or model.directory.name
        self.max_tokens = args.max_tokens
        self.stops = list(args.stop)
        self.system = args.system
        self.primed = primed if primed is not None and primed.tokens else None
        self.lock = threading.Lock()

    def complete(self, messages: list[dict], max_tokens: int | None = None, stops=(), out=None) -> dict:
        """Reply to `messages`, writing the text to `out` as it comes. Returns
        the reply, why it ended, and how many tokens each side took."""
        if self.system and not (messages and messages[0]["role"] == "system"):
            messages = [{"role": "system", "content": self.system}] + messages
        prompt = self.tokenizer.chat(messages)
        max_tokens = max_tokens or self.max_tokens
        text = _Text(out)
        with self.lock, self.compute():
            session = self._start(prompt)
            session.feed(prompt[len(session.tokens) :])
            tokens = reply(session, self.tokenizer, max_tokens, text, stops=[*self.stops, *stops])
        return {
            "text": text.value.strip(),
            "finish": "length" if len(tokens) >= max_tokens else "stop",
            "prompt_tokens": len(prompt),
            "completion_tokens": len(tokens),
        }

    def _start(self, prompt: list[int]) -> Session:
        """A session for `prompt`: a copy of the primed one if the prompt goes
        on from it, otherwise a new one."""
        primed = self.primed
        if primed is None or len(prompt) <= len(primed.tokens) or prompt[: len(primed.tokens)] != primed.tokens:
            return Session(self.model)
        session = Session(self.model)
        session.state = copy.deepcopy(primed.state)
        session.tokens = list(primed.tokens)
        return session


class _Text:
    """What a reply writes, kept, and passed on to `out`, but for the line
    end it closes with, held back until more text follows."""

    def __init__(self, out=None):
        self.out, self.value, self.held = out, "", ""

    def write(self, text: str) -> None:
        if not text:
            return
        self.value += text
        if self.out is not None:
            text, self.held = self.held + text, ""
            if text.endswith("\n"):
                text, self.held = text[:-1], "\n"
            if text:
                self.out(text)

    def flush(self) -> None:
        pass


def message(raw: dict) -> dict:
    """A chat message as the template takes it: its role and its text, whose
    parts, if it has them, are joined. Only text is understood."""
    role, content = raw["role"], raw.get("content") or ""
    if isinstance(content, list):
        if any(part.get("type") != "text" for part in content):
            raise ValueError("only text content is understood")
        content = "".join(part["text"] for part in content)
    if role not in ("system", "user", "assistant", "developer") or not isinstance(content, str):
        raise ValueError(f"a message must be a system, user or assistant's text, not {role!r}")
    return {"role": "system" if role == "developer" else role, "content": content}


def handler(chat: Chat):
    """The HTTP request handler for `chat`."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "shem"

        def log_message(self, format, *args):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                self._json(200, {"status": "ok"})
            elif path in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [self._model()]})
            elif path.startswith(("/v1/models/", "/models/")):
                self._json(200, self._model())
            else:
                self._error(404, f"nothing at {path}: the API is at /v1/chat/completions")

        def do_POST(self):
            path = self.path.split("?")[0]
            if path not in ("/v1/chat/completions", "/chat/completions"):
                return self._error(404, f"nothing at {path}: the API is at /v1/chat/completions")
            try:
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length))
                messages = [message(m) for m in request["messages"]]
                if not messages:
                    raise ValueError("there are no messages to reply to")
                max_tokens = request.get("max_completion_tokens") or request.get("max_tokens")
                if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens < 1):
                    raise ValueError("max_tokens must be a positive number")
                stops = request.get("stop") or []
                stops = [stops] if isinstance(stops, str) else list(stops)
                if request.get("n", 1) != 1:
                    raise ValueError("n must be 1: replies are greedy, so they would all be the same")
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                return self._error(400, f"not a chat completion request: {error}")
            if request.get("stream"):
                self._stream(messages, max_tokens, stops, request.get("stream_options") or {})
                return
            try:
                done = chat.complete(messages, max_tokens, stops)
            except Exception as error:  # the template, say, refusing the conversation
                return self._error(500, f"no reply: {error}")
            self._json(200, self._completion(done))

        def _model(self) -> dict:
            return {"id": chat.name, "object": "model", "created": 0, "owned_by": "shem"}

        def _completion(self, done: dict) -> dict:
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": chat.name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": done["text"]},
                        "finish_reason": done["finish"],
                    }
                ],
                "usage": _usage(done),
            }

        def _stream(self, messages, max_tokens, stops, options) -> None:
            """Server-sent events: a chunk for each piece of the reply as it
            comes, then one that says why it ended, then [DONE]."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            base = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": chat.name,
            }

            def send(delta: dict, finish: str | None = None, usage: dict | None = None) -> None:
                event = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage is not None:
                    event["usage"] = usage
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()

            send({"role": "assistant", "content": ""})
            done = chat.complete(messages, max_tokens, stops, out=lambda text: send({"content": text}))
            usage = _usage(done) if options.get("include_usage") else None
            send({}, done["finish"], usage)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, text: str) -> None:
            kind = {400: "invalid_request_error", 404: "not_found_error"}.get(status, "server_error")
            self._json(status, {"error": {"message": text, "type": kind}})

    return Handler


def _usage(done: dict) -> dict:
    return {
        "prompt_tokens": done["prompt_tokens"],
        "completion_tokens": done["completion_tokens"],
        "total_tokens": done["prompt_tokens"] + done["completion_tokens"],
    }


def address(text: str, host: str) -> tuple[str, int]:
    """`[HOST:]PORT`, with `host` when it names none. Port 0 is any free one."""
    named, _, port = text.rpartition(":")
    if not port.isdigit() or not 0 <= int(port) < 65536:
        raise ValueError(f"--serve takes [HOST:]PORT, not {text!r}")
    return named.strip("[]") or host, int(port)


def listen(host: str, port: int, netns: str | None = None) -> socket.socket:
    """A socket listening at `host`:`port` in the network namespace at
    `netns`, a container's, or in this process's own. A thread of its own
    enters the namespace to make the socket, which stays in it, and ends."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    if netns is None:
        return socket.create_server((host, port), family=family)
    made: dict = {}

    def make() -> None:
        try:
            fd = os.open(netns, os.O_RDONLY)
            try:
                os.setns(fd, os.CLONE_NEWNET)
            finally:
                os.close(fd)
            made["socket"] = socket.create_server((host, port), family=family)
        except OSError as error:
            made["error"] = error

    thread = threading.Thread(target=make)
    thread.start()
    thread.join()
    if "error" in made:
        raise made["error"]
    return made["socket"]


def start(chat: Chat, listener: socket.socket) -> ThreadingHTTPServer:
    """Serve `chat` on `listener`, on threads of its own, until shut down."""
    host, port = listener.getsockname()[:2]
    server = ThreadingHTTPServer((host, port), handler(chat), bind_and_activate=False)
    server.socket.close()
    server.socket, server.server_address = listener, (host, port)
    server.server_name, server.server_port = host, port
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def serve_forever(model, tokenizer, args, primed: Session | None = None, err=sys.stderr) -> None:
    """Serve on the host, at `--serve [HOST:]PORT` (localhost by default),
    until interrupted."""
    chat = Chat(model, tokenizer, args, primed)
    host, port = address(args.serve, "127.0.0.1")
    server = start(chat, listen(host, port))
    port = server.server_address[1]
    shown = f"[{host}]" if ":" in host else host
    print(f"shem: serving {chat.name} at http://{shown}:{port}/v1", file=err, flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
