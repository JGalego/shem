"""Serve a model over HTTP as OpenAI's chat completions API does, for the
clients and tools written for that API:

    POST /v1/chat/completions   a reply to the messages, streamed with "stream": true
    GET  /v1/models             the model, by the repository it came from
    GET  /health                200 once the model is loaded
    GET  /metrics               Prometheus metrics: requests, latency, tokens, waiting, batch size

A request with `"shem_receipt": true` gets a receipt of its reply, in the
response (and a stream's last chunk), which `shem replay` checks.

A request is refused unless it carries `Authorization: Bearer KEY` when the
environment has `SHEM_API_KEY` (`/health` stays open, for probes); a page
elsewhere may call the API only from an origin `--cors-origin` allows; a
body is at most `MAX_BODY` bytes; and at most `MAX_WAITING` requests wait
their turn on the model, the rest being told to try again.

Replies are greedy, as the driver's are, unless the image's options or the
request give a `temperature`: then, with `top_p`, `top_k` and `seed`, each
token is drawn (see `sampling`), and the same seed draws the same reply.
Requests take turns on the model. The image's system prompt comes first
unless a request gives its own, and a conversation that starts with the
image's primed session, token for token, goes on from it rather than
reading those tokens again.

`shem serve IMAGE` serves on the host. In a container, `--serve PORT`
serves in the container's network, where `docker run -p` publishes it.
"""

import base64
import contextlib
import copy
import hmac
import json
import os
import queue
import socket
import struct
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from .. import metrics as telemetry
from . import receipt as receipts
from .generate import ContextFull, Reply, Session
from .sampling import Sampler
from .tools import tool_calls

#: Replies made in one pass over the model's weights, the rest waiting for room
MAX_BATCH = 16
#: The paths the API answers, which its metrics tell apart from the rest
KNOWN_PATHS = {
    "/health",
    "/metrics",
    "/v1/models",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/chat/completions",
    "/completions",
    "/embeddings",
}
#: The largest request body taken, in bytes: a long conversation is a few hundred kilobytes
MAX_BODY = 8 << 20
#: Requests that may be waiting for or having their turn on the model; the next is told to retry
MAX_WAITING = 32
#: Seconds a client may stay silent in the middle of a request, or between requests on a connection
CLIENT_TIMEOUT = 60


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
        #: How tokens are drawn unless a request says otherwise: the image's options
        self.draw = {
            key: getattr(args, key)
            for key in ("temperature", "top_p", "top_k", "seed")
            if getattr(args, key, None) is not None
        }
        #: What the host knows of this model that a receipt names it by: the image's reference and digest
        self.subjects = dict(subject.split("=", 1) for subject in getattr(args, "subject", None) or [])
        #: The key requests must carry, from the environment, never the command line, where `ps` shows it
        self.api_key = os.environ.get("SHEM_API_KEY") or None
        #: Origins whose pages may call the API, or `*` for any
        self.cors = list(getattr(args, "cors_origin", None) or [])
        self.waiting = threading.BoundedSemaphore(MAX_WAITING)
        self.metrics = telemetry.Metrics()
        for name, kind, text in (
            ("shem_requests_total", "counter", "Requests answered, by path and status"),
            ("shem_request_seconds", "histogram", "Seconds from a request to its answer"),
            ("shem_tokens_total", "counter", "Tokens read (prompt) and made (completion)"),
            ("shem_requests_waiting", "gauge", "Requests waiting for or having their turn on the model"),
            ("shem_batch_size", "histogram", "Replies read in one pass over the model's weights"),
            ("shem_model_info", "gauge", "The model served"),
        ):
            self.metrics.describe(name, kind, text)
        self.metrics.set("shem_model_info", 1, model=self.name, precision=model.precision)
        #: Prefill, which reads a prompt in one pass, takes turns; replies are made together
        self.prefill = threading.Lock()
        self.batcher = Batcher(compute)
        self.batcher.metrics = self.metrics
        self.primed = primed if primed is not None and primed.tokens else None

    def prompt(self, messages: list[dict], tools: list[dict] | None = None) -> list[int]:
        """The tokens a reply to `messages` starts from, the image's system
        prompt first unless the request gives its own, and `tools` offered if
        there are any. Raises `ContextFull` if there is no room in the
        model's context window to reply."""
        if self.system and not (messages and messages[0]["role"] == "system"):
            messages = [{"role": "system", "content": self.system}] + messages
        return self.check(self.tokenizer.chat(messages, tools or None))

    def check(self, tokens: list[int]) -> list[int]:
        """`tokens`, if the model's window has room for a reply to them."""
        context = self.model.manifest.context
        if context is not None and len(tokens) >= context:
            raise ContextFull(f"the conversation is {len(tokens)} tokens, and the model's context window is {context}")
        return tokens

    def complete(
        self,
        messages: list[dict] | None,
        max_tokens: int | None = None,
        stops=(),
        out=None,
        draw: dict | None = None,
        receipt: bool = False,
        tools: list[dict] | None = None,
        logprobs: int | None = None,
        prompt_tokens: list[int] | None = None,
    ) -> dict:
        """Reply to `messages`, or continue `prompt_tokens`, writing the text
        to `out` as it comes, drawing tokens as `draw` says over the image's
        own settings. Returns the reply, why it ended, how many tokens each
        side took, the tool calls it makes if it was offered `tools`, the
        log probabilities of its tokens if `logprobs` (how many of the likeliest
        to give besides the chosen one), and with `receipt`, a receipt of the
        reply, for `shem replay`."""
        raw = prompt_tokens is not None
        prompt = self.check(prompt_tokens) if raw else self.prompt(messages, tools)
        sampler = None
        if settings := {**self.draw, **(draw or {})}:
            if settings.get("temperature"):
                sampler = Sampler(**settings)
        max_tokens = max_tokens or self.max_tokens
        if not raw and self.system and not (messages and messages[0]["role"] == "system"):
            messages = [{"role": "system", "content": self.system}] + messages
        job = Job()
        text = _Text(job.pieces.put)
        with self.prefill, self.compute():
            session = self._start(prompt)
            session.sampler = sampler
            if receipt:
                session.choices = []
            if logprobs is not None:
                session.logprobs, session.top_logprobs = [], logprobs
            session.feed(prompt[len(session.tokens) :])
        job.made = Reply(session, self.tokenizer, max_tokens, text, hold=bool(tools), stops=[*self.stops, *stops])
        self.batcher.submit(job)
        try:
            # What the reply writes is passed on from this request's own thread: a slow client holds up no one else
            while (piece := job.pieces.get()) is not None:
                if out is not None:
                    out(piece)
        except BaseException:
            job.cancelled = True
            raise
        if job.error is not None:
            raise job.error
        tokens, room = job.made.tokens, session.room()
        made = {}
        if receipt:
            request = {"max_tokens": max_tokens, "stop": [*self.stops, *stops]}
            made = {"receipt": receipts.record_messages(session, messages, text.value.strip(), request, self.subjects)}
        if logprobs is not None:
            made["logprobs"] = session.logprobs[: len(tokens)]
        calls = tool_calls(job.made.text) if tools else []
        # A text continuation is the model's own text, but for the line end the reply closes with
        value = text.value.removesuffix("\n") if raw else text.value.strip()
        self.metrics.inc("shem_tokens_total", len(prompt), kind="prompt")
        self.metrics.inc("shem_tokens_total", len(tokens), kind="completion")
        return {
            **made,
            "text": value,
            "calls": calls,
            "finish": "tool_calls"
            if calls
            else "length"
            if len(tokens) >= max_tokens or (room is not None and room < 1)
            else "stop",
            "prompt_tokens": len(prompt),
            "completion_tokens": len(tokens),
        }

    def embed(self, text: str, pooling: str = "last") -> tuple[list[float], int]:
        """An embedding of `text`: what the model has made of it, taken from its
        last token or averaged over them all, and scaled to length 1. The
        model is a language model's, so the vector is as good as that model is
        at representing a text in one place, which one trained for the purpose is. Returns it,
        and how many tokens the text was."""
        tokens = self.check(self.tokenizer.encode(text))
        if not tokens:
            raise ValueError("there is no text to embed")
        with self.prefill, self.compute():
            hidden = self.model.hidden(torch.tensor([tokens]), self.model.init_state(), 0)[0].float()
        vector = hidden.mean(dim=0) if pooling == "mean" else hidden[-1]
        return (vector / vector.norm().clamp(min=1e-12)).tolist(), len(tokens)

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


class Job:
    """A reply on its way, for the batcher to step: what it writes goes to
    `pieces`, which the request that made it reads, and ends with None."""

    def __init__(self):
        self.made: Reply | None = None
        self.pieces: queue.Queue = queue.Queue()
        self.error: BaseException | None = None
        self.cancelled = False

    def end(self, error: BaseException | None = None) -> None:
        self.error = error
        self.pieces.put(None)


class Batcher:
    """Makes the replies of every request being answered together, a token of
    each at a time in one pass over the model's weights: decoding is bound by
    reading them, so the more replies share a pass the more tokens a second
    the model makes. A thread of its own runs it, in the compute context
    (the model server's gives it every core), and starts with the first
    reply."""

    def __init__(self, compute=contextlib.nullcontext, size: int | None = None):
        self.compute, self.size = compute, size or MAX_BATCH
        self.waiting: queue.Queue = queue.Queue()
        #: Where the size of each pass is observed, when someone wants it
        self.metrics: telemetry.Metrics | None = None
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def submit(self, job: Job) -> None:
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
        self.waiting.put(job)

    def _run(self) -> None:
        active: list[Job] = []
        with self.compute():
            while True:
                if not active:
                    active.append(self.waiting.get())
                while len(active) < self.size:
                    try:
                        active.append(self.waiting.get_nowait())
                    except queue.Empty:
                        break
                active = self._step(active, self.metrics)

    @staticmethod
    def _step(active: list[Job], metrics: "telemetry.Metrics | None" = None) -> list[Job]:
        """Choose each reply's next token, then read them all in one pass.
        Replies that end, fail or are given up on leave."""
        going, tokens = [], []
        for job in active:
            if job.cancelled:
                continue
            try:
                token = job.made.next()
            except Exception as error:
                job.end(error)
                continue
            if token is None:
                job.end()
            else:
                going.append(job)
                tokens.append(token)
        if going:
            if metrics is not None:
                metrics.observe("shem_batch_size", len(going), buckets=(1, 2, 4, 8, 16, 32))
            try:
                Session.feed_batch([job.made.session for job in going], tokens)
            except Exception as error:  # a pass that fails may have read for some and not others: all end
                for job in going:
                    job.end(error)
                return []
        return going


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
    parts, if it has them, are joined, and an assistant's tool calls, whose
    arguments, JSON in OpenAI's API, are the objects templates take. Only
    text is understood."""
    role, content = raw["role"], raw.get("content") or ""
    if isinstance(content, list):
        if any(part.get("type") != "text" for part in content):
            raise ValueError("only text content is understood")
        content = "".join(part["text"] for part in content)
    if role not in ("system", "user", "assistant", "developer", "tool") or not isinstance(content, str):
        raise ValueError(f"a message must be a system, user, assistant or tool's text, not {role!r}")
    made = {"role": "system" if role == "developer" else role, "content": content}
    if role == "assistant" and raw.get("tool_calls"):
        calls = []
        for call in raw["tool_calls"]:
            function = call["function"]
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("a tool call's arguments must be a JSON object")
            calls.append({"type": "function", "function": {"name": function["name"], "arguments": arguments}})
        made["tool_calls"] = calls
    return made


def tools_of(request: dict) -> list[dict]:
    """The tools a request offers, if it does and does not forbid them: as a
    chat template takes them, a list of `{"type": "function", "function": ...}`."""
    tools = request.get("tools") or []
    choice = request.get("tool_choice", "auto")
    if choice not in ("auto", "none", None):
        raise ValueError("tool_choice can be auto or none: shem does not force a call")
    if choice == "none":
        return []
    for tool in tools:
        if tool.get("type") != "function" or not isinstance(tool.get("function", {}).get("name"), str):
            raise ValueError("tools must be functions, each with a name")
    return tools


def handler(chat: Chat):
    """The HTTP request handler for `chat`."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "shem"
        timeout = CLIENT_TIMEOUT

        def log_message(self, format, *args):
            pass

        def log_request(self, code="-", size="-"):
            """Every answer is counted, by the path it was for and its status."""
            path = self.path.split("?")[0]
            known = path in KNOWN_PATHS
            chat.metrics.inc("shem_requests_total", path=path if known else "other", status=code)

        def handle_one_request(self):
            started = time.perf_counter()
            super().handle_one_request()
            if getattr(self, "path", None):
                path = self.path.split("?")[0]
                chat.metrics.observe(
                    "shem_request_seconds", time.perf_counter() - started, path=path if path in KNOWN_PATHS else "other"
                )

        def _allowed(self) -> bool:
            """Whether the request carries the key, if there is one."""
            if not chat.api_key:
                return True
            given = self.headers.get("Authorization", "")
            scheme, _, token = given.partition(" ")
            return scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode(), chat.api_key.encode())

        def _origin(self) -> str | None:
            """What `Access-Control-Allow-Origin` says to this request's page, if it may call."""
            origin = self.headers.get("Origin")
            if "*" in chat.cors:
                return "*"
            return origin if origin and origin in chat.cors else None

        def do_OPTIONS(self):
            """The preflight a page's browser sends before it calls the API."""
            allow = self._origin()
            self.send_response(204 if allow else 403)
            if allow:
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def end_headers(self):
            if allow := self._origin():
                self.send_header("Access-Control-Allow-Origin", allow)
                if allow != "*":
                    self.send_header("Vary", "Origin")
            super().end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path != "/health" and not self._allowed():
                return self._unauthorized()
            if path == "/health":
                self._json(200, {"status": "ok"})
            elif path == "/metrics":
                waiting = MAX_WAITING - chat.waiting._value  # the semaphore's own count
                chat.metrics.set("shem_requests_waiting", waiting)
                data = chat.metrics.render().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [self._model()]})
            elif path.startswith(("/v1/models/", "/models/")):
                self._json(200, self._model())
            else:
                self._error(404, f"nothing at {path}: the API is at /v1/chat/completions")

        def _unauthorized(self) -> None:
            self.close_connection = True
            self._error(401, "send the API key as 'Authorization: Bearer KEY'", "invalid_api_key")

        def do_POST(self):
            path = self.path.split("?")[0]
            if not self._allowed():
                return self._unauthorized()
            kind = {
                "/v1/chat/completions": "chat",
                "/chat/completions": "chat",
                "/v1/completions": "text",
                "/completions": "text",
                "/v1/embeddings": "embeddings",
                "/embeddings": "embeddings",
            }.get(path)
            if kind is None:
                self.close_connection = True  # a body it did not read would be taken for the next request
                return self._error(
                    404, f"nothing at {path}: the API is at /v1/chat/completions, /v1/completions and /v1/embeddings"
                )
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0:
                return self._error(400, "not a request: Content-Length is not a size")
            if length > MAX_BODY:
                self.close_connection = True
                return self._error(413, f"the request is over the {MAX_BODY} bytes the API takes", "request_too_large")
            if kind == "embeddings":
                return self._embeddings(length)
            try:
                request = json.loads(self.rfile.read(length))
                ask = self._ask(request, kind)
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                what = "chat completion" if kind == "chat" else "completion"
                return self._error(400, f"not a {what} request: {error}")
            try:
                if kind == "chat":
                    chat.prompt(ask["messages"], ask["tools"])
                else:
                    chat.check(ask["prompt_tokens"])
            except ContextFull as error:
                return self._error(400, str(error), "context_length_exceeded")
            if not chat.waiting.acquire(blocking=False):
                return self._error(429, f"{MAX_WAITING} requests are waiting for the model: try again", "busy", retry=5)
            try:
                if request.get("stream"):
                    self._stream(kind, ask, request.get("stream_options") or {})
                    return
                try:
                    done = chat.complete(**ask["complete"])
                except Exception as error:  # the template, say, refusing the conversation
                    return self._error(500, f"no reply: {error}")
                self._json(200, self._completion(kind, done, ask["logprobs"]))
            finally:
                chat.waiting.release()

        def _embeddings(self, length: int) -> None:
            """`/v1/embeddings`: a vector for each text, as OpenAI's API gives them."""
            try:
                request = json.loads(self.rfile.read(length))
                texts = request["input"]
                texts = [texts] if isinstance(texts, str) else texts
                if not texts or not all(isinstance(text, str) for text in texts):
                    raise ValueError("input must be a text, or a list of texts: shem does not take token lists")
                pooling = request.get("shem_pooling", "last")
                if pooling not in ("last", "mean"):
                    raise ValueError("shem_pooling must be last or mean")
                form = request.get("encoding_format", "float")
                if form not in ("float", "base64"):
                    raise ValueError("encoding_format must be float or base64")
                if request.get("dimensions") is not None:
                    raise ValueError("dimensions: shem gives the model's own size")
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                return self._error(400, f"not an embeddings request: {error}")
            if not chat.waiting.acquire(blocking=False):
                return self._error(429, f"{MAX_WAITING} requests are waiting for the model: try again", "busy", retry=5)
            try:
                data, used = [], 0
                for index, text in enumerate(texts):
                    vector, count = chat.embed(text, pooling)
                    used += count
                    if form == "base64":
                        vector = base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()
                    data.append({"object": "embedding", "index": index, "embedding": vector})
            except ContextFull as error:
                return self._error(400, str(error), "context_length_exceeded")
            except ValueError as error:
                return self._error(400, f"not an embeddings request: {error}")
            except Exception as error:
                return self._error(500, f"no embedding: {error}")
            finally:
                chat.waiting.release()
            self._json(
                200,
                {
                    "object": "list",
                    "data": data,
                    "model": chat.name,
                    "usage": {"prompt_tokens": used, "total_tokens": used},
                },
            )

        def _ask(self, request: dict, kind: str) -> dict:
            """What a request asks for, checked, as the arguments of
            `Chat.complete` and the things the answer needs."""
            if kind == "chat":
                messages = [message(m) for m in request["messages"]]
                if not messages:
                    raise ValueError("there are no messages to reply to")
                tools = tools_of(request)
                given = {"messages": messages, "tools": tools or None}
            else:
                prompt = request["prompt"]
                if isinstance(prompt, list) and len(prompt) == 1:
                    prompt = prompt[0]
                if not isinstance(prompt, str):
                    raise ValueError("prompt must be a text: shem continues one text at a time")
                tools = []
                given = {"messages": None, "prompt_tokens": chat.tokenizer.encode(prompt)}
            max_tokens = request.get("max_completion_tokens") or request.get("max_tokens")
            if max_tokens is not None and (
                isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
            ):
                raise ValueError("max_tokens must be a positive number")
            stops = request.get("stop") or []
            stops = [stops] if isinstance(stops, str) else list(stops)
            if request.get("n", 1) != 1:
                raise ValueError("n must be 1: a reply is one draw from one seed")
            receipt = request.get("shem_receipt") is True
            if receipt and (kind != "chat" or tools):
                raise ValueError(
                    "shem_receipt is for chat completions without tools: they are what `shem replay` reads again"
                )
            draw = {}
            for key, number in (("temperature", float), ("top_p", float), ("top_k", int), ("seed", int)):
                if request.get(key) is not None:
                    if isinstance(request[key], bool) or not isinstance(request[key], int | float):
                        raise ValueError(f"{key} must be a number")
                    draw[key] = number(request[key])
            if draw.get("temperature"):
                Sampler(**draw)  # its settings, checked before the headers of a stream go out
            logprobs = None
            if kind == "chat" and request.get("logprobs"):
                logprobs = request.get("top_logprobs") or 0
            elif kind == "text" and request.get("logprobs") is not None:
                logprobs = request["logprobs"]
            if logprobs is not None:
                if isinstance(logprobs, bool) or not isinstance(logprobs, int) or not 0 <= logprobs <= 20:
                    raise ValueError("top_logprobs, or logprobs of a completion, must be a number from 0 to 20")
                if request.get("stream"):
                    raise ValueError("logprobs are not sent in a stream yet: ask without stream")
            if request.get("response_format") not in (None, {"type": "text"}):
                raise ValueError("response_format: shem does not constrain replies to JSON, only text")
            complete = {
                **given,
                "max_tokens": max_tokens,
                "stops": stops,
                "draw": draw,
                "receipt": receipt,
                "logprobs": logprobs,
            }
            return {**given, "tools": tools or None, "complete": complete, "logprobs": logprobs}

        def _model(self) -> dict:
            return {"id": chat.name, "object": "model", "created": 0, "owned_by": "shem"}

        def _token(self, token: int) -> str:
            return chat.tokenizer.decode([token])

        def _logprobs(self, kind: str, entries: list, text: str) -> dict:
            """The log probabilities of a reply's tokens in the shape of the
            endpoint's: chosen token first, then the likeliest others."""
            if kind == "chat":
                content = []
                for token, logprob, top in entries:
                    word = self._token(token)
                    content.append(
                        {
                            "token": word,
                            "logprob": logprob,
                            "bytes": list(word.encode()),
                            "top_logprobs": [
                                {"token": w, "logprob": lp, "bytes": list(w.encode())}
                                for w, lp in ((self._token(t), lp) for t, lp in top)
                            ],
                        }
                    )
                return {"content": content, "refusal": None}
            words = [self._token(token) for token, _, _ in entries]
            offsets, at = [], 0
            for word in words:
                offsets.append(at)
                at += len(word)
            return {
                "tokens": words,
                "token_logprobs": [logprob for _, logprob, _ in entries],
                "top_logprobs": [{self._token(t): lp for t, lp in top} for _, _, top in entries],
                "text_offset": offsets,
            }

        @staticmethod
        def _tool_calls(calls: list) -> list[dict]:
            return [
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
                for name, arguments in calls
            ]

        def _completion(self, kind: str, done: dict, logprobs) -> dict:
            made = {
                "created": int(time.time()),
                "model": chat.name,
                "usage": _usage(done),
                **({"shem_receipt": done["receipt"]} if "receipt" in done else {}),
            }
            shown = self._logprobs(kind, done["logprobs"], done["text"]) if logprobs is not None else None
            if kind == "text":
                return {
                    "id": f"cmpl-{uuid.uuid4().hex}",
                    "object": "text_completion",
                    **made,
                    "choices": [{"index": 0, "text": done["text"], "logprobs": shown, "finish_reason": done["finish"]}],
                }
            reply = {"role": "assistant", "content": done["text"] or (None if done["calls"] else "")}
            if done["calls"]:
                reply["tool_calls"] = self._tool_calls(done["calls"])
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                **made,
                "choices": [{"index": 0, "message": reply, "logprobs": shown, "finish_reason": done["finish"]}],
            }

        def _stream(self, kind: str, ask: dict, options: dict) -> None:
            """Server-sent events: a chunk for each piece of the reply as it
            comes, then one that says why it ended, then [DONE]."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            base = {
                "id": f"{'chatcmpl' if kind == 'chat' else 'cmpl'}-{uuid.uuid4().hex}",
                "object": "chat.completion.chunk" if kind == "chat" else "text_completion",
                "created": int(time.time()),
                "model": chat.name,
            }

            def send(
                delta: dict | str,
                finish: str | None = None,
                usage: dict | None = None,
                receipt: dict | None = None,
            ) -> None:
                if kind == "chat":
                    choice = {"index": 0, "delta": delta, "finish_reason": finish}
                else:
                    choice = {"index": 0, "text": delta, "logprobs": None, "finish_reason": finish}
                event = {**base, "choices": [choice]}
                if usage is not None:
                    event["usage"] = usage
                if receipt is not None:
                    event["shem_receipt"] = receipt
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()

            if kind == "chat":
                send({"role": "assistant", "content": ""})
            try:
                done = chat.complete(
                    **ask["complete"], out=lambda text: send({"content": text} if kind == "chat" else text)
                )
            except (BrokenPipeError, ConnectionError):
                return  # the client left
            except Exception as error:  # after the headers, an error is an event, as OpenAI sends one
                failure = {"error": {"message": f"no reply: {error}", "type": "server_error", "code": None}}
                self.wfile.write(f"data: {json.dumps(failure)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            if done["calls"]:
                calls = [{"index": i, **call} for i, call in enumerate(self._tool_calls(done["calls"]))]
                send({"tool_calls": calls})
            usage = _usage(done) if options.get("include_usage") else None
            send({} if kind == "chat" else "", done["finish"], usage, done.get("receipt"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _json(self, status: int, body: dict, retry: int | None = None) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if retry is not None:
                self.send_header("Retry-After", str(retry))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, text: str, code: str | None = None, retry: int | None = None) -> None:
            kind = {
                400: "invalid_request_error",
                401: "authentication_error",
                404: "not_found_error",
                413: "invalid_request_error",
                429: "rate_limit_error",
            }.get(status, "server_error")
            self._json(status, {"error": {"message": text, "type": kind, "code": code}}, retry)

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
