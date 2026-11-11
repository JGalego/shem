"""Tools: programs a model calls, run exactly on its compiled CPU layer.

A model image can carry compiled layers: blocks whose weights are compiled
from a specification rather than trained. `COPY --from=shem:rv32im /
/model/compiled/rv32im/` adds the RV32IM CPU of the neural vCPU, and
`shem:generate --tools DIR` offers the image's programs in DIR to the model.
When the model calls one, the host routes the call to the compiled layer: it
runs the program on a neural vCPU whose weights are that layer's, with the
host servicing its system calls in the image's root filesystem (`shem run
--vcpu neural`), and hands the output back to the model as the tool's reply.
Language goes through the model's trained blocks; exact work goes through the
compiled ones, and the host is the router between them.

A program's interface is `NAME.json` next to it, as a function's JSON schema
(`description` and `parameters`), whose parameters become the program's
arguments in order; or, by default, an `args` list and `stdin`.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

COMPILED = "compiled"
FORMAT = "shem.cpu/v1"
DEFAULT_PARAMETERS = {
    "type": "object",
    "properties": {
        "args": {"type": "array", "items": {"type": "string"}, "description": "the program's arguments"},
        "stdin": {"type": "string", "description": "text for the program's standard input"},
    },
}


def compiled_layers(model_dir: Path) -> dict[str, Path]:
    """The model's compiled layers, by name: directories under `compiled/`
    whose `config.json` describes a compiled CPU."""
    found = {}
    for config in sorted((Path(model_dir) / COMPILED).glob("*/config.json")):
        meta = json.loads(config.read_text())
        if meta.get("format") == FORMAT:
            found[config.parent.name] = config.parent
    return found


def tool_calls(text: str) -> list[tuple[str, dict]]:
    """The tool calls in a reply, in either format a chat template asks for:
    JSON (`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`) or tags
    (`<tool_call><function=NAME><parameter=P>VALUE</parameter>...`)."""
    calls = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S):
        block = block.strip()
        if block.startswith("{"):
            try:
                call = json.loads(block)
            except json.JSONDecodeError:
                continue
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"args": [arguments]}
            if isinstance(call.get("name"), str) and isinstance(arguments, dict):
                calls.append((call["name"], arguments))
        elif match := re.match(r"<function=([^>\s]+)>(.*?)</function>", block, re.S):
            parameters = re.findall(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", match.group(2), re.S)
            calls.append((match.group(1), dict(parameters)))
    return calls


class Tools:
    """The programs in `directory` of the image whose root is `root`, run on
    the compiled CPU layer at `weights`."""

    def __init__(self, root: Path, directory: str, weights: Path, socket: Path | None = None):
        self.root = Path(root)
        self.dir = self.root / directory.lstrip("/")
        self.weights = Path(weights)
        self.socket = socket
        self.specs: dict[str, dict] = {}
        for path in sorted(self.dir.iterdir()):
            if path.suffix == ".json" or not path.is_file() or not os.access(path, os.X_OK):
                continue
            spec_file = path.with_name(path.name + ".json")
            spec = json.loads(spec_file.read_text()) if spec_file.exists() else {}
            self.specs[path.name] = {
                "name": path.name,
                "description": spec.get("description", f"Run the program {path.name}"),
                "parameters": spec.get("parameters", DEFAULT_PARAMETERS),
            }
        self._server = None

    def schemas(self) -> list[dict]:
        """The tools as a chat template wants them."""
        return [{"type": "function", "function": spec} for spec in self.specs.values()]

    def run(self, name: str, arguments: dict, err=sys.stderr) -> str:
        """Run the program `name` with `arguments` on the compiled layer, and
        return what it printed, or why it could not run."""
        spec = self.specs.get(name)
        if spec is None:
            return f"error: there is no tool {name!r}; there are {', '.join(self.specs) or 'none'}"
        names = list(spec["parameters"].get("properties", {}))
        if "args" in arguments or names == ["args", "stdin"]:
            args = [str(a) for a in arguments.get("args", [])]
        else:
            args = [str(arguments[n]) for n in names if n in arguments and n != "stdin"]
        command = [
            os.environ.get("SHEM_BIN") or shutil.which("shem") or "shem",
            "run",
            "--vcpu",
            "neural",
            "--neural-socket",
            str(self.vcpu_socket()),
            "--neural-weights",
            str(self.weights),
            "--root",
            str(self.root),
            "--mem",
            "4M",
            "--stats",
            str(self.dir / name),
            *args,
        ]
        try:
            done = subprocess.run(
                command, input=str(arguments.get("stdin", "")), capture_output=True, text=True, timeout=900
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return f"error: {error}"
        retired = re.search(r"(\d+) instructions retired", done.stderr)
        steps = f"{int(retired.group(1)):,} instructions" if retired else "no instructions"
        print(f"shem: ran {name} {' '.join(args)} on the compiled RV32IM layer, {steps}", file=err, flush=True)
        output = done.stdout.strip()
        if done.returncode != 0:
            problems = "\n".join(line for line in done.stderr.splitlines() if "instructions retired" not in line)
            output += f"\n(exit status {done.returncode}){chr(10) + problems if problems else ''}"
        return output

    def vcpu_socket(self) -> Path:
        """A neural vCPU server to run on: the one given, or one started in
        this process the first time a tool runs."""
        if self.socket is not None:
            return self.socket
        if self._server is None:
            from ..server import make_server

            socket = Path(tempfile.mkdtemp()) / "vcpu.sock"
            self._server, _ = make_server(socket, single_thread=False)
            threading.Thread(target=self._server.serve_forever, daemon=True).start()
            self.socket = socket
        return self.socket
