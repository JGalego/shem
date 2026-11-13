"""Web pages of `shem registry`: what it serves, for people rather than
OCI clients.

`/` says what the registry is, how to pull from it, and the repos it has
served since it started: it can't list the Hub's. `/ORG/NAME?tag=TAG` shows
an image: its tags, how to run and pull it, its entrypoint, the model from
its `shem.json`, the comparison with transformers the Space recorded, and
its layers. The pages are plain HTML, read-only like the rest.
"""

import html
import io
import json
import tarfile
from collections import Counter

import zstandard

#: Layers up to this size are read for the model's manifest or a session
READABLE = 64 << 20
LABEL = "dev.shem."
STATE_DIR = ".shem"

STYLE = """
:root { --bg: #fbfaf7; --fg: #23211d; --muted: #6b665c; --line: #e3dfd6; --accent: #8a5a2b; --code: #f1eee7; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #1b1a18; --fg: #ece8df; --muted: #a39d90; --line: #3a3732; --accent: #d9a46a; --code: #262421; }
}
body { background: var(--bg); color: var(--fg); font: 15px/1.55 system-ui, sans-serif; margin: 0; }
main { max-width: 860px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.5rem; margin: 0 0 4px; word-break: break-word; }
h2 { font-size: 1.05rem; margin: 28px 0 8px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
a { color: var(--accent); }
.muted { color: var(--muted); }
code, pre { background: var(--code); border-radius: 4px; font: 13px/1.5 ui-monospace, monospace; }
code { padding: 1px 4px; word-break: break-all; }
pre { padding: 10px 12px; overflow-x: auto; white-space: pre; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 500; width: 30%; }
td.n, th.n { text-align: right; white-space: nowrap; }
.tags a, .tags b { display: inline-block; margin: 0 6px 6px 0; padding: 2px 8px; border: 1px solid var(--line); border-radius: 12px; text-decoration: none; }
.tags b { border-color: var(--accent); }
.ok { color: #2f7d32; } .bad { color: #b3261e; }
@media (prefers-color-scheme: dark) { .ok { color: #81c784; } .bad { color: #ef9a9a; } }
input { font: inherit; padding: 5px 8px; border: 1px solid var(--line); border-radius: 4px; background: var(--bg); color: var(--fg); width: min(320px, 100%); }
button { font: inherit; padding: 5px 12px; }
"""


def e(text) -> str:
    return html.escape(str(text))


def page(title: str, body: str) -> bytes:
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{e(title)}</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>"
    ).encode()


def size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} GB"


def rows(pairs) -> str:
    return "<table>" + "".join(f"<tr><th>{e(k)}</th><td>{v}</td></tr>" for k, v in pairs if v) + "</table>"


def home(host: str, served: list[str]) -> bytes:
    listed = (
        "<ul>" + "".join(f'<li><a href="/{e(r)}">{e(r)}</a></li>' for r in served) + "</ul>"
        if served
        else '<p class="muted">None yet.</p>'
    )
    body = f"""
<h1>shem registry</h1>
<p class="muted">A read-only OCI registry over Hugging Face repos that hold an image, as <code>shem push IMAGE hf.co/ORG/NAME</code> writes one. A repo's revisions are its tags: <code>latest</code> is <code>main</code>. Large blobs redirect to Hugging Face, so weights never pass through here.</p>
<h2>Pull</h2>
<pre>oras pull {e(host)}/ORG/NAME:latest
docker run --rm --runtime io.containerd.shem.v1 --platform linux/riscv32 {e(host)}/ORG/NAME "What is a golem?"
shem run hf.co/ORG/NAME "What is a golem?"</pre>
<h2>Open a repo</h2>
<form action="/go"><input name="repo" placeholder="ORG/NAME" aria-label="Repo" required> <button>Open</button></form>
<h2>Served since this registry started</h2>
{listed}
"""
    return page("shem registry", body)


def members(blob: bytes, media_type: str):
    """The files of a layer, as (name, read) pairs, read only as far as asked."""
    if media_type.endswith("+zstd"):
        stream = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(blob))
        archive = tarfile.open(fileobj=stream, mode="r|")
    else:
        archive = tarfile.open(fileobj=io.BytesIO(blob), mode="r|*")
    with archive:
        for member in archive:
            yield member, archive


def model_manifest(hub, repo: str, manifest: dict, model_dir: str) -> dict | None:
    """The model's `shem.json`, from the first layer that has it."""
    wanted = f"{model_dir.strip('/')}/shem.json"
    for layer in manifest.get("layers", []):
        if layer["size"] > READABLE:
            continue
        blob = hub.blob(repo, layer["digest"].removeprefix("sha256:"))
        for member, archive in members(blob, layer["mediaType"]):
            if member.name.removeprefix("./") == wanted:
                return json.load(archive.extractfile(member))
    return None


def primed(hub, repo: str, manifest: dict) -> bool | None:
    """Whether the image's last layer is a saved session, if it is small
    enough to look: a layer of nothing but `/.shem`."""
    layers = manifest.get("layers") or []
    if not layers:
        return False
    last = layers[-1]
    if last["size"] > READABLE:
        return None
    names = [
        m.name.removeprefix("./")
        for m, _ in members(hub.blob(repo, last["digest"].removeprefix("sha256:")), last["mediaType"])
    ]
    return bool(names) and all(n == STATE_DIR or n.startswith(STATE_DIR + "/") for n in names)


def option(entrypoint: list[str], name: str) -> str | None:
    for i, arg in enumerate(entrypoint[:-1]):
        if arg == name:
            return entrypoint[i + 1]
    return None


def image_sections(hub, repo: str, manifest: dict, config: dict) -> str:
    process = config.get("config") or {}
    entrypoint = process.get("Entrypoint") or []
    labels = process.get("Labels") or {}
    model_dir = labels.get(f"{LABEL}model")
    out = ""
    session = primed(hub, repo, manifest) if model_dir else None
    out += "<h2>Image</h2>" + rows(
        [
            ("Entrypoint", f"<code>{e(json.dumps(entrypoint))}</code>" if entrypoint else ""),
            ("Cmd", f"<code>{e(json.dumps(process.get('Cmd')))}</code>" if process.get("Cmd") else ""),
            (
                "System prompt",
                f"<pre>{e(option(entrypoint, '--system'))}</pre>" if option(entrypoint, "--system") else "",
            ),
            (
                "Primed session",
                {
                    True: "Yes: runs start after the system prompt",
                    False: "No",
                    None: "Unknown: its last layer is large",
                }[session]
                if model_dir
                else "",
            ),
            ("Machine", e(labels.get(f"{LABEL}machine", ""))),
        ]
    )
    if model_dir:
        model = model_manifest(hub, repo, manifest, model_dir)
        if model:
            source = model.get("source") or {}
            kinds = Counter(block["type"] for block in model.get("blocks", []))
            link = ""
            if source.get("repo"):
                rev = source.get("revision", "")
                link = f'<a href="https://huggingface.co/{e(source["repo"])}">{e(source["repo"])}</a>'
                link += f' <span class="muted">at <code>{e(rev[:12])}</code></span>' if rev else ""
            out += "<h2>Model</h2>" + rows(
                [
                    ("Source", link),
                    ("Architecture", e(source.get("architecture", ""))),
                    ("Parameters", f"{model['parameters'] / 1e9:.2f}B" if model.get("parameters") else ""),
                    ("Precision", e(", ".join(model.get("precision", [])))),
                    ("Blocks", e(", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items())))),
                    (
                        "Licence",
                        e(
                            source.get("spdx")
                            or source.get("license")
                            or labels.get("org.opencontainers.image.licenses", "")
                        ),
                    ),
                    ("Format", f"<code>{e(model.get('format', ''))}</code>"),
                ]
            )
    return out


def artifact_sections(config: dict, manifest: dict) -> str:
    descriptor, settings = config.get("descriptor") or {}, config.get("config") or {}
    out = "<h2>ModelPack artifact</h2>" + rows(
        [
            ("Name", e(descriptor.get("name", ""))),
            ("Family", e(descriptor.get("family", ""))),
            ("Architecture", e(settings.get("architecture", ""))),
            ("Format", e(settings.get("format", ""))),
            ("Parameters", e(settings.get("paramSize", ""))),
            ("Precision", e(settings.get("precision", ""))),
        ]
    )
    return out


def comparison_section(comparison: dict | None) -> str:
    if not comparison:
        return ""
    passed = comparison.get("passed")
    verdict = '<span class="ok">✓ Verified</span>' if passed else '<span class="bad">✗ Differs</span>'
    return f"<h2>Against transformers: {verdict}</h2>" + rows(
        [
            ("Largest logit difference", f"{comparison['difference']:.2e} <span class='muted'>(tolerance 1e-3)</span>"),
            ("Greedy reply", f"{comparison['length']} tokens, {'the same' if comparison['tokens'] else 'different'}"),
        ]
    )


def layer_rows(manifest: dict) -> str:
    layers = manifest.get("layers") or []
    total = sum(layer["size"] for layer in layers)
    body = ""
    for i, layer in enumerate(layers):
        annotations = layer.get("annotations") or {}
        name = next((v for k, v in annotations.items() if k.endswith(".filepath") or k.endswith(".title")), "")
        body += (
            f"<tr><td class='n'>{i}</td><td><code>{e(layer['mediaType'].rsplit('/', 1)[-1])}</code>"
            f"{' ' + e(name) if name else ''}</td><td class='n'>{size(layer['size'])}</td></tr>"
        )
    head = "<tr><th class='n'>#</th><th>Type</th><th class='n'>Size</th></tr>"
    return f"<h2>Layers <span class='muted'>({len(layers)}, {size(total)})</span></h2><table>{head}{body}</table>"


def repo_page(hub, host: str, repo: str, tag: str) -> bytes:
    tags = hub.tags(repo)
    entry = (hub.index(repo, tag).get("manifests") or [None])[0]
    if entry is None:
        return page(repo, f"<h1>{e(repo)}</h1><p>{e(repo)}:{e(tag)} lists no manifest.</p>")
    data = json.loads(hub.blob(repo, entry["digest"].removeprefix("sha256:")))
    # An index lists the image under platforms: they are all the same image
    while "manifests" in data:
        data = json.loads(hub.blob(repo, data["manifests"][0]["digest"].removeprefix("sha256:")))
    manifest = data
    config = json.loads(hub.blob(repo, manifest["config"]["digest"].removeprefix("sha256:")))
    kind = manifest["config"].get("mediaType", "")
    artifact = ".model.config." in kind
    reference = repo if tag == "latest" else f"{repo}:{tag}"
    chips = "".join(f"<b>{e(t)}</b>" if t == tag else f'<a href="/{e(repo)}?tag={e(t)}">{e(t)}</a>' for t in tags)
    run = "" if artifact else f'shem run hf.co/{e(reference)} "What is a golem?"\n'
    docker = (
        ""
        if artifact
        else f'docker run --rm --runtime io.containerd.shem.v1 --platform linux/riscv32 {e(host)}/{e(repo)}:{e(tag)} "What is a golem?"\n'
    )
    body = f"""
<h1>{e(repo)}<span class="muted">:{e(tag)}</span></h1>
<p class="muted">{"A ModelPack artifact" if artifact else "A shem image"}, manifest <code>{e(entry["digest"])}</code> · <a href="https://huggingface.co/{e(repo)}">on Hugging Face</a></p>
<div class="tags">{chips}</div>
<h2>Run and pull</h2>
<pre>{run}oras pull {e(host)}/{e(repo)}:{e(tag)}
{docker}</pre>
"""
    body += artifact_sections(config, manifest) if artifact else image_sections(hub, repo, manifest, config)
    body += comparison_section(hub.comparison(repo, tag))
    if "modelpack" in tags and tag != "modelpack":
        body += f'<p>The <a href="/{e(repo)}?tag=modelpack">modelpack</a> tag holds the model as a ModelPack artifact over the same blobs.</p>'
    body += layer_rows(manifest)
    labels = (config.get("config") or {}).get("Labels") or {}
    others = [(k, e(v)) for k, v in sorted(labels.items()) if not k.startswith(LABEL)]
    if others:
        body += "<h2>Labels</h2>" + rows(others)
    return page(f"{repo}:{tag}", body)
