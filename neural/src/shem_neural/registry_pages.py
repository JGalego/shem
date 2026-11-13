"""Web pages of `shem registry`: what it serves, for people rather than
OCI clients.

`/` introduces the registry and shows the repos it has served since it
started as cards: it can't list the Hub's. `/ORG/NAME?tag=TAG` shows an
image as a model page: badges for what it is, the system prompt, the model
from its `shem.json`, the comparison with transformers the Space recorded,
and its layers, beside a box with the commands to use it. The pages work
without JavaScript, which only adds copy buttons.
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
PROMPT = "What is a golem?"

STYLE = """
:root { color-scheme: light; --bg: #faf7f2; --panel: #ffffff; --fg: #2a241e; --muted: #75695c; --line: #e8e0d4; --accent: #b4532a; --accent-soft: #f6e6dc; --code: #f3ede4; --ok: #2e7d4f; --ok-soft: #e3f2e8; --bad: #b3261e; --bad-soft: #fbe5e3; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { color-scheme: dark; --bg: #1a1714; --panel: #221e1a; --fg: #efe8de; --muted: #a89a8a; --line: #3a332c; --accent: #e08a5a; --accent-soft: #3a261b; --code: #2a2520; --ok: #7cc79a; --ok-soft: #1e3326; --bad: #f09a92; --bad-soft: #3a1f1d; }
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0; }
a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
header { border-bottom: 1px solid var(--line); background: var(--panel); }
.bar { max-width: 1120px; margin: 0 auto; padding: 10px 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
.brand { color: var(--fg); font-weight: 650; font-size: 16px; white-space: nowrap; }
.brand:hover { text-decoration: none; }
.search { flex: 1; display: flex; min-width: 200px; max-width: 420px; margin-left: auto; }
.search input { flex: 1; min-width: 0; font: inherit; padding: 6px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--bg); color: var(--fg); }
main { max-width: 1120px; margin: 0 auto; padding: 28px 16px 56px; }
h1 { font-size: 1.6rem; line-height: 1.25; margin: 0 0 10px; word-break: break-word; font-weight: 650; }
h1 .org { color: var(--muted); font-weight: 450; }
h2 { font-size: 1rem; margin: 28px 0 10px; }
.muted { color: var(--muted); }
code, pre { font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
code { background: var(--code); padding: 1px 5px; border-radius: 4px; overflow-wrap: anywhere; }
pre { margin: 0; background: var(--code); padding: 10px 12px; border-radius: 8px; white-space: pre-wrap; overflow-wrap: anywhere; }
pre code { background: none; padding: 0; }
.hero { padding: 8px 0 20px; }
.hero h1 { font-size: 1.9rem; }
.hero p { margin: 0; max-width: 640px; }
.hero + .box { max-width: 720px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
.card { display: block; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; color: var(--fg); }
.card:hover { border-color: var(--accent); text-decoration: none; }
.card b { display: block; word-break: break-word; margin-bottom: 8px; }
.empty { background: var(--panel); border: 1px dashed var(--line); border-radius: 12px; padding: 20px; }
.chips, .badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 8px; }
.chip { padding: 2px 10px; border: 1px solid var(--line); border-radius: 999px; font-size: 13px; color: var(--fg); background: var(--panel); }
.chip.on { border-color: var(--accent); color: var(--accent); font-weight: 600; }
.badge { padding: 1px 8px; border-radius: 6px; font-size: 12.5px; background: var(--code); color: var(--muted); }
.badge.ok { background: var(--ok-soft); color: var(--ok); font-weight: 600; }
.badge.bad { background: var(--bad-soft); color: var(--bad); font-weight: 600; }
.badge.accent { background: var(--accent-soft); color: var(--accent); }
.title { margin-bottom: 4px; }
.title + .grid > div > h2:first-child { margin-top: 12px; }
.grid { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 32px; align-items: start; }
@media (max-width: 860px) { .grid { grid-template-columns: minmax(0, 1fr); } aside { order: -1; } }
.box { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
aside .box + .box { margin-top: 12px; }
.box h3 { margin: 0 0 10px; font-size: 14px; }
blockquote { margin: 0; padding: 10px 14px; border-left: 3px solid var(--accent); background: var(--panel); border-radius: 0 8px 8px 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 500; width: 36%; }
td.n, th.n { text-align: right; white-space: nowrap; width: auto; }
.facts { table-layout: fixed; } .facts th { width: 30%; } .facts td { overflow-wrap: anywhere; } .facts tr:last-child th, .facts tr:last-child td { border-bottom: 0; }
details { margin-top: 28px; }
summary { cursor: pointer; font-weight: 600; }
.sizes { display: flex; height: 10px; border-radius: 5px; overflow: hidden; margin: 10px 0; background: var(--code); }
.sizes span { background: var(--accent); opacity: .85; border-right: 1px solid var(--panel); min-width: 1px; }
.sizes span:nth-child(2n) { opacity: .55; }
.tabs { display: flex; flex-wrap: wrap; gap: 4px; }
.tabs > input { position: absolute; opacity: 0; pointer-events: none; }
.tabs > label { padding: 3px 10px; border-radius: 6px; cursor: pointer; font-size: 13px; color: var(--muted); }
.tabs > input:checked + label { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
.tabs > input:focus-visible + label { outline: 2px solid var(--accent); }
.tabs > .panel { display: none; width: 100%; margin-top: 8px; }
.tabs > input:nth-of-type(1):checked ~ .panel:nth-of-type(1), .tabs > input:nth-of-type(2):checked ~ .panel:nth-of-type(2), .tabs > input:nth-of-type(3):checked ~ .panel:nth-of-type(3) { display: block; }
.cmd { position: relative; }
.cmd pre { padding-right: 64px; }
.copy { display: none; position: absolute; top: 6px; right: 6px; font: 12px system-ui, sans-serif; padding: 2px 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--fg); cursor: pointer; }
.js .copy { display: inline-block; }
.inline-copy { position: static; margin-left: 6px; padding: 0 6px; }
"""

SCRIPT = """
document.documentElement.classList.add('js');
document.addEventListener('click', function (event) {
  var button = event.target.closest('[data-copy]');
  if (!button || !navigator.clipboard) return;
  navigator.clipboard.writeText(button.dataset.copy).then(function () {
    var label = button.textContent;
    button.textContent = 'Copied';
    setTimeout(function () { button.textContent = label; }, 1200);
  });
});
"""


def e(text) -> str:
    return html.escape(str(text))


def page(title: str, body: str, query: str = "") -> bytes:
    header = f"""<header><div class="bar">
<a class="brand" href="/">🗿 shem registry</a>
<form class="search" action="/go" role="search"><input name="repo" value="{e(query)}" placeholder="Open a repo: ORG/NAME" aria-label="Open a repo" required></form>
</div></header>"""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{e(title)}</title><style>{STYLE}</style><script>{SCRIPT}</script></head>"
        f"<body>{header}<main>{body}</main></body></html>"
    ).encode()


def size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} GB"


def rows(pairs, cls: str = "") -> str:
    body = "".join(f"<tr><th>{e(k)}</th><td>{v}</td></tr>" for k, v in pairs if v)
    return f'<table class="{cls}">{body}</table>'


def command(text: str) -> str:
    """A command, with a button that copies it."""
    return f'<div class="cmd"><pre><code>{e(text)}</code></pre><button class="copy" data-copy="{e(text)}">Copy</button></div>'


def badges(summary: dict) -> str:
    out = []
    verified = summary.get("verified")
    if verified is True:
        out.append('<span class="badge ok">✓ Verified</span>')
    elif verified is False:
        out.append('<span class="badge bad">✗ Differs from transformers</span>')
    if summary.get("kind") == "artifact":
        out.append('<span class="badge accent">ModelPack</span>')
    if summary.get("parameters"):
        out.append(f'<span class="badge">{summary["parameters"] / 1e9:.2f}B params</span>')
    for key in ("precision", "licence"):
        if summary.get(key):
            out.append(f'<span class="badge">{e(summary[key])}</span>')
    if summary.get("primed"):
        out.append('<span class="badge accent">Primed session</span>')
    return f'<div class="badges">{"".join(out)}</div>' if out else ""


def home(host: str, served: dict[str, dict]) -> bytes:
    if served:
        cards = "".join(
            f'<a class="card" href="/{e(repo)}"><b>{e(repo)}</b>{badges(summary)}'
            f'<span class="muted">{" · ".join(e(t) for t in sorted(summary.get("tags", ())))}'
            f"{' · ' + size(summary['size']) if summary.get('size') else ''}</span></a>"
            for repo, summary in sorted(served.items())
        )
        listed = f'<div class="cards">{cards}</div>'
    else:
        listed = '<div class="empty muted">Nothing yet. Open a repo above, or pull one through this registry.</div>'
    body = f"""
<section class="hero">
<h1>Hugging Face repos as OCI images</h1>
<p class="muted">Pull an image <code>shem push</code> put on the Hub with any OCI client. Revisions are tags: <code>latest</code> is <code>main</code>. Names are lowercase, as OCI's are; the Hub finds a repo in any case.</p>
</section>
<div class="box"><h3>Pull</h3>{command(f"oras pull {host}/ORG/NAME:latest")}</div>
<h2>Recently served</h2>
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
    blob = hub.blob(repo, last["digest"].removeprefix("sha256:"))
    names = [m.name.removeprefix("./") for m, _ in members(blob, last["mediaType"])]
    return bool(names) and all(n == STATE_DIR or n.startswith(STATE_DIR + "/") for n in names)


def option(entrypoint: list[str], name: str) -> str | None:
    for i, arg in enumerate(entrypoint[:-1]):
        if arg == name:
            return entrypoint[i + 1]
    return None


def image_details(hub, repo: str, manifest: dict, config: dict, summary: dict) -> str:
    """The image's system prompt and model, noting what it finds in `summary`."""
    process = config.get("config") or {}
    entrypoint = process.get("Entrypoint") or []
    labels = process.get("Labels") or {}
    model_dir = labels.get(f"{LABEL}model")
    out = ""
    if model_dir:
        summary["primed"] = primed(hub, repo, manifest)
    system = option(entrypoint, "--system")
    if system:
        out += f"<h2>System prompt</h2><blockquote>{e(system)}</blockquote>"
    model = model_manifest(hub, repo, manifest, model_dir) if model_dir else None
    if model:
        source = model.get("source") or {}
        kinds = Counter(block["type"] for block in model.get("blocks", []))
        summary["parameters"] = model.get("parameters")
        summary["precision"] = ", ".join(model.get("precision", []))
        summary["licence"] = source.get("spdx") or source.get("license")
        summary["source"] = source
        out += "<h2>Model</h2>" + rows(
            [
                ("Architecture", e(source.get("architecture", ""))),
                ("Blocks", e(", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items())))),
                ("Parameters", f"{model['parameters']:,}" if model.get("parameters") else ""),
                ("Precision", e(summary["precision"])),
                ("Format", f"<code>{e(model.get('format', ''))}</code>"),
            ]
        )
    out += "<h2>Run</h2>" + rows(
        [
            ("Entrypoint", f"<code>{e(json.dumps(entrypoint))}</code>" if entrypoint else ""),
            ("Cmd", f"<code>{e(json.dumps(process.get('Cmd')))}</code>" if process.get("Cmd") else ""),
            ("Machine", e(labels.get(f"{LABEL}machine", ""))),
        ]
    )
    return out


def artifact_details(config: dict, summary: dict) -> str:
    descriptor, settings = config.get("descriptor") or {}, config.get("config") or {}
    summary["precision"] = settings.get("precision")
    return "<h2>Model</h2>" + rows(
        [
            ("Name", e(descriptor.get("name", ""))),
            ("Family", e(descriptor.get("family", ""))),
            ("Architecture", e(settings.get("architecture", ""))),
            ("Format", e(settings.get("format", ""))),
            ("Parameters", e(settings.get("paramSize", ""))),
            ("Precision", e(settings.get("precision", ""))),
        ]
    )


def comparison_details(comparison: dict | None) -> str:
    if not comparison:
        return ""
    return "<h2>Against transformers</h2>" + rows(
        [
            ("Largest logit difference", f"{comparison['difference']:.2e} <span class='muted'>(tolerance 1e-3)</span>"),
            ("Greedy reply", f"{comparison['length']} tokens, {'the same' if comparison['tokens'] else 'different'}"),
        ]
    )


def layers(manifest: dict) -> str:
    entries = manifest.get("layers") or []
    total = sum(layer["size"] for layer in entries) or 1
    strip = "".join(
        f'<span style="width:{100 * layer["size"] / total:.3f}%" title="{i}: {size(layer["size"])}"></span>'
        for i, layer in enumerate(entries)
    )
    body = ""
    for i, layer in enumerate(entries):
        annotations = layer.get("annotations") or {}
        name = next((v for k, v in annotations.items() if k.endswith((".filepath", ".title"))), "")
        kind = layer["mediaType"].rsplit("/", 1)[-1].removeprefix("vnd.")
        body += (
            f"<tr><td class='n muted'>{i}</td><td>{e(name) if name else ''} <span class='muted'>{e(kind)}</span></td>"
            f"<td class='n'>{size(layer['size'])}</td></tr>"
        )
    return (
        f"<details><summary>Layers <span class='muted'>· {len(entries)} · {size(total)}</span></summary>"
        f"<div class='sizes' aria-hidden='true'>{strip}</div><table>{body}</table></details>"
    )


def use(host: str, repo: str, tag: str, artifact: bool) -> str:
    """The box of commands to use the image, a tab for each client."""
    reference = repo if tag == "latest" else f"{repo}:{tag}"
    clients = [("shem", f'shem run hf.co/{reference} "{PROMPT}"'), ("oras", f"oras pull {host}/{repo}:{tag}")]
    if not artifact:
        docker = (
            f'docker run --rm --runtime io.containerd.shem.v1 --platform linux/riscv32 {host}/{repo}:{tag} "{PROMPT}"'
        )
        clients.append(("docker", docker))
    inputs = "".join(
        f'<input type="radio" name="use" id="use-{name}"{" checked" if i == 0 else ""}><label for="use-{name}">{name}</label>'
        for i, (name, _) in enumerate(clients)
    )
    panels = "".join(f'<div class="panel">{command(text)}</div>' for _, text in clients)
    return f'<div class="box"><h3>Use this {"artifact" if artifact else "image"}</h3><div class="tabs">{inputs}{panels}</div></div>'


def facts(repo: str, tag: str, digest: str, manifest: dict, summary: dict, tags: list[str]) -> str:
    source = summary.get("source") or {}
    origin = ""
    if source.get("repo"):
        rev = (source.get("revision") or "")[:12]
        origin = f'<a href="https://huggingface.co/{e(source["repo"])}">{e(source["repo"])}</a>'
        origin += f' <span class="muted">@ {e(rev)}</span>' if rev else ""
    short = f'<code>{e(digest[:15])}…</code><button class="copy inline-copy" data-copy="{e(digest)}">Copy</button>'
    total = sum(layer["size"] for layer in manifest.get("layers") or [])
    other = "modelpack" if tag != "modelpack" and "modelpack" in tags else ""
    return (
        '<div class="box">'
        + rows(
            [
                ("Digest", short),
                ("Size", f"{size(total)} <span class='muted'>· {len(manifest.get('layers') or [])} layers</span>"),
                ("Source", origin),
                ("ModelPack", f'<a href="/{e(repo)}?tag={other}">{other}</a> tag' if other else ""),
                ("Hub", f'<a href="https://huggingface.co/{e(repo)}">huggingface.co/{e(repo)}</a>'),
            ],
            "facts",
        )
        + "</div>"
    )


def repo_page(hub, host: str, repo: str, tag: str, summary: dict) -> bytes:
    """The page of `repo` at `tag`. What it shows of `latest` goes into
    `summary`, for the home page's card."""
    tags = hub.tags(repo)
    summary.setdefault("tags", set()).add(tag)
    view: dict = {}
    entry = (hub.index(repo, tag).get("manifests") or [None])[0]
    if entry is None:
        return page(repo, f"<h1>{e(repo)}</h1><p>{e(repo)}:{e(tag)} lists no manifest.</p>", repo)
    data = json.loads(hub.blob(repo, entry["digest"].removeprefix("sha256:")))
    # An index lists the image under platforms: they are all the same image
    while "manifests" in data:
        data = json.loads(hub.blob(repo, data["manifests"][0]["digest"].removeprefix("sha256:")))
    manifest = data
    config = json.loads(hub.blob(repo, manifest["config"]["digest"].removeprefix("sha256:")))
    artifact = ".model.config." in manifest["config"].get("mediaType", "")
    view["kind"] = "artifact" if artifact else "image"
    view["size"] = sum(layer["size"] for layer in manifest.get("layers") or [])
    details = artifact_details(config, view) if artifact else image_details(hub, repo, manifest, config, view)
    comparison = hub.comparison(repo, tag)
    if comparison:
        view["verified"] = bool(comparison.get("passed"))
    details += comparison_details(comparison)
    labels = (config.get("config") or {}).get("Labels") or {}
    others = [(k, e(v)) for k, v in sorted(labels.items()) if not k.startswith(LABEL)]
    org, _, name = repo.partition("/")
    chips = "".join(
        f'<a class="chip on" href="/{e(repo)}?tag={e(t)}" aria-current="page">{e(t)}</a>'
        if t == tag
        else f'<a class="chip" href="/{e(repo)}?tag={e(t)}">{e(t)}</a>'
        for t in tags
    )
    body = f"""
<div class="title">
<h1><span class="org">{e(org)} /</span> {e(name)}</h1>
{badges(view)}
<div class="chips">{chips}</div>
</div>
<div class="grid">
<div>
{details}
{layers(manifest)}
{"<details><summary>Labels</summary>" + rows(others) + "</details>" if others else ""}
</div>
<aside>
{use(host, repo, tag, artifact)}
{facts(repo, tag, entry["digest"], manifest, view, tags)}
</aside>
</div>
"""
    if tag == "latest":
        summary.update(view)
    return page(f"{repo}:{tag} · shem registry", body, repo)
