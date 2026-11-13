"""The Space's log, as a list of steps: each with its state, how long it
took, how it ended, and its output, where progress bars show as bars,
digests are shortened, and links can be followed."""

import html
import re
import time

from convert import Outcome, Step

#: A progress bar as tqdm draws it: `Label: 45%|████   | 12/27 [00:01<00:02, 9.1it/s]`
BAR = re.compile(r"^(?P<label>.*?):\s*(?P<percent>\d{1,3})%\|[^|]*\|\s*(?P<rest>.*)$")
DIGEST = re.compile(r"sha256:([0-9a-f]{12})[0-9a-f]{52}")
URL = re.compile(r"https?://[^\s<>\"]+")
#: How many lines of its output a step keeps
KEEP = 200

#: Scoped to the component; Gradio's variables follow its light and dark themes
CSS = """
.steps { display: flex; flex-direction: column; gap: 6px; font-size: 14px; }
.step { border: 1px solid var(--border-color-primary); border-radius: 10px; background: var(--block-background-fill); }
.step > summary, .step > .head { display: flex; align-items: center; gap: 10px; padding: 9px 12px; list-style: none; }
.step > summary { cursor: pointer; }
.step > summary::-webkit-details-marker { display: none; }
.step .title { font-weight: 600; color: var(--body-text-color); }
.step .detail { color: var(--body-text-color-subdued); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; flex: 1; }
.step .end { color: var(--body-text-color-subdued); font-size: 12.5px; white-space: nowrap; }
.step .end.bad { color: #d4372c; }
.icon { flex: none; width: 18px; height: 18px; border-radius: 50%; display: grid; place-items: center; font-size: 11px; font-weight: 700; color: #fff; }
.done .icon { background: #2e9d5b; }
.failed .icon { background: #d4372c; }
.skipped .icon { background: var(--border-color-primary); color: var(--body-text-color-subdued); }
.running .icon { border: 2px solid var(--border-color-primary); border-top-color: var(--color-accent); animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.out { margin: 0 12px 10px 40px; padding: 8px 10px; border-radius: 8px; background: var(--background-fill-secondary); font: 12.5px/1.55 var(--font-mono); color: var(--body-text-color-subdued); max-height: 260px; overflow-y: auto; }
.out .line { white-space: pre-wrap; overflow-wrap: anywhere; }
.out a { color: var(--link-text-color); }
.bar { display: grid; grid-template-columns: minmax(0, auto) 1fr auto; gap: 10px; align-items: center; }
.bar .label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar .track { height: 6px; border-radius: 3px; background: var(--border-color-primary); overflow: hidden; }
.bar .fill { display: block; height: 100%; border-radius: 3px; background: var(--color-accent); transition: width .3s; }
.bar .rest { white-space: nowrap; }
.idle { color: var(--body-text-color-subdued); padding: 12px; border: 1px dashed var(--border-color-primary); border-radius: 10px; text-align: center; }
"""


def duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    return f"{seconds // 60:.0f} min {seconds % 60:.0f} s"


def line(text: str) -> str:
    """A line of output: a progress bar as a bar, else the text, with digests
    shortened and links made links."""
    if match := BAR.match(text):
        rest = html.escape(match["rest"].split("[")[0].strip())
        return (
            f'<div class="bar"><span class="label">{html.escape(match["label"])}</span>'
            f'<span class="track"><span class="fill" style="width:{min(int(match["percent"]), 100)}%"></span></span>'
            f'<span class="rest">{match["percent"]}% · {rest}</span></div>'
        )
    return f'<div class="line">{inline(text)}</div>'


def inline(text: str) -> str:
    """Text, escaped, with digests shortened and links made links."""
    text = html.escape(DIGEST.sub(r"sha256:\1…", text))
    # A Hub link reads as the repo it leads to
    return URL.sub(
        lambda m: f'<a href="{m[0]}" target="_blank">{m[0].removeprefix("https://huggingface.co/")}</a>', text
    )


class Log:
    """The steps of a conversion, as it reports them."""

    def __init__(self):
        self.steps: list[dict] = []

    def add(self, item: str | Step | Outcome) -> None:
        now = time.monotonic()
        if isinstance(item, Step):
            self.end("done")
            state = "skipped" if item.skipped else "running"
            self.steps.append(
                dict(
                    title=item.title,
                    detail=item.detail,
                    state=state,
                    start=now,
                    end=None,
                    lines=[],
                    bar=None,
                    outcome=None,
                )
            )
            return
        if not self.steps:
            self.add(Step("Prepare"))
        step = self.steps[-1]
        if isinstance(item, Outcome):
            step["outcome"] = item.text
            self.end("done" if item.ok else "failed")
        elif item.startswith("\r"):
            step["bar"] = item[1:]
        else:
            step["bar"] = None
            step["lines"] = [*step["lines"][-KEEP:], item]

    def end(self, state: str) -> None:
        """End the step that runs, if one does, in `state`."""
        if self.steps and self.steps[-1]["state"] == "running":
            self.steps[-1].update(state=state, end=time.monotonic())

    def html(self) -> str:
        if not self.steps:
            return '<div class="idle">The steps show here as they run.</div>'
        out = []
        for step in self.steps:
            icon = {"done": "✓", "failed": "✗", "skipped": "–", "running": ""}[step["state"]]
            took = ""
            if step["state"] in ("done", "failed"):
                took = duration(step["end"] - step["start"])
            elif step["state"] == "running":
                took = duration(time.monotonic() - step["start"])
            ending = step["outcome"] or step["detail"]
            end_class = "end bad" if step["state"] == "failed" else "end"
            head = (
                f'<span class="icon">{icon}</span><span class="title">{html.escape(step["title"])}</span>'
                f'<span class="detail">{inline(ending)}</span>'
                f'<span class="{end_class}">{took}</span>'
            )
            body = [line(text) for text in step["lines"]] + ([line(step["bar"])] if step["bar"] else [])
            if body:
                opened = " open" if step["state"] in ("running", "failed") else ""
                out.append(
                    f'<details class="step {step["state"]}"{opened}><summary>{head}</summary>'
                    f'<div class="out">{"".join(body)}</div></details>'
                )
            else:
                out.append(f'<div class="step {step["state"]}"><div class="head">{head}</div></div>')
        return '<div class="steps">' + "".join(out) + "</div>"
