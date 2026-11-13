"""shem my repo: a Hugging Face Space that turns a checkpoint into a shem
image, compares it with transformers, and pushes it to the user's account.
The steps are in convert.py; this is their UI."""

import shutil
import tempfile
from pathlib import Path

import gradio as gr
from huggingface_hub import HfApi

from convert import MAX_BYTES, MAX_PRIMED, MAX_VERIFIED, Refused, Result, check, convert, refuse, target
from log_view import CSS, Log

INTRO = f"""
# 🗿 shem my repo

Turn a Hugging Face checkpoint into a [shem](https://github.com/JGalego/shem) model image, an OCI image you can `shem run`, pull through `shem registry` with any OCI client, or `docker run` with shem's runtime. The image goes to a repo of your own, as an OCI image layout, with a [ModelPack](https://github.com/modelpack/model-spec) artifact over the same blobs on its `modelpack` branch.

Checkpoints of the architectures shem imports, whose safetensors weights take up to {MAX_BYTES / 1e9:g} GB. A system prompt is primed up to {MAX_PRIMED / 1e9:g}B parameters. Up to {MAX_VERIFIED / 1e9:g}B, the image is also compared with transformers, and the model card says how close it came.
"""


def start(
    source: str,
    system: str,
    verify: bool,
    modelpack: bool,
    private: bool,
    profile: gr.OAuthProfile | None,
    oauth_token: gr.OAuthToken | None,
):
    """Convert `source`, yielding the status, the steps and the raw log as they change."""
    if profile is None or oauth_token is None:
        yield "Log in with Hugging Face first: the image goes to your account.", Log().html(), ""
        return
    source = source.strip().removeprefix("https://huggingface.co/").removeprefix("hf.co/").strip("/")
    log = Log()
    try:
        checkpoint = check(source, HfApi(token=oauth_token.token))
        refuse(checkpoint, system)
    except Refused as error:
        yield f"❌ {error}", log.html(), log.text()
        return
    repo = target(profile.username, checkpoint.repo)
    status = f"Converting [{checkpoint.repo}](https://huggingface.co/{checkpoint.repo}) ({checkpoint.architecture}, {checkpoint.parameters / 1e9:.2f}B parameters) to `{repo}`…"
    yield status, log.html(), log.text()
    work = Path(tempfile.mkdtemp(prefix="shem-"))
    try:
        steps = convert(
            checkpoint,
            work,
            oauth_token.token,
            system,
            verify,
            push=repo,
            modelpack=modelpack,
            private=private,
            cache=work / "hf",
        )
        for item in steps:
            if isinstance(item, Result):
                log.end("done")
                yield done(item, repo), log.html(), log.text()
            else:
                log.add(item)
                yield status, log.html(), log.text()
    except Exception as error:
        log.end("failed")
        yield f"❌ Converting {checkpoint.repo} failed: {error}", log.html(), log.text()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def done(result: Result, repo: str) -> str:
    comparison = result.comparison
    if comparison and comparison["passed"]:
        verdict = f"✅ Verified against transformers: logits within {comparison['difference']:.1e}, and the same {comparison['length']}-token reply."
    elif comparison:
        verdict = "❌ The import differs from transformers: see the model card."
    else:
        verdict = f"Not verified. {result.unverified or ''}"
    return f"""## Done: [{repo}]({result.url})

{verdict}

```sh
shem run hf.co/{repo} "What is a golem?"
```
"""


with gr.Blocks(title="shem my repo") as demo:
    gr.Markdown(INTRO)
    gr.LoginButton()
    source = gr.Textbox(label="Hugging Face model", placeholder="Qwen/Qwen3-0.6B")
    system = gr.Textbox(
        label="System prompt (optional)",
        info="Primed into a saved session at build time, so runs start after it",
        lines=3,
    )
    with gr.Row():
        verify = gr.Checkbox(value=True, label="Compare with transformers")
        modelpack = gr.Checkbox(value=True, label="Also push a ModelPack artifact")
        private = gr.Checkbox(value=False, label="Private repo")
    button = gr.Button("Convert", variant="primary")
    status = gr.Markdown()
    log = gr.HTML(Log().html(), css_template=CSS)
    with gr.Accordion("Raw log", open=False):
        raw = gr.Code(language=None, show_label=False, lines=12, max_lines=24, interactive=False)
    # The spinner over the status only, so the log shows as it comes
    button.click(start, [source, system, verify, modelpack, private], [status, log, raw], show_progress_on=status)

if __name__ == "__main__":
    # One conversion at a time: each takes the machine's memory and disk
    demo.queue(default_concurrency_limit=1, max_size=20).launch(server_name="0.0.0.0", server_port=7860)
