"""shem replay: a Hugging Face Space that checks a receipt of a model's reply
by running it again, and shows what a change to the prompt changes. The
steps are in replay.py; this is their UI."""

import html
import json
import os

import gradio as gr

from replay import CLOSE, IMAGE, MAX_BYTES, MAX_TOKENS, Note, Refused, Reply, Verdict, ask, difference, read, replay

#: A receipt to start from: of an image the shem my repo Space made
EXAMPLE = """{
 "format": "shem.receipt/v1",
 "image": "hf.co/jgalego/smollm2-135m-instruct-shem@sha256:5339b1f9226f33bdf3a557e21349dd7a4ad498f29dcebf1a9ef3601dcf9134fd",
 "prompt": "Who are you?",
 "reply": "I am a Shem, a golem created by the word of the Lord.",
 "tokens": [57, 744, 253, 334, 1676, 28, 253, 685, 12164, 2312, 411, 260, 2229, 282, 260, 5151, 30, 2],
 "margins": [1.468582, 3.081518, 0.482496, 0.352477, 4.440741, 2.73415, 3.147675, 0.110825, 4.741564, 0.202389, 2.020863, 0.145825, 0.144969, 0.023771, 0.029436, 0.159035, 0.954207, 0.645074],
 "runtime": {"shem": "0.3.0", "torch": "2.15.0+cpu", "device": "cpu", "machine": "x86_64", "precision": "float32", "blocks": {"attention": 1, "mlp": 1}}
}
"""

INTRO = f"""
# 🧾 shem replay

**Did the model really say that?** A receipt says what a model was asked and what it replied: the model image by its digest, the prompt, the reply, and every token the model chose. [shem](https://github.com/JGalego/shem) writes one with `shem run --receipt FILE`, and decodes greedily, so the same image and prompt choose the same tokens. This runs the receipt's prompt on its image again and compares them.

Edit the reply or the prompt in the receipt below and it stops matching. Receipts of images in public Hugging Face repos, up to {MAX_BYTES / 1e9:g} GB and {MAX_TOKENS} tokens: for any other, `shem replay FILE` does the same on your machine.
"""

CHANGE = """
## What would a change to the prompt change?

Nothing is sampled, so when the same image gets another prompt, any difference in the reply comes from the change, and from nothing else. Replay a receipt, then change its prompt here.
"""

#: Scoped to the component; Gradio's variables follow its light and dark themes
CSS = """
.tokens { font: 15px/2.1 var(--font-mono); white-space: pre-wrap; overflow-wrap: anywhere; }
.token { padding: 3px 0; border-bottom: 3px solid transparent; }
.token.firm { border-color: #2e9d5b55; }
.token.narrow { border-color: #d9a400; background: #d9a40022; }
.token.close { border-color: #d4372c; background: #d4372c22; }
.token.end { color: var(--body-text-color-subdued); padding: 3px 4px; }
.legend { margin-top: 10px; font-size: 12.5px; color: var(--body-text-color-subdued); display: flex; flex-wrap: wrap; gap: 6px 16px; }
.legend .token { padding: 0 2px; }
.changed { font-size: 15px; line-height: 1.7; white-space: pre-wrap; overflow-wrap: anywhere; }
.changed del { color: #d4372c; background: #d4372c1c; text-decoration-thickness: 1px; }
.changed ins { color: #1f7a45; background: #2e9d5b24; text-decoration: none; }
.empty { color: var(--body-text-color-subdued); font-size: 13.5px; }
"""
#: Under this margin a choice is shown as narrow, though it holds on any device
NARROW = 0.25
NO_TOKENS = (
    '<div class="empty">The tokens of a replayed receipt show here, each marked by how firmly it was chosen.</div>'
)
NO_CHANGE = '<div class="empty">The changed reply shows here, against the receipt\'s.</div>'


def e(text) -> str:
    return html.escape(str(text))


def banner(verdict: Verdict, receipt: dict) -> str:
    """The verdict, in a heading and a sentence."""
    named = IMAGE.fullmatch(receipt["image"])
    image = f"[`{named['reference']}`](https://huggingface.co/{named['repo']}) at `{named['digest'][:19]}…`"
    if verdict.word == "verified":
        return f"## ✅ Verified\n\nThe model chose the receipt's {len(receipt['tokens'])} tokens again: {image} gives this reply to this prompt."
    if verdict.word == "undecided":
        return f"## ⚠️ Undecided\n\nOn {image}, {verdict.why}."
    return f"## ❌ Mismatch\n\nOn {image}, {verdict.why}."


def strip(verdict: Verdict, receipt: dict) -> str:
    """The receipt's tokens, each marked by how firmly the model chose it."""
    if not verdict.pieces:
        return NO_TOKENS
    close = CLOSE.get((receipt.get("runtime") or {}).get("precision"), CLOSE["float32"])
    spans = []
    for at, (text, margin) in enumerate(verdict.pieces, 1):
        kind = "close" if margin < close else "narrow" if margin < NARROW else "firm"
        title = f"Token {at}: chosen by {margin:.3g} over the runner-up"
        if text:
            spans.append(f'<span class="token {kind}" title="{e(title)}">{e(text)}</span>')
        else:
            spans.append(f'<span class="token end {kind}" title="{e(title)}, to end the reply">⏎</span>')
    legend = (
        '<div class="legend"><span><span class="token firm">firm</span> chosen by a wide margin</span>'
        f'<span><span class="token narrow">narrow</span> by under {NARROW:g}, in logits</span>'
        f'<span><span class="token close">close</span> by under {close:g}: another device may choose otherwise</span>'
        "<span>Hover over a token for its margin</span></div>"
    )
    return f'<div class="tokens">{"".join(spans)}</div>{legend}'


def check(text: str):
    """Replay the receipt in `text`, yielding the status, the reply, the
    tokens and the prompt to change as they come."""
    try:
        receipt = read(text)
        for item in replay(text):
            if isinstance(item, Note):
                yield f"⏳ {item.text}…", gr.skip(), gr.skip(), gr.skip()
            elif isinstance(item, Reply):
                yield gr.skip(), item.text, gr.skip(), gr.skip()
            elif isinstance(item, Verdict):
                yield banner(item, receipt), item.reply, strip(item, receipt), receipt["prompt"]
    except Refused as error:
        yield f"## ❌ Not replayed\n\n{error}", "", NO_TOKENS, gr.skip()
    except Exception as error:
        yield f"## ❌ Not replayed\n\nThe replay failed: {error}", "", NO_TOKENS, gr.skip()


def change(text: str, prompt: str):
    """Reply to `prompt` on the receipt's image, yielding the status, how
    the reply differs from the receipt's, and the new reply's own receipt."""
    try:
        receipt = read(text)
        if prompt.strip() == receipt["prompt"].strip():
            raise Refused("That is the receipt's own prompt: change a word of it.")
        for item in ask(text, prompt):
            if isinstance(item, Note):
                yield f"⏳ {item.text}…", gr.skip(), gr.skip()
            elif isinstance(item, Reply):
                yield gr.skip(), f'<div class="changed">{e(item.text)}</div>', gr.skip()
            else:
                marks = {"same": "{}", "removed": "<del>{}</del>", "added": "<ins>{}</ins>"}
                runs = difference(receipt["reply"], item["reply"])
                changed = "".join(marks[kind].format(e(words)) for kind, words in runs)
                same = all(kind == "same" for kind, _ in runs)
                status = (
                    "The reply is the same, word for word."
                    if same
                    else "The reply changed: the receipt's words are struck out, and the new ones are in green."
                )
                fields = [f" {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}" for key, value in item.items()]
                yield status, f'<div class="changed">{changed}</div>', "{\n" + ",\n".join(fields) + "\n}\n"
    except Refused as error:
        yield f"❌ {error}", gr.skip(), gr.skip()
    except Exception as error:
        yield f"❌ The run failed: {error}", gr.skip(), gr.skip()


with gr.Blocks(title="shem replay", analytics_enabled=False) as demo:
    gr.Markdown(INTRO)
    receipt = gr.Code(EXAMPLE, language="json", label="Receipt", lines=12, max_lines=24)
    button = gr.Button("Replay", variant="primary")
    status = gr.Markdown()
    reply = gr.Textbox(label="The reply, replayed", lines=3, interactive=False)
    tokens = gr.HTML(NO_TOKENS, css_template=CSS, label="The receipt's tokens")
    gr.Markdown(CHANGE)
    prompt = gr.Textbox(label="Changed prompt", placeholder="Replay a receipt first", lines=2)
    again = gr.Button("Run the changed prompt")
    outcome = gr.Markdown()
    changed = gr.HTML(NO_CHANGE, css_template=CSS, label="The reply, against the receipt's")
    with gr.Accordion("The changed reply's own receipt", open=False):
        issued = gr.Code(language="json", show_label=False, lines=10, max_lines=24, interactive=False)
    # One model at a time, whichever button asks for it: the spinners over the statuses only
    busy = {"concurrency_limit": 1, "concurrency_id": "model"}
    button.click(check, [receipt], [status, reply, tokens, prompt], show_progress_on=status, **busy)
    again.click(change, [receipt, prompt], [outcome, changed, issued], show_progress_on=outcome, **busy)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
    )
