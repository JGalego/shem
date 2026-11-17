"""LLaVA against transformers, on tiny random models with a CLIP and with a
SigLIP encoder, built small by transformers itself: the imported model, reading
a picture's rows where the chat has its image tokens, must give transformers'
logits for the same pixels, and the pixels must be the processor's."""

import json

import pytest
import torch
from test_vision import ask, png

from shem_neural.model.blocks import Model
from shem_neural.model.generate import Session, next_turn
from shem_neural.model.importers import import_checkpoint
from shem_neural.model.manifest import Manifest
from shem_neural.model.tokenizer import Tokenizer
from shem_neural.model.vision import picture

transformers = pytest.importorskip("transformers")
Image = pytest.importorskip("PIL.Image")

VOCAB = 64
IMAGE = 60
#: A LLaVA chat template as its repo has it (llava-interleave's), in a file of its own as the Hub keeps them
TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n'}}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>' }}{% endfor %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ '\\n' + content['text'] }}{% endfor %}"
    "{{'<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
KINDS = {
    # CLIP, the layer before the last, without its class token, as LLaVA 1.5 reads it
    "clip": dict(
        vision=lambda: transformers.CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            image_size=28,
            patch_size=7,
        ),
        feature=-2,
        select="default",
        pre=dict(
            size={"shortest_edge": 28},
            crop_size={"height": 28, "width": 28},
            do_center_crop=True,
            resample=3,
            image_mean=CLIP_MEAN,
            image_std=CLIP_STD,
        ),
        tokens=16,
    ),
    # The same, with a tokenizer that begins a text with `<s>` as Llama's does, which the template does not write
    "clip-bos": dict(
        vision=lambda: transformers.CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            image_size=28,
            patch_size=7,
        ),
        feature=-2,
        select="default",
        pre=dict(
            size={"shortest_edge": 28},
            crop_size={"height": 28, "width": 28},
            do_center_crop=True,
            resample=3,
            image_mean=CLIP_MEAN,
            image_std=CLIP_STD,
        ),
        tokens=16,
        bos=True,
    ),
    # SigLIP, the last layer, all its patches, as LLaVA-interleave reads it
    "siglip": dict(
        vision=lambda: transformers.SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            image_size=28,
            patch_size=7,
        ),
        feature=-1,
        select="full",
        pre=dict(size={"height": 28, "width": 28}, resample=3, image_mean=[0.5] * 3, image_std=[0.5] * 3),
        tokens=16,
    ),
}


def config(kind: str):
    spec = KINDS[kind]
    text = transformers.LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return transformers.LlavaConfig(
        vision_config=spec["vision"](),
        text_config=text,
        image_token_index=IMAGE,
        vision_feature_layer=spec["feature"],
        vision_feature_select_strategy=spec["select"],
    )


def files(directory, kind: str):
    from tokenizers import AddedToken, models, pre_tokenizers, processors
    from tokenizers import Tokenizer as Hf

    vocab = {f"t{i}": i for i in range(VOCAB)}
    names = {IMAGE: "<image>", 3: "<|im_start|>", 4: "<|im_end|>", 5: "user", 6: "assistant", 7: "system"}
    if KINDS[kind].get("bos"):
        names[1] = "<s>"
    for id_, name in names.items():
        vocab.pop(f"t{id_}")
        vocab[name] = id_
    inner = Hf(models.WordLevel(vocab, unk_token="t0"))
    inner.pre_tokenizer = pre_tokenizers.Whitespace()
    inner.add_special_tokens([AddedToken(name, special=True) for name in names.values()])
    if KINDS[kind].get("bos"):
        inner.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    inner.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))
    # The template is in a file of its own, as llava-interleave's is
    (directory / "chat_template.json").write_text(json.dumps({"chat_template": TEMPLATE}))
    (directory / "preprocessor_config.json").write_text(json.dumps(KINDS[kind]["pre"]))


@pytest.fixture(scope="module", params=sorted(KINDS))
def llava(request, tmp_path_factory):
    torch.manual_seed(0)
    reference = transformers.LlavaForConditionalGeneration(config(request.param)).eval()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    checkpoint = tmp_path_factory.mktemp("llava")
    reference.save_pretrained(checkpoint)
    files(checkpoint, request.param)
    out = tmp_path_factory.mktemp("imported")
    import_checkpoint(checkpoint, out)
    return request.param, reference, out


def processor(kind: str):
    pre = KINDS[kind]["pre"]
    cls = transformers.CLIPImageProcessor if kind == "clip" else transformers.SiglipImageProcessor
    return cls(**{key: value for key, value in pre.items() if key != "resample"}, resample=pre["resample"])


def logits(reference, model, tokens: list[int], urls: list[str]) -> torch.Tensor:
    pixels = torch.stack([view for url in urls for view in model.vision.views({"url": url})])
    with torch.no_grad():
        return reference(input_ids=torch.tensor([tokens]), pixel_values=pixels).logits


def test_the_import_keeps_the_encoder_and_the_projector(llava):
    kind, _, out = llava
    manifest = Manifest.load(out)
    assert manifest.vision.params["family"] == "llava"
    expected = {"token": "<image>", "id": IMAGE, "count": KINDS[kind]["tokens"]}
    assert manifest.tokenizer["image"] == expected | ({"bos": 1} if KINDS[kind].get("bos") else {})
    # Only the layers the config reads are kept: all but the last for the layer before it
    kept = {name.split(".")[0] for name in manifest.vision.tensors if name[0].isdigit()}
    assert len(kept) == (3 if kind == "siglip" else 2)
    assert ("cls" in manifest.vision.tensors) == (kind != "siglip")
    Model(out, verify=True)


def test_a_picture_is_the_processors_pixels(llava):
    kind, _, out = llava
    model = Model(out)
    for size in [(40, 30), (30, 40), (28, 28), (61, 100)]:
        url = png(3, size=size)
        ours = model.vision.views({"url": url})
        assert len(ours) == 1
        theirs = processor(kind)(
            images=Image.open(__import__("io").BytesIO(picture(url))).convert("RGB"), return_tensors="pt"
        )
        assert torch.allclose(ours[0], theirs["pixel_values"][0], atol=1e-5), (
            size,
            (ours[0] - theirs["pixel_values"][0]).abs().max(),
        )


def test_logits_match_transformers_with_images(llava):
    kind, reference, out = llava
    model = Model(out)
    tokenizer = Tokenizer(out, Manifest.load(out).tokenizer)
    urls = [png(1), png(2, size=(31, 17))]
    messages = [ask("t9 t10", *urls)]
    tokens = tokenizer.chat(messages)
    assert tokens.count(IMAGE) == 32, "16 tokens for each of the two pictures"
    # A tokenizer that begins its texts with `<s>` begins the prompt with it, once
    assert (tokens[0] == 1) == bool(KINDS[kind].get("bos"))
    assert tokens.count(1) == (1 if KINDS[kind].get("bos") else 0)
    session = Session(model)
    session.attach_images(messages)
    session.feed(tokens)
    expected = logits(reference, model, tokens, urls)
    assert torch.allclose(session.logits, expected[0, -1], atol=1e-4), (session.logits - expected[0, -1]).abs().max()


def test_a_conversation_reads_each_turns_picture(llava):
    _, reference, out = llava
    model = Model(out)
    tokenizer = Tokenizer(out, Manifest.load(out).tokenizer)
    first, second = ask("t9", png(5)), ask("t10 t11", png(6, size=(20, 60)))
    session = Session(model)
    session.feed(next_turn(session, tokenizer, [first]))
    session.feed([20, 21])
    session.messages = [first, {"role": "assistant", "content": "t20 t21"}]
    session.feed(next_turn(session, tokenizer, [second]))
    expected = logits(reference, model, session.tokens, [png(5), png(6, size=(20, 60))])
    assert torch.allclose(session.logits, expected[0, -1], atol=1e-4)
