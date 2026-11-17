"""Vision-language models against transformers, on a tiny random SmolVLM.

It is built small and random by transformers itself, so these need no
downloads: the imported model, reading a picture's rows where the chat has
image tokens, must give transformers' logits for the same pixels, and the
pieces around it (the pictures' decoding, the image tokens, the API's parts,
sessions that go on across turns, receipts) must do what they say.
"""

import base64
import io
import json

import numpy as np
import pytest
import torch

from shem_neural.model import receipt
from shem_neural.model.blocks import FingerprintMismatch, Model
from shem_neural.model.generate import Session, next_turn
from shem_neural.model.importers import import_checkpoint
from shem_neural.model.manifest import Manifest
from shem_neural.model.serve import message
from shem_neural.model.tokenizer import Tokenizer
from shem_neural.model.vision import data_url, grid, larger, picture, tiles, written

transformers = pytest.importorskip("transformers")
Image = pytest.importorskip("PIL.Image")

VOCAB = 64
IMAGE, FAKE, WHOLE = 60, 61, 62
#: SmolVLM's chat template, as its repo has it
TEMPLATE = (
    "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}"
    "{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}"
    "{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}"
    "{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n"
    "{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)


def config():
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
    vision = dict(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, image_size=32, patch_size=8
    )
    return transformers.SmolVLMConfig(
        text_config=text.to_dict(), vision_config=vision, scale_factor=2, image_token_id=IMAGE
    )


def tokenizer_files(directory, split=False):
    """A word-level tokenizer whose special tokens are SmolVLM's, and its chat template."""
    from tokenizers import AddedToken, models, pre_tokenizers
    from tokenizers import Tokenizer as Hf

    vocab = {f"t{i}": i for i in range(VOCAB)}
    names = {IMAGE: "<image>", FAKE: "<fake_token_around_image>", WHOLE: "<global-img>"}
    names |= {3: "<|im_start|>", 4: "<end_of_utterance>", 5: "User", 6: "Assistant", 7: "System", 8: ":"}
    # The tokens that name a tile by its row and column
    names |= {40 + 4 * r + c: f"<row_{r + 1}_col_{c + 1}>" for r in range(4) for c in range(4)}
    for id_, name in names.items():
        vocab.pop(f"t{id_}")
        vocab[name] = id_
    inner = Hf(models.WordLevel(vocab, unk_token="t0"))
    inner.pre_tokenizer = pre_tokenizers.Whitespace()
    inner.add_special_tokens([AddedToken(name, special=True) for name in names.values()])
    inner.save(str(directory / "tokenizer.json"))
    settings = {"chat_template": TEMPLATE, "eos_token": "<end_of_utterance>"}
    (directory / "tokenizer_config.json").write_text(json.dumps(settings))
    settings = {"image_mean": [0.5] * 3, "image_std": [0.5] * 3, "do_image_splitting": split}
    (directory / "preprocessor_config.json").write_text(json.dumps(settings))


def build(tmp_path_factory, split):
    """The reference model, and the checkpoint imported from it, whose pictures are cut in tiles or not."""
    torch.manual_seed(0)
    reference = transformers.SmolVLMForConditionalGeneration(config()).eval()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    checkpoint = tmp_path_factory.mktemp("smolvlm")
    reference.save_pretrained(checkpoint)
    tokenizer_files(checkpoint, split)
    out = tmp_path_factory.mktemp("imported")
    import_checkpoint(checkpoint, out)
    return reference, out


@pytest.fixture(scope="module")
def vlm(tmp_path_factory):
    """A tiny SmolVLM that reads each picture whole."""
    return build(tmp_path_factory, split=False)


@pytest.fixture(scope="module")
def tiled(tmp_path_factory):
    """The same, with pictures cut in tiles, as the processor does by default."""
    return build(tmp_path_factory, split=True)


def png(seed: int, size=(20, 27), mode="RGB") -> str:
    """A random picture, as a `data:` URL."""
    pixels = np.random.default_rng(seed).integers(0, 256, (*size[::-1], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).convert(mode).save(buffer, "PNG")
    return data_url(buffer.getvalue())


def img(url: str, **more) -> dict:
    """A picture as a message holds it."""
    return {"type": "image", "url": url, **more}


def ask(text: str, *urls: str) -> dict:
    return {
        "role": "user",
        "content": [*({"type": "image", "url": url} for url in urls), {"type": "text", "text": text}],
    }


def reference_logits(reference, model, tokens: list[int], urls: list[str]) -> torch.Tensor:
    """transformers' logits for `tokens`, with the pixels shem reads the pictures as."""
    pixels = torch.stack([view for url in urls for view in model.vision.views(img(url))])[None]
    ones = torch.ones(1, pixels.shape[1], 32, 32, dtype=torch.long)
    with torch.no_grad():
        return reference(input_ids=torch.tensor([tokens]), pixel_values=pixels, pixel_attention_mask=ones).logits


def test_the_import_keeps_the_encoder_and_describes_how_images_are_written(vlm):
    _, out = vlm
    manifest = Manifest.load(out)
    assert manifest.vision.type == "vision"
    assert manifest.tokenizer["image"]["count"] == 4, "a 4x4 grid of patches, folded 2x2"
    assert manifest.tokenizer["image"] == {
        "token": "<image>",
        "id": IMAGE,
        "count": 4,
        "fake": "<fake_token_around_image>",
        "global": "<global-img>",
        "edge": 128,
        "tile": 32,
        "split": False,
    }
    # The encoder's tensors are counted among the model's, and in its fingerprint
    assert any("vision_model" in name for name in manifest.vision.tensors.values())
    Model(out, verify=True)
    tampered = Manifest.load(out)
    tampered.fingerprint = "sha256:" + "0" * 64
    tampered.save(out)
    with pytest.raises(FingerprintMismatch):
        Model(out, verify=True)
    manifest.save(out)


def test_images_are_written_as_the_run_of_tokens_the_model_reads(vlm):
    _, out = vlm
    tokenizer = Tokenizer(out, Manifest.load(out).tokenizer)
    ids = tokenizer.chat([ask("t9", png(1))])
    start = ids.index(FAKE)
    assert ids[start : start + 8] == [FAKE, WHOLE, IMAGE, IMAGE, IMAGE, IMAGE, FAKE, 9][:8]
    assert ids.count(IMAGE) == 4
    assert len(tokenizer.chat([ask("t9", png(1), png(2))])) == len(ids) + 7
    # A plain message still renders: the template takes the parts
    assert tokenizer.chat([{"role": "user", "content": "t9"}])
    with pytest.raises(ValueError, match="image"):
        tokenizer.chat([{"role": "user", "content": "<image>"}])


def test_logits_match_transformers_with_an_image(vlm):
    reference, out = vlm
    model = Model(out)
    tokenizer = Tokenizer(out, Manifest.load(out).tokenizer)
    urls = [png(1)]
    tokens = tokenizer.chat([ask("t9 t10", *urls)])
    session = Session(model)
    session.attach_images([ask("t9 t10", *urls)])
    session.feed(tokens)
    expected = reference_logits(reference, model, tokens, urls)
    assert torch.allclose(session.logits, expected[0, -1], atol=1e-4), (session.logits - expected[0, -1]).abs().max()
    # An image changes what the model says: it is read, not skipped
    other = Session(model)
    other.attach_images([ask("t9 t10", png(2))])
    other.feed(tokenizer.chat([ask("t9 t10", png(2))]))
    assert not torch.allclose(session.logits, other.logits, atol=1e-3)


def test_the_whole_pass_and_the_encoder_alone_match(vlm):
    reference, out = vlm
    model = Model(out)
    urls = [png(3), png(4, size=(31, 9))]
    tokenizer = Tokenizer(out, Manifest.load(out).tokenizer)
    tokens = tokenizer.chat([ask("t9", *urls)])
    rows = torch.cat([model.vision.encode(img(url)) for url in urls])
    got = model.forward(torch.tensor([tokens]), model.init_state(), 0, rows)
    expected = reference_logits(reference, model, tokens, urls)
    assert torch.allclose(got, expected, atol=1e-4), (got - expected).abs().max()
    # The encoder's own rows: what transformers puts in the image tokens' places
    pixels = torch.stack([view for url in urls for view in model.vision.views(img(url))])
    with torch.no_grad():
        features = reference.model.get_image_features(
            pixels[None], torch.ones(1, 2, 32, 32, dtype=torch.long)
        ).pooler_output[0]
    assert torch.allclose(rows, features, atol=1e-5), (rows - features).abs().max()


def test_a_conversation_reads_each_turns_images_as_it_goes(vlm):
    reference, out = vlm
    model = Model(out)
    tokenizer = Tokenizer(out, Manifest.load(out).tokenizer)
    first, second = ask("t9", png(5)), ask("t10 t11", png(6))
    session = Session(model)
    session.feed(next_turn(session, tokenizer, [first]))
    reply = [20, 21, 22]
    session.feed(reply)
    session.messages = [first, {"role": "assistant", "content": "t20 t21 t22"}]
    session.feed(next_turn(session, tokenizer, [second]))
    assert session.pending is None or len(session.pending) == 0
    assert session.tokens.count(IMAGE) == 8
    # What the session read is one pass over those tokens with both pictures
    expected = reference_logits(reference, model, session.tokens, [png(5), png(6)])
    assert torch.allclose(session.logits, expected[0, -1], atol=1e-4), (session.logits - expected[0, -1]).abs().max()


def test_a_saved_session_goes_on_with_the_next_image(vlm, tmp_path):
    reference, out = vlm
    model = Model(out)
    tokenizer = Tokenizer(out, Manifest.load(out).tokenizer)
    first, second = ask("t9", png(5)), ask("t10", png(6))
    session = Session(model)
    session.feed(next_turn(session, tokenizer, [first]))
    session.feed([20, 21])
    session.messages = [first, {"role": "assistant", "content": "t20 t21"}]
    session.save(tmp_path)
    loaded = Session.load(model, tmp_path)
    loaded.feed(next_turn(loaded, tokenizer, [second]))
    expected = reference_logits(reference, model, loaded.tokens, [png(5), png(6)])
    assert torch.allclose(loaded.logits, expected[0, -1], atol=1e-4)


def test_pictures_become_the_normalized_square_the_encoder_reads(vlm):
    _, out = vlm
    vision = Model(out).vision
    solid = io.BytesIO()
    Image.new("RGB", (5, 40), (255, 0, 51)).save(solid, "PNG")
    pixels = vision.views(img(data_url(solid.getvalue())))[0]
    assert pixels.shape == (3, 32, 32)
    assert torch.allclose(pixels[0], torch.full((32, 32), 1.0))
    assert torch.allclose(pixels[1], torch.full((32, 32), -1.0))
    assert torch.allclose(pixels[2], torch.full((32, 32), -0.6), atol=1e-6)
    # Whatever the mode, the same picture in RGB: palette, grey and transparent ones too
    rgb = vision.views(img(png(7)))[0]
    for mode in ("RGBA", "P", "L"):
        converted = vision.views(img(png(7, mode=mode)))[0]
        assert converted.shape == rgb.shape
    assert torch.equal(vision.views(img(png(7, mode="RGBA")))[0], rgb)
    assert torch.equal(vision.views(img(png(7)))[0], rgb), "the same bytes, the same pixels"


def test_what_is_not_a_picture_the_runtime_reads_is_refused(vlm):
    _, out = vlm
    vision = Model(out).vision
    with pytest.raises(ValueError, match="not an image"):
        vision.views(img(data_url(b"this is not a picture")))
    for url, why in [
        ("https://example.com/cat.png", "fetch"),
        ("file:///etc/passwd", "fetch"),
        ("data:image/png,rawbytes", "fetch"),
        ("data:image/png;base64,", "fetch"),
        ("data:image/png;base64,%%%", "base64"),
    ]:
        with pytest.raises(ValueError, match=why):
            picture(url)
    bomb = io.BytesIO()
    Image.new("1", (9000, 9000)).save(bomb, "PNG")
    with pytest.raises(ValueError, match="pixels"):
        vision.views(img(data_url(bomb.getvalue())))


def test_the_api_takes_images_as_data_urls_in_a_users_message():
    url = png(1)
    parts = [{"type": "text", "text": "what is it"}, {"type": "image_url", "image_url": {"url": url}}]
    assert message({"role": "user", "content": parts}) == {
        "role": "user",
        "content": [{"type": "text", "text": "what is it"}, {"type": "image", "url": url}],
    }
    plain = [{"type": "image_url", "image_url": url}]
    assert message({"role": "user", "content": plain})["content"] == [{"type": "image", "url": url}]
    with pytest.raises(ValueError, match="user"):
        message({"role": "assistant", "content": parts})
    with pytest.raises(ValueError, match="data"):
        message({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://x/y.png"}}]})
    with pytest.raises(ValueError, match="text and images"):
        message({"role": "user", "content": [{"type": "input_audio"}]})
    # Text parts alone are joined, as they were
    assert message({"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}) == {
        "role": "user",
        "content": "ab",
    }


def test_a_model_without_an_encoder_says_so(tmp_path_factory):
    torch.manual_seed(0)
    reference = transformers.LlamaForCausalLM(config().text_config)
    checkpoint = tmp_path_factory.mktemp("llama")
    reference.save_pretrained(checkpoint)
    tokenizer_files(checkpoint)
    out = tmp_path_factory.mktemp("text")
    import_checkpoint(checkpoint, out)
    model = Model(out)
    assert model.vision is None
    with pytest.raises(ValueError, match="text only"):
        Session(model).attach_images([ask("t9", png(1))])


def test_a_receipt_of_a_reply_about_an_image_replays(vlm):
    _, out = vlm
    model = Model(out)
    tokenizer = Tokenizer(out, Manifest.load(out).tokenizer)
    messages = [ask("t9 t10", png(8))]
    session = Session(model)
    session.choices = []
    session.attach_images(messages)
    session.feed(tokenizer.chat(messages))
    tokens = []
    for _ in range(4):
        token = session.choose()
        tokens.append(token)
        session.feed([token])
    text = tokenizer.decode(tokens).strip()
    made = receipt.record_messages(session, messages, text, {"max_tokens": 4})
    assert made["prompt"] == "t9 t10"
    verdict, why = receipt.replay_messages(model, tokenizer, made)
    assert verdict == 0, why
    # The same receipt, with another picture, is not the reply that picture gets
    made["messages"] = [ask("t9 t10", png(9))]
    assert receipt.replay_messages(model, tokenizer, made)[0] != 0
    assert base64  # the pictures travel in the receipt as they were given


def test_the_driver_reads_the_pictures_it_is_given(vlm, tmp_path):
    from shem_neural.model import generate

    _, out = vlm
    path = tmp_path / "cat.png"
    path.write_bytes(picture(png(10)))
    args = generate.resolve(generate.options([str(out), "--image", str(path), "--max-tokens", "3", "t9", "t10"]))
    model = Model(args.model)
    tokenizer = Tokenizer(args.model, model.manifest.tokenizer)
    written = io.StringIO()
    assert generate.run(model, tokenizer, args, "t9 t10", written, io.StringIO()) == 0
    # What it says is what the model says to that picture, token by token
    messages = [ask("t9 t10", png(10))]
    session = Session(model)
    session.attach_images(messages)
    session.feed(tokenizer.chat(messages))
    tokens = []
    for _ in range(3):
        tokens.append(session.choose())
        session.feed(tokens[-1:])
    assert written.getvalue().strip() == tokenizer.decode(tokens).strip()
    assert generate.images(args) == [png(10)]


def test_the_server_answers_a_request_with_an_image(vlm):
    """Over HTTP, as OpenAI's API has it: an `image_url` part with a data URL."""
    import urllib.error
    import urllib.request
    from types import SimpleNamespace

    from shem_neural.model import serve

    _, out = vlm
    model = Model(out)
    tokenizer = Tokenizer(out, model.manifest.tokenizer)
    chat = serve.Chat(model, tokenizer, SimpleNamespace(max_tokens=4, stop=[], system=None, subject=[]))
    server = serve.start(chat, serve.listen("127.0.0.1", 0))
    host, port = server.server_address

    def post(content):
        body = {"messages": [{"role": "user", "content": content}], "max_tokens": 4, "shem_receipt": True}
        request = urllib.request.Request(
            f"http://{host}:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    try:
        url = png(11)
        status, answer = post([{"type": "text", "text": "t9"}, {"type": "image_url", "image_url": {"url": url}}])
        assert status == 200, answer
        usage = answer["usage"]
        assert usage["prompt_tokens"] == len(tokenizer.chat([ask("t9", url)])), "the image's tokens are counted"
        # The same question about another picture, over the same API, is another reply to the model
        direct = chat.complete([ask("t9", url)], max_tokens=4)
        assert answer["choices"][0]["message"]["content"] == direct["text"]
        # The receipt carries the picture, so that `shem replay` reads the same one
        assert answer["shem_receipt"]["messages"][-1]["content"][0] == {"type": "text", "text": "t9"}
        assert answer["shem_receipt"]["messages"][-1]["content"][1] == {"type": "image", "url": url}
        status, refused = post([{"type": "image_url", "image_url": {"url": "http://169.254.169.254/latest"}}])
        assert status == 400 and "fetch" in json.dumps(refused)
    finally:
        server.shutdown()
        server.server_close()


def test_pictures_are_resized_as_the_processor_does():
    """The longest edge first made 4 times the encoder's size, and the other
    in proportion, rounded down to an integer and then up to an even one."""
    assert larger((640, 480), 128) == (128, 96)
    assert larger((480, 640), 128) == (96, 128)
    assert larger((100, 100), 128) == (128, 128)
    # Sizes as transformers' own function gives them, for pictures of odd shapes
    for size, expected in [((7, 1000), (1, 128)), ((1000, 7), (128, 1)), ((5, 40), (16, 128)), ((333, 111), (128, 42))]:
        assert larger(size, 128) == expected


SPLIT = {"token": "<image>", "count": 4, "fake": "<f>", "global": "<g>", "edge": 128, "tile": 32, "split": True}


def test_large_pictures_are_cut_in_tiles_and_small_ones_are_not():
    # 128 x 96 is 4 x 3 tiles of 32; the processor's own rounding up gives the rest
    assert tiles((640, 480), SPLIT) == (128, 96, 3, 4)
    assert tiles((480, 640), SPLIT) == (96, 128, 4, 3)
    assert tiles((100, 100), SPLIT) == (128, 128, 4, 4)
    assert tiles((333, 111), SPLIT) == (128, 64, 2, 4), "42 high is rounded up to 64"
    assert tiles((7, 1000), SPLIT) == (32, 128, 4, 1)
    one = SPLIT | {"edge": 32}
    assert tiles((50, 50), one) == (32, 32, 1, 1)
    # One tile is no cutting, and a model told not to cut never does
    url = png(1, size=(50, 50))
    assert grid(img(url), one) == (0, 0)
    assert grid(img(url), SPLIT) == (4, 4)
    assert grid(img(url), SPLIT | {"split": False}) == (0, 0)
    assert grid(img(url, detail="low"), SPLIT) == (0, 0), "low detail is the whole picture in one view"
    assert grid(img(url, detail="high"), SPLIT) == (4, 4)


def test_tiles_are_written_as_the_processor_writes_them():
    from transformers.models.smolvlm.processing_smolvlm import get_image_prompt_string

    for rows, cols in [(0, 0), (1, 2), (3, 4), (4, 1)]:
        expected = get_image_prompt_string(
            rows,
            cols,
            SPLIT["count"],
            fake_token_around_image="<f>",
            image_token="<image>",
            global_image_token="<g>",
        )
        assert written(SPLIT, rows, cols) == expected


def test_a_picture_cut_in_tiles_gives_transformers_logits(tiled):
    reference, out = tiled
    model = Model(out)
    tokenizer = Tokenizer(out, Manifest.load(out).tokenizer)
    urls = [png(12, size=(40, 30)), png(13, size=(25, 25))]
    # 40 x 30 is 4 x 3 tiles and the whole, the square one 4 x 4 and the whole
    messages = [ask("t9", *urls)]
    tokens = tokenizer.chat(messages)
    assert tokens.count(IMAGE) == 4 * (12 + 1) + 4 * (16 + 1) == sum(model.vision.tokens(img(url)) for url in urls)
    assert tokens.count(40) == 2, "<row_1_col_1> is written once for each picture"
    session = Session(model)
    session.attach_images(messages)
    session.feed(tokens)
    expected = reference_logits(reference, model, tokens, urls)
    assert torch.allclose(session.logits, expected[0, -1], atol=1e-4), (session.logits - expected[0, -1]).abs().max()


def test_pictures_cut_in_tiles_are_read_turn_by_turn(tiled):
    reference, out = tiled
    model = Model(out)
    tokenizer = Tokenizer(out, Manifest.load(out).tokenizer)
    first, second = ask("t9", png(5, size=(40, 30))), ask("t10", png(6, size=(20, 60)))
    session = Session(model)
    session.feed(next_turn(session, tokenizer, [first]))
    session.feed([20, 21])
    session.messages = [first, {"role": "assistant", "content": "t20 t21"}]
    session.feed(next_turn(session, tokenizer, [second]))
    assert session.pending is None or len(session.pending) == 0
    expected = reference_logits(reference, model, session.tokens, [png(5, size=(40, 30)), png(6, size=(20, 60))])
    assert torch.allclose(session.logits, expected[0, -1], atol=1e-4)


def test_low_detail_reads_a_large_picture_whole_and_in_far_fewer_tokens(tiled):
    reference, out = tiled
    model = Model(out)
    tokenizer = Tokenizer(out, Manifest.load(out).tokenizer)
    url = png(14, size=(40, 30))
    cut, low = [ask("t9", url)], [{"role": "user", "content": [img(url, detail="low"), {"type": "text", "text": "t9"}]}]
    assert len(tokenizer.chat(cut)) - len(tokenizer.chat(low)) > 4 * 12
    assert tokenizer.chat(low).count(IMAGE) == 4
    session = Session(model)
    session.attach_images(low)
    session.feed(tokenizer.chat(low))
    # One view: what transformers gives for that picture read whole
    pixels = torch.stack(model.vision.views(low[0]["content"][0]))[None]
    assert pixels.shape[1] == 1
    with torch.no_grad():
        expected = reference(
            input_ids=torch.tensor([tokenizer.chat(low)]),
            pixel_values=pixels,
            pixel_attention_mask=torch.ones(1, 1, 32, 32, dtype=torch.long),
        ).logits[0, -1]
    assert torch.allclose(session.logits, expected, atol=1e-4)
    # The API takes it as OpenAI's `detail`
    part = {"type": "image_url", "image_url": {"url": url, "detail": "low"}}
    assert message({"role": "user", "content": [part]})["content"] == [img(url, detail="low")]
    with pytest.raises(ValueError, match="detail"):
        message({"role": "user", "content": [{"type": "image_url", "image_url": {"url": url, "detail": "max"}}]})
