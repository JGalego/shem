"""Vision encoders: pictures made into the rows a language model reads where
its text has image tokens.

A vision-language model reads an image as a run of embeddings that stand
where the chat template put image tokens. Here they come in three steps,
which the manifest's `vision` section and its tokenizer's `image` section
describe:

- the picture is decoded, resized as the family's processor does (its
  longest edge made long, then the encoder's square) and normalized
  (`Vision.pixels`, with Pillow, the `images` extra);
- the encoder, a transformer over the picture's patches, and the connector
  that folds neighbouring patches into one and projects them to the language
  model's width, make `count` rows (`Vision.encode`);
- the tokenizer writes the run of image tokens (`count` of them, between the
  tokens the family wraps an image in), and the session puts the rows in
  place of the tokens' own embeddings as it reads them.

SmolVLM is the family shem reads: a SigLIP-style encoder, its patches' position
embeddings chosen as for pictures of any shape, and a pixel-shuffle connector.
Its processor can also cut a large picture into tiles that the model reads
beside a smaller view of the whole; shem reads the whole picture, resized to
the encoder's square, and says so in the docs.

Images reach a model as `data:` URLs in a message's content, as `{"type":
"image", "url": ...}` parts: the runtime never fetches a link, which would
let a request make it reach into the network around it.
"""

import base64
import binascii
import hashlib
import io
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

#: Pictures larger than this many pixels are refused before they are decoded
MAX_PIXELS = 1 << 26
#: Encoded pictures held, to read one that comes back in a conversation again
CACHED = 16
FAMILIES = ("smolvlm",)
ACTIVATIONS = {
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
    "silu": F.silu,
    "quick_gelu": lambda x: x * torch.sigmoid(1.702 * x),
}


def image_urls(messages: list[dict]) -> list[str]:
    """The pictures in `messages`, as their URLs, in the order they are read."""
    return [
        part["url"]
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image"
    ]


def plain(content) -> str:
    """The text of a message's `content`, which is a string or a list of parts."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if part.get("type") == "text")


def data_url(data: bytes, media_type: str = "image/png") -> str:
    """The `data:` URL of a picture's bytes, how messages hold them."""
    return f"data:{media_type};base64,{base64.b64encode(data).decode()}"


def picture(url: str) -> bytes:
    """The bytes of the picture a `data:` URL holds. Any other URL is
    refused: the runtime does not fetch what a request points at."""
    head, _, body = url.partition(",")
    if not head.startswith("data:") or not head.endswith(";base64") or not body:
        raise ValueError("an image must be a base64 `data:` URL: shem does not fetch links")
    try:
        return base64.b64decode(body, validate=True)
    except binascii.Error as error:
        raise ValueError(f"an image's data is not base64: {error}") from None


class Vision:
    """The encoder of a vision-language model, from its manifest's `vision`
    section `params` and its `tensors`, loaded as the language model is."""

    def __init__(self, params: dict, tensors: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device):
        if params["family"] not in FAMILIES:
            raise ValueError(f"no vision encoder for {params['family']!r}: shem reads {list(FAMILIES)}")
        if params["act"] not in ACTIVATIONS:
            raise ValueError(f"no activation {params['act']!r}: shem has {sorted(ACTIVATIONS)}")
        self.params, self.t, self.dtype, self.device = params, tensors, dtype, device
        self.size, self.patch = params["image_size"], params["patch_size"]
        self.image_token_id = params["image_token_id"]
        self.count = params["count"]
        self.edge = params.get("edge", 4 * self.size)
        self.mean = torch.tensor(params["mean"], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(params["std"], dtype=torch.float32).view(3, 1, 1)
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def pixels(self, data: bytes) -> torch.Tensor:
        """The picture in `data` as the encoder reads it: (3, size, size)
        floats, the picture in RGB resized to the square with Lanczos
        resampling, scaled to [0, 1] and normalized."""
        try:
            from PIL import Image
        except ImportError:
            raise ValueError("reading images needs Pillow: pip install 'shem-neural[images]'") from None
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError(f"the image is {image.width}x{image.height}, past {MAX_PIXELS} pixels")
                lanczos = Image.Resampling.LANCZOS
                # As the family's processor does: the longest edge made `edge` long, then the square
                resized = (
                    image.convert("RGB").resize(self.larger(image.size), lanczos).resize((self.size,) * 2, lanczos)
                )
        except OSError as error:  # Pillow's UnidentifiedImageError and the decoders' failures
            raise ValueError(f"not an image shem can read: {error}") from None
        array = torch.from_numpy(np.asarray(resized, dtype=np.float32).copy()).permute(2, 0, 1)
        return (array * (1 / 255) - self.mean) / self.std

    def larger(self, size: tuple[int, int]) -> tuple[int, int]:
        """The size a picture of `size` (width, height) is first resized to:
        its longest edge `edge` long and the other in proportion, rounded
        down and then up to an even number, as the processor does."""
        width, height = size
        ratio = width / height
        if width >= height:
            width = self.edge
            height = int(width / ratio)
            height += height % 2
        else:
            height = self.edge
            width = int(height * ratio)
            width += width % 2
        return max(width, 1), max(height, 1)

    def encode(self, url: str) -> torch.Tensor:
        """The rows the picture at `url` is read as: (count, hidden size)."""
        key = hashlib.sha256(url.encode()).hexdigest()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        rows = self.rows(self.pixels(picture(url)))
        self.cache[key] = rows
        if len(self.cache) > CACHED:
            self.cache.popitem(last=False)
        return rows

    @torch.no_grad()
    def rows(self, pixels: torch.Tensor) -> torch.Tensor:
        """The encoder and the connector over `pixels` (3, size, size)."""
        p, t = self.params, self.t
        x = pixels.to(self.device, self.dtype)[None]
        patches = F.conv2d(x, t["patch.weight"], t["patch.bias"], stride=self.patch)
        h = patches.flatten(2).transpose(1, 2) + t["position.weight"][self.positions(patches.shape[2:])]
        eps, heads = p["eps"], p["heads"]
        activation = ACTIVATIONS[p["act"]]
        for i in range(p["layers"]):

            def w(name, i=i):
                return t[f"{i}.{name}"]

            norm = F.layer_norm(h, h.shape[-1:], w("norm1.weight"), w("norm1.bias"), eps)
            shape = (*norm.shape[:-1], heads, -1)
            q, k, v = (
                F.linear(norm, w(f"{n}.weight"), w(f"{n}.bias")).view(shape).transpose(1, 2) for n in ("q", "k", "v")
            )
            scores = torch.matmul(q, k.transpose(-1, -2)) * q.shape[-1] ** -0.5
            attended = torch.matmul(F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype), v)
            attended = attended.transpose(1, 2).reshape(norm.shape)
            h = h + F.linear(attended, w("out.weight"), w("out.bias"))
            norm = F.layer_norm(h, h.shape[-1:], w("norm2.weight"), w("norm2.bias"), eps)
            h = h + F.linear(activation(F.linear(norm, w("fc1.weight"), w("fc1.bias"))), w("fc2.weight"), w("fc2.bias"))
        h = F.layer_norm(h, h.shape[-1:], t["post_norm.weight"], t["post_norm.bias"], eps)
        return F.linear(self.shuffle(h), t["connector.weight"])[0]

    def positions(self, grid: tuple[int, ...]) -> torch.Tensor:
        """Which of the encoder's learned positions each patch of a picture
        takes: the picture's rows and columns, each placed on the learned
        grid's by the fraction of the way across it they sit, as the
        encoder was trained to read pictures of any shape."""
        side = self.size // self.patch
        boundaries = torch.arange(1 / side, 1.0, 1 / side)
        coords = [torch.clamp(torch.arange(n, dtype=torch.float32) * (1.0 / n), max=1.0 - 1e-6) for n in grid]
        rows, cols = (torch.bucketize(c, boundaries, right=True) for c in coords)
        return (rows[:, None] * side + cols[None, :]).reshape(-1).to(self.device)

    def shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """Neighbouring patches folded into one, scale by scale along each
        side: (1, patches, width) to (1, patches / scale**2, width * scale**2)."""
        scale = self.params["scale"]
        batch, seq, width = x.shape
        side = int(seq**0.5)
        x = x.view(batch, side, side // scale, width * scale).permute(0, 2, 1, 3)
        x = x.reshape(batch, side // scale, side // scale, width * scale**2).permute(0, 2, 1, 3)
        return x.reshape(batch, seq // scale**2, width * scale**2)
