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
import functools
import hashlib
import io
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

#: Pictures larger than this many pixels are refused before they are decoded
MAX_PIXELS = 1 << 26
#: Views of a picture the encoder reads in one pass
VIEWS_AT_ONCE = 4
#: Encoded pictures held, to read one that comes back in a conversation again
CACHED = 16
FAMILIES = ("smolvlm", "llava")
ACTIVATIONS = {
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
    "silu": F.silu,
    "quick_gelu": lambda x: x * torch.sigmoid(1.702 * x),
}


def image_parts(messages: list[dict]) -> list[dict]:
    """The pictures in `messages`, as their parts (`url`, and `detail` if it
    has one), in the order they are read."""
    return [
        part
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


@functools.lru_cache(maxsize=32)
def dimensions(url: str) -> tuple[int, int]:
    """The width and height of the picture a `data:` URL holds, read from its header."""
    try:
        from PIL import Image
    except ImportError:
        raise ValueError("reading images needs Pillow: pip install 'shem-neural[images]'") from None
    try:
        with Image.open(io.BytesIO(picture(url))) as image:
            if image.width * image.height > MAX_PIXELS:
                raise ValueError(f"the image is {image.width}x{image.height}, past {MAX_PIXELS} pixels")
            return image.size
    except OSError as error:  # Pillow's UnidentifiedImageError and the decoders' failures
        raise ValueError(f"not an image shem can read: {error}") from None


def larger(size: tuple[int, int], edge: int) -> tuple[int, int]:
    """The size a picture of `size` (width, height) is first resized to: its
    longest edge `edge` long and the other in proportion, rounded down and
    then up to an even number, as the processor does."""
    width, height = size
    ratio = width / height
    if width >= height:
        width = edge
        height = int(width / ratio)
        height += height % 2
    else:
        height = edge
        width = int(height * ratio)
        width += width % 2
    return max(width, 1), max(height, 1)


def tiles(size: tuple[int, int], image: dict) -> tuple[int, int, int, int]:
    """How a picture of `size` is cut into tiles, if `image` (the tokenizer's
    section) says it is: the size it is resized to, a multiple of the tile's
    on each side (width, height), and the rows and columns of tiles that has.
    One tile is no cutting: the picture is its own view."""
    width, height = larger(size, image["edge"])
    ratio, tile = width / height, image["tile"]
    if width >= height:
        width = math.ceil(width / tile) * tile
        height = math.ceil(int(width / ratio) / tile) * tile
    else:
        height = math.ceil(height / tile) * tile
        width = math.ceil(int(height * ratio) / tile) * tile
    return width, height, height // tile, width // tile


def grid(part: dict, image: dict) -> tuple[int, int]:
    """The rows and columns of tiles the picture of `part` is read as, or
    (0, 0) when it is read whole, in one view: when the model is not one
    that cuts pictures, or the part's `detail` is `low`, or it is small."""
    if not image.get("split") or part.get("detail") == "low":
        return 0, 0
    _, _, rows, cols = tiles(dimensions(part["url"]), image)
    return (rows, cols) if rows * cols > 1 else (0, 0)


def written(image: dict, rows: int, cols: int) -> str:
    """The run of tokens that stands for one picture, laid out in `rows` by
    `cols` tiles read each by its own tokens (as the processor writes them)
    and the whole picture after them, or in one view with (0, 0)."""
    token = image["token"]
    view = token * image["count"]
    if "fake" not in image:
        return view  # LLaVA writes the picture's tokens and nothing around them
    fake, whole = image["fake"], image["global"]
    if not rows:
        return f"{fake}{whole}{view}{fake}"
    lines = ("".join(f"{fake}<row_{r + 1}_col_{c + 1}>{view}" for c in range(cols)) for r in range(rows))
    return "\n".join([*lines, ""]) + f"\n{fake}{whole}{view}{fake}"


class Vision:
    """The encoder of a vision-language model, from its manifest's `vision`
    section `params` and its `tensors`, loaded as the language model is."""

    def __init__(
        self, params: dict, tensors: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device, image: dict
    ):
        if params["family"] not in FAMILIES:
            raise ValueError(f"no vision encoder for {params['family']!r}: shem reads {list(FAMILIES)}")
        if params["act"] not in ACTIVATIONS:
            raise ValueError(f"no activation {params['act']!r}: shem has {sorted(ACTIVATIONS)}")
        self.params, self.t, self.dtype, self.device = params, tensors, dtype, device
        self.size, self.patch = params["image_size"], params["patch_size"]
        self.image_token_id = params["image_token_id"]
        #: The tokenizer's section: how a picture is laid out in tiles and written
        self.image = dict(image)
        self.mean = torch.tensor(params["mean"], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(params["std"], dtype=torch.float32).view(3, 1, 1)
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def opened(self, part: dict):
        """The picture of `part`, decoded, in RGB: a Pillow image."""
        try:
            from PIL import Image
        except ImportError:
            raise ValueError("reading images needs Pillow: pip install 'shem-neural[images]'") from None
        try:
            with Image.open(io.BytesIO(picture(part["url"]))) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError(f"the image is {image.width}x{image.height}, past {MAX_PIXELS} pixels")
                return image.convert("RGB")
        except OSError as error:  # Pillow's UnidentifiedImageError and the decoders' failures
            raise ValueError(f"not an image shem can read: {error}") from None

    def normalized(self, views) -> list[torch.Tensor]:
        """Pillow images as (3, height, width) floats: scaled to [0, 1] and normalized."""
        arrays = [torch.from_numpy(np.asarray(view, dtype=np.float32).copy()).permute(2, 0, 1) for view in views]
        return [(array * (1 / 255) - self.mean) / self.std for array in arrays]

    def views(self, part: dict) -> list[torch.Tensor]:
        """The picture of `part` as the encoder reads it: a (3, size, size)
        tensor for each view, normalized. Each tile of a picture cut in tiles
        is a view, row by row, and the whole picture, resized to the square,
        is the last. A picture not cut has that one."""
        from PIL import Image

        lanczos, side = Image.Resampling.LANCZOS, self.size
        # As the family's processor does: the longest edge made `edge` long first
        image = self.opened(part)
        whole = image.resize(larger(image.size, self.image["edge"]), lanczos)
        rows, cols = grid(part, self.image)
        if rows:
            width, height, _, _ = tiles(dimensions(part["url"]), self.image)
            whole = whole.resize((width, height), lanczos)
            crops = [
                whole.crop((c * side, r * side, (c + 1) * side, (r + 1) * side))
                for r in range(rows)
                for c in range(cols)
            ]
            views = [*crops, whole.resize((side, side), lanczos)]
        else:
            views = [whole.resize((side, side), lanczos)]
        return self.normalized(views)

    def tokens(self, part: dict) -> int:
        """How many image tokens the picture of `part` is written as."""
        rows, cols = grid(part, self.image)
        return self.image["count"] * (rows * cols + 1 if rows else 1)

    def encode(self, part: dict) -> torch.Tensor:
        """The rows the picture of `part` is read as: (tokens, hidden size)."""
        key = hashlib.sha256(f"{part.get('detail')}\0{part['url']}".encode()).hexdigest()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        # A few views at a time: their attention over 1,024 patches each is megabytes a head
        rows = torch.cat([self.rows(chunk) for chunk in torch.stack(self.views(part)).split(VIEWS_AT_ONCE)])
        self.cache[key] = rows
        if len(self.cache) > CACHED:
            self.cache.popitem(last=False)
        return rows

    @torch.no_grad()
    def rows(self, pixels: torch.Tensor) -> torch.Tensor:
        """The encoder and the connector over `pixels`, a view each
        (views, 3, size, size): `count` rows for each, one view's after
        another's."""
        p, t = self.params, self.t
        x = pixels.to(self.device, self.dtype)
        patches = F.conv2d(x, t["patch.weight"], t["patch.bias"], stride=self.patch)
        h = patches.flatten(2).transpose(1, 2) + t["position.weight"][self.positions(patches.shape[2:])]
        h = self.layers(h, p["layers"])
        h = F.layer_norm(h, h.shape[-1:], t["post_norm.weight"], t["post_norm.bias"], p["eps"])
        return F.linear(self.shuffle(h), t["connector.weight"]).flatten(0, 1)

    def layers(self, h: torch.Tensor, count: int) -> torch.Tensor:
        """The first `count` layers of the encoder over `h` (views, patches, width):
        each normalizes, attends to every patch and adds that, then normalizes and adds a feed-forward."""
        p, t = self.params, self.t
        eps, heads = p["eps"], p["heads"]
        activation = ACTIVATIONS[p["act"]]
        for i in range(count):

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
        return h

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


class Llava(Vision):
    """LLaVA's encoder: a CLIP or SigLIP vision transformer over one picture's
    patches, as the checkpoint says, and a two-layer projector to the language
    model's width. The picture is resized, and for CLIP's processors cropped, to
    one square; the layer of the encoder that is read, and whether its class
    token goes, are the model's config's."""

    def views(self, part: dict) -> list[torch.Tensor]:
        from PIL import Image

        image, p = self.opened(part), self.params
        resample = Image.Resampling(p["resample"])
        if p["resize"] == "shortest_edge":
            # The short edge made `size` long and the long one in proportion, cut down to a whole number
            width, height = image.size
            short, long = (width, height) if width <= height else (height, width)
            new_short, new_long = self.size, int(self.size * long / short)
            image = image.resize((new_short, new_long) if width <= height else (new_long, new_short), resample)
            # Then the middle of it
            left, top = (image.width - self.size) // 2, (image.height - self.size) // 2
            image = image.crop((left, top, left + self.size, top + self.size))
        else:
            image = image.resize((self.size, self.size), resample)
        return self.normalized([image])

    @torch.no_grad()
    def rows(self, pixels: torch.Tensor) -> torch.Tensor:
        """The encoder's chosen layer over `pixels` (views, 3, size, size), and
        the projector: `count` rows for each view."""
        p, t = self.params, self.t
        x = pixels.to(self.device, self.dtype)
        patches = F.conv2d(x, t["patch.weight"], t.get("patch.bias"), stride=self.patch)
        h = patches.flatten(2).transpose(1, 2)
        if "cls" in t:
            h = torch.cat([t["cls"].expand(h.shape[0], 1, -1), h], dim=1)
        h = h + t["position.weight"][: h.shape[1]]
        if "pre_norm.weight" in t:
            h = F.layer_norm(h, h.shape[-1:], t["pre_norm.weight"], t["pre_norm.bias"], p["eps"])
        h = self.layers(h, p["layers"])
        if p["select"] == "default":
            h = h[:, 1:]
        h = F.linear(h, t["proj1.weight"], t.get("proj1.bias"))
        h = F.linear(ACTIVATIONS[p["projector_act"]](h), t["proj2.weight"], t.get("proj2.bias"))
        return h.flatten(0, 1)


def make(
    params: dict, tensors: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device, image: dict
) -> Vision:
    """The encoder the manifest's `vision` section describes."""
    if params["family"] not in FAMILIES:
        raise ValueError(f"no vision encoder for {params['family']!r}: shem reads {list(FAMILIES)}")
    return (Llava if params["family"] == "llava" else Vision)(params, tensors, dtype, device, image)
