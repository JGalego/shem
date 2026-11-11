"""Models as images: a language model as a stack of typed blocks.

A model is a residual stream: an embedding turns tokens into vectors, each
block reads the stream and adds to it, and a head turns the last vector into
next-token logits. Nothing here is specific to one model family:

- `manifest` describes a model as files: `shem.json`, which lists the blocks
  in order with their types and parameters, and one weights file per block,
  which `shem build` puts in its own image layer;
- `blocks` implements the block types, registered by name: `attention`,
  `gated_deltanet`, `mlp`, and the `embedding` and `head` at either end;
- `importers` turns a Hugging Face checkpoint into that layout, one importer
  per architecture;
- `tokenizer` encodes text and renders the model's chat template;
- `generate` is the `shem:generate` driver: it loads a model from its files
  and runs a session.

A new model family needs an importer, and a new kind of layer a block type.
"""
