"""Language-model import, execution and session state.

`manifest` defines the on-disk layout, `blocks` implements its block types,
`importers` maps checkpoints into it, and `generate` drives sessions using
the checkpoint's `tokenizer` and chat template.
"""
