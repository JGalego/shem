"""Sessions saved by an earlier shem, resumed by this one.

`sessions/` holds a tiny model with a block of every type and a session it
saved, both committed: the session is what an earlier runtime wrote. This
runtime must resume it as if it had read the session's tokens itself. A
change to what a block computes or keeps as state, without a bump of its
`VERSION`, fails here, as an old session would continue wrong. With the bump,
the saved session is stale, and the test says to save it again:

    uv run --project neural python neural/tests/test_sessions.py
"""

import io
import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from shem_neural.model.blocks import BLOCKS, Model
from shem_neural.model.generate import Session, staleness
from shem_neural.model.manifest import INDEX, BlockSpec, Manifest

SAVED = Path(__file__).parent / "sessions"
#: The session's tokens, then the tokens the test continues it with
TOKENS = [3, 17, 42, 8, 25, 61, 9, 33, 50, 12, 4, 27]
PROBE = [5, 40, 18]
#: The tiny families whose blocks the model takes, by their importers' tests,
#: and the types of block it takes from each, the first of each type, or all:
#: Qwen3.5's Gated DeltaNet, MLPs and attention on part of each head, then
#: attention on all of it with norms on queries and keys, experts, and a selective scan
DONORS = {"qwen3.5": None, "qwen3-moe": ["attention", "moe"], "mamba": ["mamba"]}
#: The blocks whose state the session keeps
STATEFUL = ["attention", "gated_deltanet", "mamba"]


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether these logits are one model's for the same tokens, read in
    passes of other lengths: equal up to the order of floating-point sums."""
    return torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def cold(model: Model, tokens: list[int]) -> torch.Tensor:
    session = Session(model)
    session.feed(tokens)
    return session.logits


@pytest.fixture(scope="module")
def model():
    return Model(SAVED / "model")


def test_the_model_has_a_block_of_every_type(model):
    assert sorted(model.versions) == sorted(BLOCKS)


def test_saved_sessions_resume_as_a_cold_start(model):
    meta = json.loads((SAVED / "session" / "session.json").read_text())
    if stale := staleness(model, meta):
        pytest.fail(
            f"the saved session was saved {stale}: if that is a block's new VERSION, "
            "save it again with `uv run --project neural python neural/tests/test_sessions.py`"
        )
    err = io.StringIO()
    resumed = Session.load(model, SAVED / "session", err)
    assert err.getvalue() == "", "a session of this runtime's blocks is not read again"
    assert resumed.tokens == TOKENS
    resumed.feed(PROBE)
    want = cold(model, TOKENS + PROBE)
    assert same(resumed.logits, want), (
        f"a session saved before this change resumes otherwise than a cold start, by "
        f"{(resumed.logits - want).abs().max():.2e}: bump the VERSION of the type of block that changed"
    )


@pytest.mark.parametrize("kind", STATEFUL)
def test_state_read_otherwise_resumes_otherwise(model, kind):
    """The test above would notice a block reading its state otherwise: here,
    as the opposite of what it was."""
    resumed = Session.load(model, SAVED / "session")
    for spec, state in zip(model.manifest.blocks, resumed.state, strict=True):
        if spec.type == kind:
            for name in state:
                state[name] = -state[name]
    resumed.feed(PROBE)
    assert not same(resumed.logits, cold(model, TOKENS + PROBE))


def stored(tmp_path: Path, **changes) -> Path:
    """A copy of the saved session, with `changes` to its `session.json`:
    a value of None removes the field."""
    copy = tmp_path / "session"
    shutil.copytree(SAVED / "session", copy)
    meta = json.loads((copy / "session.json").read_text())
    for key, value in changes.items():
        if value is None:
            meta.pop(key)
        else:
            meta[key] = value
    (copy / "session.json").write_text(json.dumps(meta))
    return copy


@pytest.mark.parametrize(
    "changes, said",
    [
        (dict(blocks={**dict.fromkeys(BLOCKS, 1), "attention": 0}), "by another version of shem's attention blocks"),
        (dict(precision="bfloat16"), "in bfloat16, and this model runs in float32"),
    ],
)
def test_stale_sessions_are_read_again_from_their_tokens(model, tmp_path, changes, said):
    err = io.StringIO()
    resumed = Session.load(model, stored(tmp_path, **changes), err)
    assert err.getvalue() == f"shem: the session was saved {said}: reading its {len(TOKENS)} tokens again\n"
    assert resumed.tokens == TOKENS
    assert torch.equal(resumed.logits, cold(model, TOKENS))


def test_sessions_saved_before_versions_are_of_the_first(model, tmp_path, monkeypatch):
    legacy, err = stored(tmp_path, blocks=None, precision=None), io.StringIO()
    resumed = Session.load(model, legacy, err)
    assert err.getvalue() == ""
    assert resumed.logits is None, "its state, not its tokens read again"
    monkeypatch.setattr(BLOCKS["mamba"], "VERSION", 2)
    Session.load(model, legacy, err)
    assert "mamba blocks" in err.getvalue()


def test_sessions_record_their_precision_and_versions(model, tmp_path):
    session = Session(model)
    session.feed(TOKENS[:3])
    meta = json.loads(session.to_files()["session.json"])
    assert meta["precision"] == "float32"
    assert meta["blocks"] == {kind: cls.VERSION for kind, cls in BLOCKS.items()}


def save(out: Path) -> None:
    """Write the tiny model, with blocks of every type from the importers'
    tiny random checkpoints, and a session of it, into `out`."""
    import tempfile

    from test_importers import CONFIGS, SMALL, tokenizer_files
    from transformers import AutoModelForCausalLM

    from shem_neural.model.importers import import_checkpoint

    torch.manual_seed(0)
    blocks, tensors = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for family, kinds in DONORS.items():
            reference = AutoModelForCausalLM.from_config(CONFIGS[family]())
            with torch.no_grad():
                for parameter in reference.parameters():
                    parameter.add_(torch.randn_like(parameter) * 0.1)
            checkpoint, imported = Path(tmp) / family, Path(tmp) / f"{family}-imported"
            reference.save_pretrained(checkpoint)
            tokenizer_files(checkpoint, SMALL["vocab_size"])
            import_checkpoint(checkpoint, imported)
            donor = Model(imported)
            if kinds is None:
                # The family whose embedding, head and tokenizer the model has too
                first, base = donor, donor.manifest
                (out / "model").mkdir(parents=True)
                for name in ("file", "chat_template"):
                    shutil.copy(imported / base.tokenizer[name], out / "model")
            wanted = set(kinds or [])
            for spec, block in zip(donor.manifest.blocks, donor.blocks, strict=True):
                if kinds is None or spec.type in wanted:
                    wanted.discard(spec.type)
                    names = {local: f"blocks.{len(blocks)}.{local}" for local in block.t}
                    tensors |= {names[local]: tensor for local, tensor in block.t.items()}
                    blocks.append(BlockSpec(spec.type, spec.params, names))
    tensors["embed.weight"] = first.embed
    tensors["head.norm"] = first.head_norm
    head = {"norm": "head.norm"}
    if not base.head.params.get("tied"):
        tensors["head.weight"] = first.head_weight
        head["weight"] = "head.weight"
    tensors = {name: tensor.contiguous() for name, tensor in tensors.items()}
    save_file(tensors, str(out / "model" / "model.safetensors"), metadata={"format": "pt"})
    weight_map = dict.fromkeys(tensors, "model.safetensors")
    (out / "model" / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1) + "\n")
    Manifest(
        hidden_size=base.hidden_size,
        tokenizer=base.tokenizer,
        embed=BlockSpec("embedding", base.embed.params, {"weight": "embed.weight"}),
        blocks=blocks,
        head=BlockSpec("head", base.head.params, head),
        source={"architecture": "tiny random Qwen3.5, Qwen3-MoE and Mamba blocks, for test_sessions"},
        parameters=sum(tensor.numel() for tensor in tensors.values()),
        precision=["float32"],
    ).save(out / "model")

    session = Session(Model(out / "model"))
    session.feed(TOKENS)
    session.save(out / "session")


if __name__ == "__main__":
    shutil.rmtree(SAVED, ignore_errors=True)
    save(SAVED)
    print(f"saved {SAVED}")
