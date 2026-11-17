"""The Hugging Face Spaces' own logic: what they accept, refuse and show, with no network.

`space/` is not a package, so its modules are imported from where they are.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "space"))

import convert  # noqa: E402
import replay  # noqa: E402

DIGEST = "sha256:" + "ab" * 32


def receipt(**changes) -> dict:
    return {
        "format": "shem.receipt/v1",
        "image": f"hf.co/me/model@{DIGEST}",
        "prompt": "Who are you?",
        "reply": "I am Shem.",
        "tokens": [1, 2, 3],
        "margins": [0.5, 0.25, 1.0],
        "runtime": {"precision": "float32"},
    } | changes


def test_a_receipt_of_an_image_on_the_hub_is_read():
    read = replay.read(json.dumps(receipt()))
    assert read["tokens"] == [1, 2, 3]


@pytest.mark.parametrize(
    "text, why",
    [
        ("not json", "not a shem.receipt/v1"),
        (json.dumps(receipt(format="other")), "not a shem.receipt/v1"),
        (json.dumps(receipt(margins=[1.0])), "not a shem.receipt/v1"),
        (json.dumps(receipt(prompt=5)), "not a shem.receipt/v1"),
        (json.dumps(receipt(image="ghcr.io/me/model@" + DIGEST)), "not one in a Hugging Face repo"),
        (json.dumps(receipt(image="hf.co/me/model")), "not one in a Hugging Face repo"),
        (json.dumps(receipt(state="x@" + DIGEST)), "published session"),
        (json.dumps(receipt(tokens=[0] * 600, margins=[1.0] * 600)), "replays up to"),
        (json.dumps(receipt(prompt="   ")), "no prompt"),
    ],
)
def test_receipts_it_will_not_replay_are_refused_with_a_reason(text, why):
    with pytest.raises(replay.Refused, match=why):
        replay.read(text)


class Hub:
    """A stand-in for the Hub's API."""

    def __init__(self, files=None, error=None):
        self.files, self.error = files or {}, error

    def model_info(self, repo, revision=None, files_metadata=False):
        if self.error:
            raise self.error
        return SimpleNamespace(siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in self.files.items()])


def test_an_image_the_hub_does_not_hold_for_anyone_is_refused():
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    def error(cls):
        return cls("x", response=SimpleNamespace(headers={}, request=None, status_code=404))

    replay.admit(receipt(), Hub({"oci-layout": 1, "index.json": 10}))
    with pytest.raises(replay.Refused, match="gated"):
        replay.admit(receipt(), Hub(error=error(GatedRepoError)))
    with pytest.raises(replay.Refused, match="no public repo"):
        replay.admit(receipt(), Hub(error=error(RepositoryNotFoundError)))
    with pytest.raises(replay.Refused, match="holds a checkpoint"):
        replay.admit(receipt(), Hub({"model.safetensors": 10}))
    with pytest.raises(replay.Refused, match="GB"):
        replay.admit(receipt(), Hub({"oci-layout": 1, "blobs/sha256/x": int(replay.MAX_BYTES) + 1}))


def test_the_last_thing_shem_said_is_its_verdict():
    assert replay.last("shem: wrote a receipt\nshem: verified: the model chose the tokens again\n\n") == (
        "verified: the model chose the tokens again"
    )
    assert replay.last("") == ""


def test_a_changed_prompts_reply_is_shown_a_word_at_a_time():
    runs = replay.difference("I am a golem of clay.", "I am a golem of stone.")
    assert runs == [("same", "I am a golem of "), ("removed", "clay."), ("added", "stone.")]
    assert replay.difference("same text", "same text") == [("same", "same text")]
    assert replay.difference("", "new") == [("added", "new")]


def test_the_tokens_of_a_receipt_are_not_shown_if_the_image_is_not_unpacked(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "HOME", tmp_path)
    assert replay.pieces(receipt()) == []


class Checkpoints:
    """A stand-in for the Hub's API as `convert.check` uses it."""

    def __init__(self, tmp_path, architecture="LlamaForCausalLM", counts=None, gated=False):
        self.directory, self.architecture, self.gated = tmp_path, architecture, gated
        self.counts = {"BF16": 135_000_000} if counts is None else counts

    def model_info(self, source, files_metadata=False):
        card = SimpleNamespace(to_dict=lambda: {"license": "apache-2.0"})
        return SimpleNamespace(id=source, sha="c" * 40, card_data=card, gated=self.gated)

    def hf_hub_download(self, repo, name, revision=None):
        path = self.directory / "config.json"
        path.write_text(json.dumps({"architectures": [self.architecture]}))
        return str(path)

    def get_safetensors_metadata(self, repo, revision=None):
        if not self.counts:
            raise ValueError("none")
        return SimpleNamespace(parameter_count=self.counts)


def test_a_checkpoint_the_space_can_convert_is_described_before_it_is_downloaded(tmp_path):
    found = convert.check("me/model", Checkpoints(tmp_path))
    assert (found.repo, found.commit, found.architecture) == ("me/model", "c" * 40, "LlamaForCausalLM")
    assert found.parameters == 135_000_000 and found.size == 270_000_000 and found.license == "apache-2.0"
    assert found.verifiable


@pytest.mark.parametrize(
    "api, why",
    [
        (lambda p: Checkpoints(p, architecture="NoSuchForCausalLM"), "NoSuchForCausalLM; shem imports"),
        (lambda p: Checkpoints(p, counts={}), "no safetensors"),
        (lambda p: Checkpoints(p, counts={"F32": int(convert.MAX_BYTES)}), "this Space takes up to"),
    ],
)
def test_checkpoints_it_cannot_convert_are_refused_before_a_download(tmp_path, api, why):
    with pytest.raises(convert.Refused, match=why):
        convert.check("me/model", api(tmp_path))


def test_the_shemfile_pins_the_commit_and_primes_a_system_prompt():
    found = convert.Checkpoint("me/Model", "c" * 40, "LlamaForCausalLM", 1000, 2000)
    assert convert.shemfile(found, None) == f"FROM hf.co/me/Model:{'c' * 40}\n"
    primed = convert.shemfile(found, "  You are a golem.  ")
    assert primed.splitlines()[1:] == [
        'ENTRYPOINT ["shem:generate", "--system", "You are a golem."]',
        "WARM --until=prompt",
    ]
    big = convert.Checkpoint("me/Big", "c" * 40, "LlamaForCausalLM", int(convert.MAX_PRIMED) + 1, 2000)
    with pytest.raises(convert.Refused, match="Priming"):
        convert.refuse(big, "a prompt")
    convert.refuse(big, "")  # no prompt, nothing to run


def test_the_target_repo_is_the_users_own_in_the_names_oci_allows():
    assert convert.target("jane", "Org/My_Model-7B Instruct") == "jane/my_model-7b-instruct-shem"
