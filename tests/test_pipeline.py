import copy
import json
import pytest

from format_conversation_dataset.pipeline import response_examples, verify_splits, prepare, read_source
from format_conversation_dataset.tokenize import encode_example
from format_conversation_dataset.parsers import parse_text, parse_whisperx
from format_conversation_dataset.format_transcription import process_speakers


class TinyTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize, return_dict, enable_thinking, add_generation_prompt=False):
        if add_generation_prompt:
            return list(json.dumps(messages).encode()) + list(b"<assistant>")
        return list(json.dumps(messages[:-1]).encode()) + list(b"<assistant>") + list(messages[-1]["content"].encode()) + [0]

    def decode(self, tokens):
        return bytes(t for t in tokens if t).decode()


def example_rows(split="train", suffix=""):
    # Original test dialogue, deliberately unrelated to an actual conversation.
    turns = list(parse_text("SPEAKER_01: Which crate?\nSPEAKER_00: The blue one." + suffix +
                            "\nSPEAKER_00: It has a brass handle.\nSPEAKER_01: What is inside?\nSPEAKER_00: A folded map." + suffix))
    kwargs = dict(group=split, source_sha256="source-" + split, split=split,
                  assistant_speakers=["SPEAKER_00"], system="Answer in context.")
    _, reviews, _ = response_examples(turns, **kwargs)
    for r in reviews:
        r.update(verdict="keep", reviewer="test-reviewer")
    return response_examples(turns, reviews=reviews, **kwargs)[0], turns, reviews, kwargs


def test_contiguous_response_and_preceding_exchange_are_preserved():
    rows, _, _, _ = example_rows()
    assert rows[0]["completion"][0]["content"] == "The blue one. It has a brass handle."
    assert rows[0]["provenance"]["target_turns"] == [2, 3]
    assert rows[1]["provenance"]["protected_context_turns"] == [2, 3, 4]
    assert all(t < 5 for t in rows[1]["provenance"]["context_turns"])


def test_review_is_required_and_is_bound_to_text_and_boundaries():
    _, turns, reviews, kwargs = example_rows()
    assert response_examples(turns, **kwargs)[0] == []
    turns[1]["text"] = "Changed response"
    with pytest.raises(ValueError, match="stale"):
        response_examples(turns, reviews=reviews, **kwargs)
    reviews[0]["target_turns"] = [99]
    with pytest.raises(ValueError, match="boundaries"):
        response_examples(turns, reviews=reviews, **kwargs)


def test_model_review_requires_opt_in_and_retains_origin():
    _, turns, reviews, kwargs = example_rows()
    for r in reviews:
        r["origin"] = "model"
    assert response_examples(turns, reviews=reviews, **kwargs)[0] == []
    rows = response_examples(turns, reviews=reviews, allow_screened=True, **kwargs)[0]
    assert all(r["provenance"]["review_level"] == "model-screened" for r in rows)


def test_unknown_speakers_are_preserved_but_not_silently_trained():
    turns = list(parse_text("unlabeled words\nSPEAKER_00: response"))
    assert turns[0]["text"] == "unlabeled words"
    _, _, skipped = response_examples(turns, group="g", source_sha256="s", split="train",
                                      assistant_speakers=["SPEAKER_00"], system="")
    assert skipped == {"unresolved_context_speaker": 1}


def test_word_speaker_boundaries_and_timing_survive_import():
    turns = list(parse_whisperx({"segments": [{"text": "Yes. Why?", "words": [
        {"word": "Yes.", "speaker": "SPEAKER_01", "start": 0, "end": .4},
        {"word": "Why?", "speaker": "SPEAKER_00", "start": .5, "end": .8}]}]}))
    assert [t["speaker"] for t in turns] == ["SPEAKER_01", "SPEAKER_00"]
    assert turns[1]["start"] == .5 and turns[0]["end"] == .4


@pytest.mark.parametrize("field", ["group", "source_sha256", "source_identity"])
def test_provenance_prevents_cross_split_leakage(field):
    train = example_rows()[0]
    val = example_rows("validation", " Different.")[0]
    for row in train + val:
        row["provenance"][field] = "shared"
    with pytest.raises(ValueError, match="crosses"):
        verify_splits({"train": train, "validation": val})


def test_exact_duplicate_response_across_splits_is_rejected():
    with pytest.raises(ValueError, match="crosses"):
        verify_splits({"train": example_rows()[0], "validation": example_rows("validation")[0]})


def test_source_variants_and_future_context_are_rejected():
    rows = example_rows()[0]
    rows[1]["provenance"]["source_sha256"] = "alternate-transcription"
    with pytest.raises(ValueError, match="variant"):
        verify_splits({"train": rows})
    rows = example_rows()[0]
    rows[0]["provenance"]["context_turns"].append(2)
    with pytest.raises(ValueError, match="future"):
        verify_splits({"train": rows})


def test_completion_mask_includes_eos_and_never_the_prompt():
    row = example_rows()[0][0]
    tokens, audit = encode_example(row, TinyTokenizer(), 4096)
    assert tokens["labels"][:audit["prefix_tokens"]] == [-100] * audit["prefix_tokens"]
    assert tokens["labels"][audit["prefix_tokens"]:] == list(row["completion"][0]["content"].encode()) + [0]
    assert audit["decoded_completion"] == row["completion"][0]["content"]


def test_overlong_required_exchange_and_plain_prompt_skip_without_truncation():
    row = example_rows()[0][1]
    tokens, audit = encode_example(row, TinyTokenizer(), 50)
    assert tokens is None and audit["reason"] == "protected_response_exchange_exceeds_window"
    row["prompt"] = [{"role": "user", "content": "x" * 100}]
    assert encode_example(row, TinyTokenizer(), 50)[0] is None


def test_template_boundary_mismatch_is_fatal():
    class BadTokenizer(TinyTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            result = super().apply_chat_template(*args, **kwargs)
            return [123] + result if kwargs.get("add_generation_prompt") else result
    with pytest.raises(ValueError, match="prefix"):
        encode_example(example_rows()[0][0], BadTokenizer(), 4096)


def test_snapshot_matches_hashes_and_cannot_be_overwritten(tmp_path):
    from format_conversation_dataset.util import digest
    datasets = {"train": example_rows()[0], "validation": example_rows("validation", " Different.")[0]}
    output = tmp_path / "snapshot"
    manifest = prepare(datasets, output, TinyTokenizer())
    assert manifest["examples"]["train"] == 2
    for split, expected in manifest["tokenized_sha256"].items():
        assert digest((output / f"{split}.tokens.jsonl").read_bytes()) == expected
    with pytest.raises(ValueError, match="immutable"):
        prepare(datasets, output, TinyTokenizer())


def test_legacy_interface_accepts_integer_speaker_without_dropping_unknown_lines():
    document = process_speakers("Speaker SPEAKER_01: Hello\nSpeaker SPEAKER_00: Hi", 0)
    assert document["messages"][-1] == {"role": "assistant", "content": "Hi"}
    with pytest.raises(ValueError, match="Unrecognized"):
        process_speakers("unlabeled content", 0)


def test_chat_exports_are_not_source_transcripts(tmp_path):
    source = tmp_path / "input.json"
    source.write_text('{"messages": []}')
    with pytest.raises(ValueError, match="chat export"):
        read_source(source)
