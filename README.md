# Conversational Dataset Formatter

Turn speaker-labeled conversations into **responses to other people**, with the context needed to understand each response. Preserve the original wording, review the targets, keep related sources in the same split, and train only on the assistant's completion.

The library works with arbitrary speakers, subjects, and assistant roles. It accepts speaker-labeled text and WhisperX JSON with word-level timing. It does not need a hosted model or a running application.

## Install

From a checkout, with Python 3.10 or newer:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[tokenize]'
conversation-dataset --help
```

Install from this checkout for the current pipeline; an older package release has only the original conversation converter. Download the tokenizer you intend to train with separately. Preparation loads that tokenizer locally and does not download model weights.

## Prepare a dataset

Keep source material and generated outputs in a separate local directory. The filenames below are placeholders, not bundled inputs.

1. Create a review worksheet for each source. Speaker IDs must match the source exactly.

   ```sh
   conversation-dataset review /path/to/private/input.txt \
     --assistant SPEAKER_00 --output /path/to/private/review.json
   ```

   The worksheet includes the source turns and candidate responses. After reading the response **and its preceding exchange**, set suitable candidates to `"verdict": "keep"` and identify the reviewer. Leave unsuitable candidates as `pending` or set `reject`. `origin` is `human` for an actual human review or `model` for automated screening. The latter requires explicit opt-in during the build.

2. Create a manifest outside the checkout. Use separate conversation groups for training and validation. All related sessions and alternate versions belong to the same group. Assign the same `source_identity` to alternate transcriptions of a recording; the build permits only one variant.

   ```json
   {
     "system": "Respond to participants' requests in context. Ask when information is missing.",
     "sources": [
       {
         "path": "training-input.txt",
         "reviews": "training-review.json",
         "group": "group-a",
         "source_identity": "recording-a",
         "split": "train",
         "assistant_speakers": ["SPEAKER_00"]
       },
       {
         "path": "validation-input.txt",
         "reviews": "validation-review.json",
         "group": "group-b",
         "source_identity": "recording-b",
         "split": "validation",
         "assistant_speakers": ["SPEAKER_00"]
       }
     ]
   }
   ```

   Paths resolve relative to the manifest. `test` is also supported. Split assignment is explicit: randomly dividing turns from the same conversation would leak context across the boundary.

3. Build a new snapshot with the actual local tokenizer.

   ```sh
   conversation-dataset build /path/to/private/manifest.json \
     --tokenizer /path/to/local/model \
     --sequence-len 4096 --output /path/to/private/snapshot
   ```

   Add `--allow-screened` only when deliberately including model-screened responses. Their provenance remains `model-screened`; screening is never relabeled as human review.

## What the pipeline protects

- Consecutive turns by the same assistant become one complete response. The full preceding participant stimulus and preceding assistant exchange are protected during context fitting.
- Review decisions bind to source hashes, response boundaries, and exact response text. Source changes require a new review.
- Unknown speaker lines survive import. Ambiguous context is skipped rather than silently attributed to the assistant.
- Related groups, source identities, duplicate examples, and identical responses cannot cross training/evaluation splits. Different transcriptions of one recording cannot be counted as separate sources.
- Targets and future turns never enter their own context. Targets and required exchanges are never truncated to fit a token limit.
- The tokenizer's generation prefix must match the full chat serialization exactly. Only completion tokens and the end-of-turn token receive loss; prompt labels are `-100`.
- Existing snapshots are not overwritten. Hashes, skipped-example reasons, review origins, and token-mask audits accompany the outputs.

These checks cannot discover that differently labeled groups secretly belong to the same source, or certify semantic quality. Source grouping and target review still require judgment. Strict duplicate-response checks can also reject common short replies; curate those examples instead of treating repeated filler as useful independent evidence.

## Output contract

Each snapshot contains `manifest.json`, `{split}.sources.jsonl`, `{split}.tokens.jsonl`, and `{split}.mask-audit.json`. The tokenized rows have `input_ids`, `attention_mask`, and `labels`. The manifest records counts, sequence length, and SHA-256 hashes.

This is the input contract consumed by **qwen-ttrpg**. Other trainers can use the same pretokenized rows. `pipeline.prepare` also accepts already-reviewed prompt/completion rows for structured tasks; each needs group/source/split provenance and preceding/target turn numbers. The formatter does not invent task labels or rule knowledge.

All generated files are private working artifacts. They contain source text and are intentionally excluded from source control.

## Original conversion API

The original whole-conversation interface remains available:

```python
from format_conversation_dataset.format_transcription import convert_file

convert_file(
    "input.txt", assistant_speaker=0, output_file_path="conversation.json",
    system_context="Participate in the conversation.",
)
```

It produces a `messages` array, not reviewed response examples or token masks. Unrecognized speaker lines now raise an error instead of disappearing. Use the review pipeline when speaker labels need attention.

## Development

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/check_public_tree.py
```

CI checks Python 3.10 and 3.12. Tests use short, original synthetic phrases; no source conversations or recordings are distributed. For a release audit, `check_public_tree.py --history --terms-file /path/to/private/identifiers.txt` also scans reachable branch/tag history and an external list of private identifiers. The list itself must stay outside the repository.
