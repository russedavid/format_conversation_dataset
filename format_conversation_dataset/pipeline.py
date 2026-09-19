"""Build auditable responses without changing source text or guessing speaker roles."""

from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

from .parsers import parse_text, parse_whisperx
from .tokenize import encode_example
from .util import digest, now, packed

SPLITS = {"train", "validation", "test"}


def read_source(path):
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8-sig")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = None
    if document is not None:
        if not isinstance(document, dict) or not isinstance(document.get("segments"), list):
            raise ValueError("Expected speaker text or word-aligned segments, not a chat export.")
        turns = list(parse_whisperx(document))
    else:
        turns = list(parse_text(text))
    if not turns:
        raise ValueError("Source has no turns.")
    return turns, digest(raw)


def response_examples(turns, *, group, source_sha256, split, assistant_speakers,
                      system, reviews=None, allow_screened=False, context_turns=64,
                      source_identity=None):
    """A review is bound to an exact contiguous response, never an ordinal alone.

    Review items contain target_turns (one-based), target_sha256, verdict,
    origin ('human' or 'model'), and reviewer. Unreviewed targets are not exported.
    source_identity links alternate transcriptions of the same recording.
    """
    if split not in SPLITS or not group or not source_sha256 or context_turns < 1:
        raise ValueError("Choose a group, source identity, split, and positive context window.")
    assistants = set(assistant_speakers)
    if not assistants or "Unknown" in assistants or "UNKNOWN" in assistants:
        raise ValueError("Map explicit assistant speaker labels before export.")
    indexed = [dict(t, turn=i) for i, t in enumerate(turns, 1)]
    decisions = {}
    for review in reviews or []:
        key = tuple(review["target_turns"])
        if key in decisions:
            raise ValueError("Duplicate review decisions for a response.")
        decisions[key] = review
    rows, candidates, skipped = [], [], Counter()
    i = 0
    while i < len(indexed):
        turn = indexed[i]
        if turn["speaker"] not in assistants:
            i += 1
            continue
        end = i + 1
        while end < len(indexed) and indexed[end]["speaker"] == turn["speaker"]:
            end += 1
        targets = indexed[i:end]
        target = " ".join(t["text"] for t in targets)
        numbers = tuple(t["turn"] for t in targets)
        candidate = {"target_turns": list(numbers), "target_sha256": digest(target),
                     "verdict": "pending", "origin": "human", "reviewer": ""}
        candidates.append(candidate)
        # Protect the full stimulus plus the preceding assistant exchange.
        start = i
        while start > 0 and indexed[start - 1]["speaker"] not in assistants:
            start -= 1
        has_stimulus = start < i
        if start > 0:
            speaker = indexed[start - 1]["speaker"]
            while start > 0 and indexed[start - 1]["speaker"] == speaker:
                start -= 1
        context = indexed[min(max(0, i - context_turns), start):i]
        candidate.update(response=target, context=context, has_stimulus=has_stimulus,
                         protected_context_turns=[t["turn"] for t in indexed[start:i]])
        decision = decisions.get(numbers)
        reason = None
        if not has_stimulus:
            reason = "no_response_stimulus"
        elif any(t["speaker"] in {"Unknown", "UNKNOWN", ""} for t in context):
            reason = "unresolved_context_speaker"
        elif not target.strip():
            reason = "empty_response"
        elif decision is None or decision.get("verdict") != "keep":
            reason = "not_reviewed"
        elif decision.get("target_sha256") != digest(target):
            raise ValueError("Review is stale: response text changed.")
        elif decision.get("origin") not in {"human", "model"} or not decision.get("reviewer"):
            raise ValueError("Kept responses require an explicit review origin and reviewer.")
        elif decision["origin"] == "model" and not allow_screened:
            reason = "model_screening_not_enabled"
        if reason:
            skipped[reason] += 1
        else:
            dialogue = [{"turn": t["turn"], "speaker": t["speaker"], "text": t["text"],
                         "role": "assistant" if t["speaker"] in assistants else "participant"}
                        for t in context]
            prompt = [{"role": "system", "content": system},
                      {"role": "user", "content": packed({"recent_dialogue": dialogue,
                        "instruction": "Respond to the latest participants in context."})}]
            completion = [{"role": "assistant", "content": target}]
            provenance = {"group": group, "source_sha256": source_sha256,
                          "source_identity": source_identity or source_sha256,
                          "split": split, "target_turn": numbers[0], "target_turns": list(numbers),
                          "target_sha256": digest(target),
                          "context_turns": [t["turn"] for t in context],
                          "protected_context_turns": [t["turn"] for t in indexed[start:i]],
                          "review_level": "reviewed" if decision["origin"] == "human" else "model-screened",
                          "reviewer": decision["reviewer"]}
            rows.append({"id": digest(packed(prompt) + packed(completion)),
                         "prompt": prompt, "completion": completion, "provenance": provenance})
        i = end
    known = {tuple(c["target_turns"]) for c in candidates}
    if set(decisions) - known:
        raise ValueError("Review references missing or changed response boundaries.")
    return rows, candidates, dict(skipped)


def verify_splits(datasets, *, check_response_duplicates=True):
    ownership = {}
    variants = {}
    for split, rows in datasets.items():
        if split not in SPLITS:
            raise ValueError("Unknown split.")
        for row in rows:
            p = row["provenance"]
            if p["split"] != split:
                raise ValueError("Row split does not match its dataset.")
            group = p.get("group", p.get("story"))
            if not group or not p.get("source_sha256"):
                raise ValueError("Group and source provenance are required.")
            identities = [("group", group), ("source", p["source_sha256"]),
                                ("source_identity", p.get("source_identity", p["source_sha256"])),
                                ("example", digest(packed(row["prompt"]) + packed(row["completion"])))]
            if check_response_duplicates:
                identities.append(("response", digest(packed(row["completion"]))))
            for kind, value in identities:
                key = (kind, value)
                if key in ownership and ownership[key] != split:
                    raise ValueError(f"{kind} crosses training/evaluation splits.")
                ownership[key] = split
            identity = p.get("source_identity", p["source_sha256"])
            if identity in variants and variants[identity] != p["source_sha256"]:
                raise ValueError("Choose one transcript variant per source identity.")
            variants[identity] = p["source_sha256"]
            targets = p.get("target_turns", [p["target_turn"]])
            if not targets or any(t >= min(targets) for t in p.get("context_turns", [])):
                raise ValueError("Context contains a target or future turn.")
            if len(row["completion"]) != 1 or row["completion"][0]["role"] != "assistant":
                raise ValueError("Exactly one assistant completion is required.")


def prepare(datasets, output, tokenizer, sequence_len=4096, *, check_response_duplicates=True):
    """Write a new immutable snapshot with loss-mask and split provenance audits."""
    verify_splits(datasets, check_response_duplicates=check_response_duplicates)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("Choose a new output directory; existing snapshots are immutable.")
    encoded, audits, kept, omitted = {}, {}, {}, Counter()
    for split in sorted(SPLITS):
        encoded[split], audits[split], kept[split] = [], [], []
        seen = set()
        for row in datasets.get(split, []):
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            tokens, audit = encode_example(row, tokenizer, sequence_len)
            if tokens is None:
                omitted[split + ":" + audit["reason"]] += 1
                continue
            encoded[split].append(tokens)
            audits[split].append(audit)
            kept[split].append(row)
    if not encoded["train"] or not encoded["validation"]:
        raise ValueError("Nonempty, separate training and validation examples are required.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".preparing-", dir=output.parent))
    try:
        hashes, source_hashes = {}, {}
        for split in sorted(SPLITS):
            for suffix, items in [("sources", kept[split]), ("tokens", encoded[split])]:
                payload = "".join(packed(r) + "\n" for r in items)
                (staging / f"{split}.{suffix}.jsonl").write_text(payload, encoding="utf-8")
                (hashes if suffix == "tokens" else source_hashes)[split] = digest(payload)
            (staging / f"{split}.mask-audit.json").write_text(packed(audits[split]), encoding="utf-8")
        manifest = {"created": now(), "format_version": 1, "sequence_len": sequence_len,
                    "check_response_duplicates": check_response_duplicates,
                    "examples": {s: len(encoded[s]) for s in sorted(SPLITS)},
                    "tokenized_sha256": hashes, "sources_sha256": source_hashes,
                    "omitted": dict(omitted), "label_policy": "Only assistant completion and end-of-turn tokens have loss."}
        (staging / "manifest.json").write_text(packed(manifest), encoding="utf-8")
        if output.exists():
            raise ValueError("Output appeared during preparation; refusing to replace it.")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest
