"""Lossless text and word-aligned speaker parsing."""
import re


def parse_text(text):
    """Preserve every nonblank line, including unknown/unrecognized speaker labels."""
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = re.match(
            r"^(?:Speaker[ _])?(SPEAKER_\d+|Unknown|UNKNOWN)\s*:\s*(.*)$", line
        )
        yield dict(
            speaker=match[1] if match else "Unknown",
            text=match[2] if match else line,
            start=None,
            end=None,
            metadata={"source_line": line_no, "timing": "unavailable"},
        )


def parse_whisperx(document):
    """Keep word timing and split mixed segments at observed word-speaker changes."""
    for i, segment in enumerate(document["segments"]):
        words = segment.get("words") or []
        if not words:
            yield dict(
                speaker=segment.get("speaker", "Unknown"),
                text=segment["text"],
                start=segment.get("start"),
                end=segment.get("end"),
                metadata={"segment": i, "words": [], "timing": "segment"},
            )
            continue
        group = []
        speaker = None
        for word in words:
            current = word.get("speaker", segment.get("speaker", "Unknown"))
            if group and speaker != current:
                yield word_group(group, speaker, i)
                group = []
            speaker = current
            group.append(word)
        if group:
            yield word_group(group, speaker, i)


def word_group(words, speaker, segment):
    return dict(
        speaker=speaker or "Unknown",
        text=" ".join(w.get("word", "") for w in words).strip(),
        start=next((w["start"] for w in words if "start" in w), None),
        end=next((w["end"] for w in reversed(words) if "end" in w), None),
        metadata={"segment": segment, "words": words, "timing": "word"},
    )
