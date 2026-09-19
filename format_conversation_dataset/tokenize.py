"""Completion-only masks; keep the full response and its required exchange."""
import copy
import json
from .util import packed


def encode_example(row, tokenizer, sequence_len):
    prompt = copy.deepcopy(row["prompt"])
    completion = row["completion"]
    removed = []
    while True:
        prefix = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full = tokenizer.apply_chat_template(
            prompt + completion, tokenize=True, return_dict=False, enable_thinking=False
        )
        if full[: len(prefix)] != prefix:
            raise ValueError(
                "Chat-template prefix differs before completion; do not guess the loss boundary."
            )
        if len(full) <= sequence_len:
            break
        try:
            content = json.loads(prompt[-1]["content"])
        except (json.JSONDecodeError, TypeError):
            return None, {"reason": "target_or_minimal_context_exceeds_window"}
        if not isinstance(content, dict):
            return None, {"reason": "target_or_minimal_context_exceeds_window"}
        dialogue = content.get("recent_dialogue", [])
        protected = set(row.get("provenance", {}).get("protected_context_turns", []))
        if dialogue and dialogue[0]["turn"] in protected:
            return None, {"reason": "protected_response_exchange_exceeds_window"}
        if len(dialogue) <= 1:
            return None, {"reason": "target_or_minimal_context_exceeds_window"}
        removed.append(dialogue.pop(0)["turn"])
        prompt[-1]["content"] = packed(content)
    target = full[len(prefix) :]
    if not target or tokenizer.eos_token_id not in target:
        raise ValueError("Assistant completion must include its end-of-turn token.")
    labels = [-100] * len(prefix) + target
    assert len(labels) == len(full) and all(x == -100 for x in labels[: len(prefix)])
    encoded = {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}
    audit = {
        "id": row["id"],
        "prefix_tokens": len(prefix),
        "completion_tokens": len(target),
        "total_tokens": len(full),
        "removed_context_turns": removed,
        "protected_context_turns": row.get("provenance", {}).get(
            "protected_context_turns", []
        ),
        "decoded_completion": tokenizer.decode(target),
        "provenance": row["provenance"],
    }
    return encoded, audit
