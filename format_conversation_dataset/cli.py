"""Local review and dataset preparation; no uploads or model downloads."""

import argparse
import json
from pathlib import Path

from .pipeline import read_source, response_examples, prepare
from .util import packed


def arguments(parser):
    parser.add_argument("source")
    parser.add_argument("--assistant", action="append", required=True, help="Exact speaker label; repeat if needed")
    parser.add_argument("--output", required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review", help="Create a review worksheet with source-bound response hashes")
    arguments(review)
    build = commands.add_parser("build", help="Build reviewed data from an external source manifest")
    build.add_argument("manifest")
    build.add_argument("--tokenizer", required=True, help="An already-downloaded local tokenizer directory")
    build.add_argument("--output", required=True)
    build.add_argument("--sequence-len", type=int, default=4096)
    build.add_argument("--allow-screened", action="store_true")
    args = parser.parse_args()
    if args.command == "review":
        turns, fingerprint = read_source(args.source)
        _, candidates, _ = response_examples(turns, group="review", source_sha256=fingerprint,
                                             split="train", assistant_speakers=args.assistant, system="")
        document = {"source_sha256": fingerprint, "turns": turns, "reviews": candidates}
        with Path(args.output).open("x", encoding="utf-8") as f:
            f.write(packed(document) + "\n")
        print(f"Wrote {len(candidates)} pending responses; edit verdict and reviewer after review.")
        return
    path = Path(args.manifest).resolve()
    document = json.loads(path.read_text())
    datasets = {"train": [], "validation": [], "test": []}
    skipped = {}
    for source in document["sources"]:
        source_path = path.parent / source["path"]
        turns, fingerprint = read_source(source_path)
        review = json.loads((path.parent / source["reviews"]).read_text())
        if review["source_sha256"] != fingerprint:
            raise ValueError("Review worksheet is stale: original source changed.")
        rows, _, reasons = response_examples(
            turns, group=source["group"], source_sha256=fingerprint, split=source["split"],
            source_identity=source.get("source_identity"), assistant_speakers=source["assistant_speakers"],
            system=document["system"], reviews=review["reviews"], allow_screened=args.allow_screened)
        datasets[source["split"]].extend(rows)
        skipped[source["group"]] = reasons
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    manifest = prepare(datasets, args.output, tokenizer, args.sequence_len)
    print(json.dumps({"manifest": manifest, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
