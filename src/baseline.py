"""Generate predictions from the untouched instruction model on the test set."""

import argparse
import json

from .common import (add_run_arguments, experiment_identity, load_model, predict,
                     read_jsonl, run_directory, selected_config, write_jsonl)
from .metrics import score, score_by_tag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--limit", type=int, help="Use only the first N test examples for a quick smoke run")
    parser.add_argument("--split", choices=("hard", "easy"), default="hard")
    parser.add_argument("--strategy", choices=("zero", "few"), default="zero")
    args = parser.parse_args()
    cfg = selected_config(args)
    test_path = cfg["data"]["test_path" if args.split == "hard" else "easy_test_path"]
    rows = read_jsonl(test_path)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[:args.limit]
    model, tokenizer = load_model(cfg["project"]["model_name"], quantization=cfg["quantization"],
                                  revision=cfg["project"].get("model_revision"))
    examples = []
    if args.strategy == "few":
        train_rows = read_jsonl(cfg["data"]["train_path"])
        for tag in ("missing", "relative", "former"):
            examples.append(next(row for row in train_rows if tag in row["tags"]))
    output = []
    for index, row in enumerate(rows, 1):
        output.append({"input": row["input"], "reference": row["output"], "tags": row["tags"],
                       "prediction": predict(model, tokenizer, row["input"], cfg["generation"], examples)})
        if index % 10 == 0 or index == len(rows):
            print(f"{args.strategy} {args.split}: {index}/{len(rows)}")
    stem = f"{args.strategy}_{args.split}"
    destination = run_directory(cfg) / f"{stem}_predictions.jsonl"
    write_jsonl(destination, output)
    (destination.parent / f"{stem}_manifest.json").write_text(
        json.dumps(experiment_identity(cfg), indent=2) + "\n", encoding="utf-8"
    )
    baseline_scores = score(output)
    (destination.parent / f"{stem}_metrics.json").write_text(
        json.dumps({"overall": baseline_scores, "by_tag": score_by_tag(output)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Valid JSON: {baseline_scores['valid_json']:.3f}; "
          f"exact match: {baseline_scores['exact_match']:.3f}; "
          f"global F1: {baseline_scores['global_f1']:.3f}")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
