"""Generate predictions from the untouched instruction model on the test set."""

import argparse
import json

from .common import (add_run_arguments, experiment_identity, load_model, predict,
                     read_jsonl, run_directory, selected_config, write_jsonl)
from .metrics import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--limit", type=int, help="Use only the first N test examples for a quick smoke run")
    args = parser.parse_args()
    cfg = selected_config(args)
    rows = read_jsonl(cfg["data"]["test_path"])
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[:args.limit]
    model, tokenizer = load_model(cfg["project"]["model_name"], quantization=cfg["quantization"],
                                  revision=cfg["project"].get("model_revision"))
    output = []
    for index, row in enumerate(rows, 1):
        output.append({"input": row["input"], "reference": row["output"],
                       "prediction": predict(model, tokenizer, row["input"], cfg["generation"])})
        if index % 10 == 0 or index == len(rows):
            print(f"Baseline: {index}/{len(rows)}")
    destination = run_directory(cfg) / "baseline_predictions.jsonl"
    write_jsonl(destination, output)
    (destination.parent / "baseline_manifest.json").write_text(
        json.dumps(experiment_identity(cfg), indent=2) + "\n", encoding="utf-8"
    )
    baseline_scores = score(output)
    (destination.parent / "baseline_metrics.json").write_text(
        json.dumps(baseline_scores, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Valid JSON: {baseline_scores['valid_json']:.3f}; "
          f"exact match: {baseline_scores['exact_match']:.3f}; "
          f"global F1: {baseline_scores['global_f1']:.3f}")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
