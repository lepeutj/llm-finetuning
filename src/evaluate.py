"""Compare zero-shot, few-shot, and LoRA on the same held-out examples."""

import argparse
import json

from .common import (add_run_arguments, experiment_identity, load_model, predict,
                     read_jsonl, run_directory, selected_config, write_jsonl)
from .metrics import score, score_by_tag


def _baseline(parser, run_dir, strategy, split, expected, test, allow_partial):
    stem = f"{strategy}_{split}"
    manifest = run_dir / f"{stem}_manifest.json"
    if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")) != expected:
        parser.error(f"Missing or mismatched {manifest}; rerun the {stem} baseline")
    rows = read_jsonl(run_dir / f"{stem}_predictions.jsonl")
    if not rows or len(rows) > len(test) or (len(rows) < len(test) and not allow_partial):
        parser.error(f"{stem} has {len(rows)} predictions; expected {len(test)}. "
                     "Use --allow-partial only for a smoke test")
    for prediction, reference in zip(rows, test):
        if (prediction["input"] != reference["input"] or
                prediction["reference"] != reference["output"] or
                prediction["tags"] != reference["tags"]):
            parser.error(f"{stem} predictions do not match the current {split} test set")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--split", choices=("hard", "easy"), default="hard")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Permit a limited baseline or one-step adapter for a smoke test")
    args = parser.parse_args()
    cfg = selected_config(args)
    expected = experiment_identity(cfg)
    run_dir = run_directory(cfg)
    test_path = cfg["data"]["test_path" if args.split == "hard" else "easy_test_path"]
    test = read_jsonl(test_path)
    strategies = ("zero", "few") if args.split == "hard" else ("zero",)
    baselines = {strategy: _baseline(parser, run_dir, strategy, args.split, expected, test,
                                     args.allow_partial) for strategy in strategies}
    counts = {len(rows) for rows in baselines.values()}
    if len(counts) != 1:
        parser.error("Baseline strategies have different sample counts; rerun them with the same limit")
    test = test[:counts.pop()]

    adapter = run_dir / "adapter"
    manifest = adapter / "experiment_manifest.json"
    if not manifest.exists():
        parser.error(f"Adapter missing at {adapter}; run python -m src.train first")
    training_run = json.loads(manifest.read_text(encoding="utf-8"))
    if training_run["identity"] != expected:
        parser.error("Adapter was trained with different data or model settings")
    if training_run["training_run"]["smoke_test"] and not args.allow_partial:
        parser.error("Adapter was trained with a smoke-test limit; rerun full training or use --allow-partial")

    model, tokenizer = load_model(cfg["project"]["model_name"], adapter=adapter,
                                  quantization=cfg["quantization"],
                                  revision=cfg["project"].get("model_revision"))
    fine_tuned = []
    for index, row in enumerate(test, 1):
        fine_tuned.append({"input": row["input"], "reference": row["output"], "tags": row["tags"],
                           "prediction": predict(model, tokenizer, row["input"], cfg["generation"])})
        if index % 10 == 0 or index == len(test):
            print(f"LoRA {args.split}: {index}/{len(test)}")
    write_jsonl(run_dir / f"lora_{args.split}_predictions.jsonl", fine_tuned)
    predictions = {**baselines, "lora": fine_tuned}
    results = {name: {"overall": score(rows), "by_tag": score_by_tag(rows)}
               for name, rows in predictions.items()}
    results["smoke_test"] = args.allow_partial and (
        len(test) < int(cfg["data"]["test_size" if args.split == "hard" else "easy_test_size"])
        or training_run["training_run"]["smoke_test"]
    )
    destination = run_dir / f"comparison_{args.split}.json"
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{args.split.upper()} comparison on {len(test)} examples"
          + (" [SMOKE TEST]" if results["smoke_test"] else ""))
    print(f"{'Method':<12} {'Valid JSON':>10} {'Exact':>8} {'Global F1':>10} {'Null year':>10}")
    for name, details in predictions.items():
        metrics = results[name]["overall"]
        null_score = metrics["since_missing_accuracy"]
        print(f"{name:<12} {metrics['valid_json']:>10.3f} {metrics['exact_match']:>8.3f} "
              f"{metrics['global_f1']:>10.3f} "
              f"{null_score if null_score is not None else float('nan'):>10.3f}")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
