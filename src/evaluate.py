"""Compare cached baseline predictions against a LoRA adapter on the same test rows."""

import argparse
import json

from .common import (add_run_arguments, experiment_identity, load_model, predict,
                     read_jsonl, run_directory, selected_config, write_jsonl)
from .metrics import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    args = parser.parse_args()
    cfg = selected_config(args)
    expected = experiment_identity(cfg)
    run_dir = run_directory(cfg)
    manifests = (run_dir / "baseline_manifest.json", run_dir / "adapter" / "experiment_manifest.json")
    for manifest_path in manifests:
        if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding="utf-8")) != expected:
            parser.error(f"Missing or mismatched {manifest_path}; rerun baseline and training with the current config")
    baseline = read_jsonl(run_dir / "baseline_predictions.jsonl")
    test = read_jsonl(cfg["data"]["test_path"])
    if not baseline:
        parser.error("Baseline predictions are empty; run python -m src.baseline first")
    if len(baseline) > len(test) or any(
        left["input"] != right["input"] or left["reference"] != right["output"]
        for left, right in zip(baseline, test)
    ):
        parser.error("Baseline predictions do not match the current test set; rerun baseline")
    test = test[:len(baseline)]
    adapter = run_dir / "adapter"
    if not (adapter / "adapter_config.json").exists():
        parser.error(f"Adapter missing at {adapter}; run python -m src.train first")
    model, tokenizer = load_model(cfg["project"]["model_name"], adapter=adapter, quantization=cfg["quantization"])
    fine_tuned = []
    for index, row in enumerate(test, 1):
        fine_tuned.append({"input": row["input"], "reference": row["output"],
                           "prediction": predict(model, tokenizer, row["input"], cfg["generation"])})
        if index % 10 == 0 or index == len(test):
            print(f"LoRA evaluation: {index}/{len(test)}")
    write_jsonl(run_dir / "lora_predictions.jsonl", fine_tuned)
    results = {"base": score(baseline), "lora": score(fine_tuned)}
    destination = run_dir / "comparison.json"
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nMetric              Base     LoRA")
    print("---------------------------------")
    for label, key in (("Valid JSON", "valid_json"), ("Exact match", "exact_match")):
        print(f"{label:<19} {results['base'][key]:.3f}    {results['lora'][key]:.3f}")
    for field in ("name", "job", "company", "city", "since"):
        print(f"{field.title() + ' F1':<19} {results['base']['field_f1'][field]:.3f}    {results['lora']['field_f1'][field]:.3f}")
    print(f"{'Global F1':<19} {results['base']['global_f1']:.3f}    {results['lora']['global_f1']:.3f}")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
