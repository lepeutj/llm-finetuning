"""Recompute tool-calling metrics from saved predictions without loading a model."""
import argparse
import json

from .tool_runtime import add_arguments, run_dir, score, settings


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    cfg = settings(args)
    target = run_dir(cfg)
    base = read(target / "base_predictions.jsonl")
    tuned = read(target / "lora_predictions.jsonl")
    adapter_manifest = json.loads(
        (target / "adapter" / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    comparison = {
        "base": score(base),
        "lora": score(tuned),
        "smoke_test": len(base) < int(cfg["data"]["test_size"]) or adapter_manifest["smoke_test"],
    }
    destination = target / "comparison.json"
    destination.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
