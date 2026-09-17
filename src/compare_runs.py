"""List completed model/quantization experiments without loading model weights."""

import json

from .common import config, path


def main():
    root = path(config()["training"]["results_dir"])
    files = sorted(root.glob("*/comparison.json"))
    if not files:
        print("No completed comparisons. Run baseline, train, and evaluate first.")
        return
    print(f"{'Model':<38} {'Mode':<7} {'Test ID':<8} {'Base F1':>8} {'LoRA F1':>8} {'Base exact':>11} {'LoRA exact':>11}")
    for file in files:
        scores = json.loads(file.read_text(encoding="utf-8"))
        manifest = json.loads((file.parent / "baseline_manifest.json").read_text(encoding="utf-8"))
        model = manifest["model_name"].split("/")[-1]
        test_id = manifest["data_sha256"]["test"][:8]
        print(f"{model:<38} {manifest['quantization']['mode']:<7} {test_id:<8} "
              f"{scores['base']['global_f1']:>8.3f} {scores['lora']['global_f1']:>8.3f} "
              f"{scores['base']['exact_match']:>11.3f} {scores['lora']['exact_match']:>11.3f}")


if __name__ == "__main__":
    main()
