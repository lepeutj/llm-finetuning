"""List completed model/quantization experiments without loading model weights."""

import json

from .common import config, path


def main():
    root = path(config()["training"]["results_dir"])
    files = sorted(root.glob("*/comparison_*.json"))
    if not files:
        print("No completed comparisons. Run baseline, train, and evaluate first.")
        return
    print(f"{'Model':<35} {'Split':<5} {'Mode':<5} {'Test ID':<8} {'Zero F1':>8} {'Few F1':>8} {'LoRA F1':>8}")
    for file in files:
        scores = json.loads(file.read_text(encoding="utf-8"))
        split = file.stem.removeprefix("comparison_")
        manifest = json.loads((file.parent / f"zero_{split}_manifest.json").read_text(encoding="utf-8"))
        model = manifest["model_name"].split("/")[-1]
        test_id = manifest["data_sha256"]["test" if split == "hard" else "easy_test"][:8]
        few = scores.get("few", {}).get("overall", {}).get("global_f1")
        few_text = f"{few:.3f}" if few is not None else "n/a"
        suffix = " SMOKE" if scores.get("smoke_test") else ""
        print(f"{model:<35} {split:<5} {manifest['quantization']['mode']:<5} {test_id:<8} "
              f"{scores['zero']['overall']['global_f1']:>8.3f} {few_text:>8} "
              f"{scores['lora']['overall']['global_f1']:>8.3f}{suffix}")


if __name__ == "__main__":
    main()
