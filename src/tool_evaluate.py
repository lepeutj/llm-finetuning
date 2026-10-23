"""Evaluate a saved tool-calling adapter against its matching baseline."""
import argparse
import json
from .tool_runtime import add_arguments, identity, load_model, predict_action, read_rows, run_dir, score, settings

def main():
    parser = argparse.ArgumentParser(description=__doc__); add_arguments(parser)
    parser.add_argument("--allow-partial", action="store_true"); args = parser.parse_args(); cfg = settings(args)
    target = run_dir(cfg); expected = identity(cfg); test = read_rows(cfg, "test")
    manifest = json.loads((target / "base_manifest.json").read_text(encoding="utf-8"))
    if manifest != expected: parser.error("Baseline does not match current model or data")
    baseline = [json.loads(x) for x in (target / "base_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(baseline) != len(test) and not args.allow_partial: parser.error("Partial baseline; pass --allow-partial only for a smoke test")
    test = test[:len(baseline)]; adapter = target / "adapter"
    adapter_manifest = json.loads((adapter / "experiment_manifest.json").read_text(encoding="utf-8"))
    if adapter_manifest["identity"] != expected: parser.error("Adapter does not match current model or data")
    if adapter_manifest["smoke_test"] and not args.allow_partial: parser.error("Smoke adapter; pass --allow-partial")
    model, tokenizer = load_model(cfg["project"]["model_name"], adapter, cfg["quantization"], cfg["project"].get("model_revision"))
    tuned = []
    for index, row in enumerate(test, 1):
        tuned.append({"id": row["id"], "tools": row["tools"], "reference": row["expected_action"],
                      "tags": row["tags"], "prediction": predict_action(model, tokenizer, row, cfg["generation"])})
        if index % 10 == 0 or index == len(test): print(f"LoRA: {index}/{len(test)}")
    (target / "lora_predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in tuned), encoding="utf-8")
    comparison = {"base": score(baseline), "lora": score(tuned),
                  "smoke_test": len(test) < cfg["data"]["test_size"] or adapter_manifest["smoke_test"]}
    (target / "comparison.json").write_text(json.dumps(comparison, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2)); print(f"Saved {target / 'comparison.json'}")

if __name__ == "__main__": main()
