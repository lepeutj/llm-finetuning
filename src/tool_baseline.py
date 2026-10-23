"""Evaluate the untouched model on held-out tool-routing examples."""
import argparse
import json
from .tool_runtime import add_arguments, identity, load_model, predict_action, read_rows, run_dir, score, settings

def main():
    parser = argparse.ArgumentParser(description=__doc__); add_arguments(parser)
    parser.add_argument("--limit", type=int); args = parser.parse_args(); cfg = settings(args)
    rows = read_rows(cfg, "test")[:args.limit] if args.limit else read_rows(cfg, "test")
    model, tokenizer = load_model(cfg["project"]["model_name"], quantization=cfg["quantization"],
                                  revision=cfg["project"].get("model_revision"))
    predictions = []
    for index, row in enumerate(rows, 1):
        predictions.append({"id": row["id"], "tools": row["tools"], "reference": row["expected_action"],
                            "tags": row["tags"], "prediction": predict_action(model, tokenizer, row, cfg["generation"])})
        if index % 10 == 0 or index == len(rows): print(f"base: {index}/{len(rows)}")
    target = run_dir(cfg); target.mkdir(parents=True, exist_ok=True)
    (target / "base_predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in predictions), encoding="utf-8")
    (target / "base_manifest.json").write_text(json.dumps(identity(cfg), indent=2)+"\n", encoding="utf-8")
    metrics = score(predictions); (target / "base_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"Saved {target}")

if __name__ == "__main__": main()
