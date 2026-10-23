"""Route one request through the base model or trained tool-calling adapter."""
import argparse
import json
from .tool_catalog import TOOLS
from .tool_runtime import add_arguments, load_model, parse_action, predict_action, run_dir, settings

def main():
    parser = argparse.ArgumentParser(description=__doc__); add_arguments(parser)
    parser.add_argument("request"); parser.add_argument("--base", action="store_true"); args = parser.parse_args(); cfg = settings(args)
    adapter = None if args.base else run_dir(cfg) / "adapter"
    model, tokenizer = load_model(cfg["project"]["model_name"], adapter, cfg["quantization"], cfg["project"].get("model_revision"))
    row = {"request": args.request, "tools": TOOLS}; raw = predict_action(model, tokenizer, row, cfg["generation"])
    parsed = parse_action(raw, TOOLS); print(json.dumps(parsed, indent=2, ensure_ascii=False) if parsed else raw)

if __name__ == "__main__": main()
