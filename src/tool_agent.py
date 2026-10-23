"""Run one complete local agent turn: route, execute a simulated tool, and answer."""
import argparse
import json
from .tool_catalog import TOOLS, execute
from .tool_runtime import (add_arguments, followup_messages, load_model, parse_action,
                           predict_action, predict_messages, run_dir, settings)

def main():
    parser = argparse.ArgumentParser(description=__doc__); add_arguments(parser)
    parser.add_argument("request"); parser.add_argument("--base", action="store_true")
    args = parser.parse_args(); cfg = settings(args); adapter = None if args.base else run_dir(cfg) / "adapter"
    model, tokenizer = load_model(cfg["project"]["model_name"], adapter, cfg["quantization"], cfg["project"].get("model_revision"))
    row = {"request": args.request, "tools": TOOLS}; raw = predict_action(model, tokenizer, row, cfg["generation"])
    decision = parse_action(raw, TOOLS)
    print("Decision:", json.dumps(decision, ensure_ascii=False, indent=2) if decision else raw)
    if not decision or decision["type"] != "tool_call": return
    result = execute(decision["name"], decision["arguments"]); print("Tool result:", json.dumps(result, ensure_ascii=False, indent=2))
    row["expected_action"] = decision
    row["trajectory"] = [{"role": "user", "content": args.request}, {"role": "assistant", "content": decision},
                         {"role": "tool", "name": decision["name"], "content": result}]
    final = predict_messages(model, tokenizer, followup_messages(row), cfg["generation"])
    print("Final response:", final)

if __name__ == "__main__": main()
