"""Extract a JSON response from one sentence using the base model or LoRA."""

import argparse

from .common import add_run_arguments, load_model, predict, run_directory, selected_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("sentence", help="French employment sentence to extract")
    parser.add_argument("--base", action="store_true", help="Use the untouched base model")
    args = parser.parse_args()
    cfg = selected_config(args)
    adapter = None if args.base else run_directory(cfg) / "adapter"
    if adapter and not (adapter / "adapter_config.json").exists():
        parser.error("LoRA adapter missing; run training or pass --base")
    model, tokenizer = load_model(cfg["project"]["model_name"], adapter, cfg["quantization"],
                                  cfg["project"].get("model_revision"))
    print(predict(model, tokenizer, args.sentence, cfg["generation"]))


if __name__ == "__main__":
    main()
