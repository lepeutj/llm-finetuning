"""Supervised completion-only training of a Qwen LoRA adapter."""

import argparse
import json

from .common import (add_run_arguments, experiment_identity, lora_target_modules, messages,
                     quantization_options, read_jsonl, run_directory, selected_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--max-train-samples", type=int, help="Small smoke run before full training")
    parser.add_argument("--max-steps", type=int, help="Override epochs with a small number of optimizer steps")
    args = parser.parse_args()
    if args.max_train_samples is not None and args.max_train_samples < 1:
        parser.error("--max-train-samples must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    cfg = selected_config(args)
    model_name = cfg["project"]["model_name"]
    settings = cfg["training"]
    set_seed(int(cfg["project"]["seed"]))
    source = read_jsonl(cfg["data"]["train_path"])
    if args.max_train_samples is not None:
        source = source[:args.max_train_samples]
    def convert(row):
        return {
            "prompt": messages(row["input"]),
            "completion": [{"role": "assistant", "content": json.dumps(row["output"], ensure_ascii=False)}],
        }
    train_data = Dataset.from_list([convert(row) for row in source])
    validation_data = Dataset.from_list([convert(row) for row in read_jsonl(cfg["data"]["validation_path"])])
    revision = cfg["project"].get("model_revision")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if not tokenizer.chat_template:
        raise ValueError(f"{model_name} has no tokenizer chat template; choose an instruction/chat model")
    if tokenizer.eos_token is None:
        raise ValueError(f"{model_name} has no EOS token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, **quantization_options(cfg["quantization"])
    )
    if not torch.cuda.is_available():
        print("CUDA is unavailable. CPU training may be very slow; try --max-steps 1 first.")
    output_dir = run_directory(cfg) / "adapter"
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(settings["num_train_epochs"]),
        max_steps=args.max_steps if args.max_steps is not None else -1,
        per_device_train_batch_size=int(settings["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(settings["gradient_accumulation_steps"]),
        learning_rate=float(settings["learning_rate"]),
        max_length=int(settings["max_seq_length"]),
        completion_only_loss=True,
        eos_token=tokenizer.eos_token,
        eval_strategy="epoch" if args.max_steps is None else "no",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=int(cfg["project"]["seed"]),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=bool(settings["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_eval_batch_size=1,
        dataset_num_proc=1,
    )
    lora = LoraConfig(
        r=int(settings["lora_r"]),
        lora_alpha=int(settings["lora_alpha"]),
        lora_dropout=float(settings["lora_dropout"]),
        target_modules=lora_target_modules(settings["lora_target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=train_data,
        eval_dataset=validation_data, peft_config=lora, processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(experiment_identity(cfg), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
