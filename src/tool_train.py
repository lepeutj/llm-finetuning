"""Train a QLoRA adapter to select tools and produce schema-valid arguments."""
import argparse
import json
from .tool_runtime import add_arguments, followup_messages, identity, lora_target_modules, messages, quantization_options, read_rows, run_dir, settings
from .training_artifacts import save_training_artifacts

def main():
    parser = argparse.ArgumentParser(description=__doc__); add_arguments(parser)
    parser.add_argument("--max-train-samples", type=int); parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(); cfg = settings(args); train = read_rows(cfg, "train")
    if args.max_train_samples: train = train[:args.max_train_samples]
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer
    set_seed(cfg["project"]["seed"]); name = cfg["project"]["model_name"]; revision = cfg["project"].get("model_revision")
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    def convert(rows):
        samples = []
        for row in rows:
            samples.append({"prompt": messages(row), "completion": [{"role": "assistant", "content": json.dumps(row["expected_action"], ensure_ascii=False)}]})
            if row["expected_action"]["type"] == "tool_call" and len(row["trajectory"]) >= 4:
                final = row["trajectory"][-1]["content"]
                samples.append({"prompt": followup_messages(row), "completion": [{"role": "assistant", "content": json.dumps(final, ensure_ascii=False)}]})
        return samples
    train_data = Dataset.from_list(convert(train)); validation = Dataset.from_list(convert(read_rows(cfg, "validation")))
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, **quantization_options(cfg["quantization"]))
    model.config.use_cache = False; s = cfg["training"]; output = run_dir(cfg) / "adapter"
    training_args = SFTConfig(output_dir=str(output), num_train_epochs=s["num_train_epochs"], max_steps=args.max_steps or -1,
        per_device_train_batch_size=s["per_device_train_batch_size"], gradient_accumulation_steps=s["gradient_accumulation_steps"],
        learning_rate=s["learning_rate"], max_length=s["max_seq_length"], completion_only_loss=True,
        eos_token=tokenizer.eos_token, eval_strategy="no" if args.max_steps else "epoch", save_strategy="no", logging_steps=10,
        report_to="none", seed=cfg["project"]["seed"], fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), gradient_checkpointing=s["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False}, per_device_eval_batch_size=1, dataset_num_proc=1)
    lora = LoraConfig(r=s["lora_r"], lora_alpha=s["lora_alpha"], lora_dropout=s["lora_dropout"],
                      target_modules=lora_target_modules(s["lora_target_modules"]), bias="none", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=validation,
                         peft_config=lora, processing_class=tokenizer)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train(); trainer.model.save_pretrained(output); tokenizer.save_pretrained(output)
    trainable, total = trainer.model.get_nb_trainable_parameters()
    summary = {"trainer_metrics": train_result.metrics, "train_examples": len(train_data),
               "validation_examples": len(validation), "trainable_parameters": trainable,
               "total_parameters": total,
               "trainable_parameter_percent": 100 * trainable / total,
               "peak_gpu_memory_gib": (torch.cuda.max_memory_allocated() / 1024 ** 3
                                       if torch.cuda.is_available() else None)}
    save_training_artifacts(output, trainer.state.log_history, summary)
    (output / "experiment_manifest.json").write_text(json.dumps({"identity": identity(cfg), "smoke_test": bool(args.max_steps or args.max_train_samples)}, indent=2)+"\n", encoding="utf-8")
    print(f"Saved adapter and training curves to {output}")

if __name__ == "__main__": main()
