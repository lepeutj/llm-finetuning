"""Shared configuration, prompting, and model loading."""

import json
import hashlib
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("name", "job", "company", "city", "since")
# Increment when prompt formatting or scoring semantics change so old results
# cannot be silently compared with predictions from a different pipeline.
PIPELINE_VERSION = 2
SYSTEM_PROMPT = (
    "Extract the person's current employment from the passage. Ignore previous jobs, "
    "previous locations, and unrelated dates. "
    "Return exactly one JSON object with the keys name, job, company, city, since. "
    "Use an integer for since when the start year is stated or can be calculated. "
    "If the start year is not provided, use null. Do not include explanations or Markdown."
)


def config(path=None):
    with (Path(path) if path else ROOT / "config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def path(value):
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_jsonl(value):
    with path(value).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(value, rows):
    target = path(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def experiment_identity(settings):
    return {
        "pipeline_version": PIPELINE_VERSION,
        "model_name": settings["project"]["model_name"],
        "model_revision": settings["project"].get("model_revision"),
        "quantization": settings["quantization"],
        "seed": settings["project"]["seed"],
        "data_sha256": {
            split: hashlib.sha256(path(settings["data"][f"{split}_path"]).read_bytes()).hexdigest()
            for split in ("train", "validation", "test", "easy_test")
        },
    }


def add_run_arguments(parser):
    parser.add_argument("--model", help="Hugging Face instruction model ID; overrides config.yaml")
    parser.add_argument("--revision", help="Hugging Face model revision or commit; overrides config.yaml")
    parser.add_argument("--quantization", choices=("none", "int8", "nf4", "fp4"),
                        help="Base-weight precision; overrides config.yaml")


def selected_config(args):
    settings = config()
    if args.model:
        settings["project"]["model_name"] = args.model
        if not args.revision:
            settings["project"]["model_revision"] = None
    if args.revision:
        settings["project"]["model_revision"] = args.revision
    if args.quantization:
        settings["quantization"]["mode"] = args.quantization
    return settings


def run_directory(settings):
    model = settings["project"]["model_name"]
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[:60]
    identity = experiment_identity(settings)
    experiment = {"identity": identity, "training": settings["training"],
                  "generation": settings["generation"]}
    digest = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return path(settings["training"]["results_dir"]) / f"{slug}-{settings['quantization']['mode']}-{digest}"


def lora_target_modules(value):
    """Accept either named layers or PEFT's architecture-independent option."""
    if value == "all-linear":
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return value
    raise ValueError("training.lora_target_modules must be a nonempty list or 'all-linear'")


def messages(sentence, examples=None):
    # Some instruction models do not accept a separate system role. A single
    # user turn keeps training and inference prompts portable across families.
    turns = []
    for index, example in enumerate(examples or []):
        prefix = f"{SYSTEM_PROMPT}\n\n" if index == 0 else ""
        turns.extend([
            {"role": "user", "content": f"{prefix}Passage:\n{example['input']}"},
            {"role": "assistant", "content": json.dumps(example["output"], ensure_ascii=False)},
        ])
    prefix = f"{SYSTEM_PROMPT}\n\n" if not turns else ""
    turns.append({"role": "user", "content": f"{prefix}Passage:\n{sentence}"})
    return turns


def quantization_options(settings):
    """Build Hugging Face loading options for the selected precision mode."""
    import torch

    mode = settings.get("mode", "none")
    if mode == "none":
        return {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
    if mode not in ("int8", "nf4", "fp4"):
        raise ValueError(f"Unknown quantization mode: {mode}. Choose none, int8, nf4, or fp4")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{mode} requires a CUDA GPU in this demo; use quantization.mode: none on CPU")
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Install bitsandbytes to use a quantized mode") from error
    from transformers import BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if mode == "int8":
        quantization = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=mode,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(settings.get("double_quant", True)),
        )
    return {"quantization_config": quantization, "device_map": {"": torch.cuda.current_device()}, "torch_dtype": compute_dtype}


def load_model(model_name, adapter=None, quantization=None, revision=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if not tokenizer.chat_template:
        raise ValueError(f"{model_name} has no tokenizer chat template; choose an instruction/chat model")
    options = quantization_options(quantization or {"mode": "none"})
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, **options)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    if (quantization or {}).get("mode", "none") == "none":
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, sentence, settings, examples=None):
    import torch

    text = tokenizer.apply_chat_template(messages(sentence, examples), tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=int(settings["max_new_tokens"]),
            do_sample=bool(settings["do_sample"]),
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
