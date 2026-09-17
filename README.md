# Qwen2.5 3B LoRA and QLoRA: French information extraction

This experiment asks [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) to convert a French employment sentence into a five-field JSON object. It measures the untouched instruction model, trains a LoRA adapter, then evaluates both on the same held-out examples. You can switch to [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) in `config.yaml` for a lighter run. Documentation and instructions are in English; the input sentences remain French because that is the task being studied.

```text
Input:  Paul Martin est ingénieur chez Safran à Bordeaux depuis 2021.
Output: {"name":"Paul Martin","job":"ingénieur","company":"Safran","city":"Bordeaux","since":2021}
```

## Requirements and quick start

Use Python 3.10 or newer. The 3B model needs several GB just for weights in half precision; inference and especially training need additional memory. The defaults target a **single NVIDIA GPU with about 8 GB of VRAM**, using NF4, a microbatch of one, short sequences, and gradient checkpointing. This is a starting configuration, not a guarantee: free VRAM, GPU architecture, drivers, and system RAM all matter. If you mean 8 GB of **system RAM** rather than GPU VRAM, a 3B run may not fit comfortably even with 4-bit GPU loading. CPU inference is possible in `none` mode but slow, and CPU training may be impractically slow. The first run downloads model weights and needs Internet access. Check the [3B model license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) before redistributing weights or using them commercially.

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install bitsandbytes
python -m src.prepare_dataset
python -m src.check_setup
python -m src.baseline
python -m src.train
python -m src.evaluate
python -m src.compare_runs
```

Use any installed Python 3.10+ if 3.11 is unavailable. For CUDA, install the appropriate [PyTorch build](https://pytorch.org/get-started/locally/) **before** `requirements.txt` and confirm `python -m src.check_setup` sees the GPU. On macOS/Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`. A default pip installation may select a CPU PyTorch build.

## Model precision and quantization

The default `quantization.mode` is `nf4`. Install `bitsandbytes` for `int8`, `nf4`, or `fp4`. This demo requires a CUDA GPU for those modes and fails early with a clear error otherwise. For a limited-VRAM GPU, start with the short smoke run before attempting all 600 examples. `none` is the simplest precision reference but consumes substantially more memory.

| Mode | Base weights | Training method | Main tradeoff |
| --- | --- | --- | --- |
| `none` | FP16 on CUDA; FP32 on CPU | LoRA | Simplest reference, highest memory use |
| `int8` | bitsandbytes LLM.int8 | 8-bit base + LoRA | Lower memory with modest quantization |
| `nf4` | bitsandbytes 4-bit NormalFloat | QLoRA | Lowest practical training memory; recommended 4-bit mode |
| `fp4` | bitsandbytes 4-bit floating point | QLoRA | Alternative 4-bit format for comparison |

`quantization.double_quant` further quantizes metadata in 4-bit modes to save memory. Compute operations still use FP16 or BF16 where supported. The quantized base stays frozen; only LoRA weights are trained. Modes are loaded through Hugging Face Transformers, PEFT, TRL, and bitsandbytes. Each configuration gets its own directory under `results/`; manifests reject mixed datasets, models, or quantization settings.

All model commands accept `--model` and `--quantization`, so you can test several Hugging Face checkpoints without editing code. Start with the Qwen2.5 instruction family, which uses compatible chat templates and LoRA module names:

```powershell
python -m src.baseline --model Qwen/Qwen2.5-1.5B-Instruct --quantization nf4
python -m src.train --model Qwen/Qwen2.5-1.5B-Instruct --quantization nf4
python -m src.evaluate --model Qwen/Qwen2.5-1.5B-Instruct --quantization nf4
python -m src.compare_runs
```

Other instruction models can be supplied by Hugging Face ID if their tokenizer has a chat template and their attention layers match `training.lora_target_modules`. This code does not enable remote model code. When changing model families, check their license and update the target modules if necessary. Keep the same `--model` and `--quantization` values across baseline, training, and evaluation.

For a quick pipeline check, run `python -m src.baseline --limit 3`, `python -m src.train --max-train-samples 16 --max-steps 1`, then `python -m src.evaluate`. This checks wiring, **not model quality**. Repeat the full commands above for an actual comparison; training again replaces the adapter, and baseline without `--limit` scores all 100 test examples. If there is no CUDA GPU, add `--model Qwen/Qwen2.5-1.5B-Instruct --quantization none` to each command for a CPU smoke run.

Try one sentence after training:

```powershell
python -m src.inference "Paul Martin est ingénieur chez Safran à Bordeaux depuis 2021."
python -m src.inference --base "Paul Martin est ingénieur chez Safran à Bordeaux depuis 2021."
```

## What each command does

1. `prepare_dataset` creates deterministic JSONL files from templates and vocabulary in `src/prepare_dataset.py`. There are 600 train, 100 validation, and 100 test examples by default. Full names never cross splits. Templates, companies, cities, and jobs are shared, so this tests new combinations of familiar patterns rather than broad real-world generalization.
2. `baseline` applies the model's chat template and requests one JSON object. It saves raw responses and baseline metrics in that configuration's result directory before training.
3. `train` uses TRL supervised fine-tuning with PEFT LoRA. Prompt and completion are separate, so loss is computed on the assistant's JSON completion. It saves adapter weights and tokenizer files in that configuration's `adapter/` directory.
4. `evaluate` loads the original model plus the adapter, generates responses for the same test rows, and writes `lora_predictions.jsonl` and `comparison.json` in the same result directory. It rejects baseline predictions that do not match the current test set.

`config.yaml` documents every parameter. Paths are relative to the repository root. `project.seed` controls the synthetic split and training randomness. `project.model_revision` can pin a Hugging Face commit for repeatable model loading; the equivalent CLI option is `--revision`. `generation.max_new_tokens` caps each response; `generation.do_sample: false` selects greedy decoding. In training, `per_device_train_batch_size` is the microbatch and `gradient_accumulation_steps` simulates a larger batch. Effective batch size is their product times device count. `num_train_epochs` sets passes through the training set; `learning_rate` controls adapter update size; `max_seq_length` caps combined prompt and answer length. LoRA's `r` sets low-rank capacity, `alpha / r` scales the update, `dropout` regularizes it, and `lora_target_modules` selects attention projections. Higher rank and more target modules add trainable parameters and memory use.

## Metrics

`Valid JSON` is the fraction of responses that parse as a JSON object with **exactly** five required keys and correct value types, including integer `since`. `Exact match` requires every value to equal the reference. `Name`, `Job`, `Company`, `City`, and `Since F1` score each field as one exact categorical prediction per example. A missing or malformed response gets zero for every field. `Global F1` is the micro average over five fields; with one prediction and one reference per field, this equals field accuracy. Matching is case-sensitive and does not normalize accents or whitespace. The generated score file is the source of truth; no results are entered manually here.

## Methods for adapting an LLM

| Method | What changes | When it is useful |
| --- | --- | --- |
| Prompting / few-shot examples | No weights; instructions and examples are added to the prompt | First baseline and small experiments |
| Retrieval-augmented generation (RAG) | No weights; external documents are retrieved at inference time | Answers need changing facts or citations |
| Full supervised fine-tuning | All model weights are updated from input/output pairs | Large datasets and ample compute are available |
| LoRA / PEFT | Small low-rank matrices are trained while base weights stay frozen | Task adaptation with lower storage and training memory; used here |
| QLoRA | LoRA with a quantized frozen base model | Lower GPU memory is needed, with extra quantization complexity |
| Preference tuning (for example DPO) | Model learns from preferred versus rejected answers | Behavior and style preferences are the target |

This demo uses supervised fine-tuning through LoRA. The baseline may already solve much of this simple synthetic task; LoRA is not guaranteed to improve the held-out score. Any improvement should be checked on human-written sentences before making a broader claim.

Implementation references: [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) explains prompt/completion training and completion-only loss; [PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora) describes rank, scaling, dropout, and target modules; [Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes) covers 8-bit and 4-bit loading and hardware support.

## Project layout

```text
config.yaml                 Experiment parameters
data/train.jsonl            Synthetic training examples
data/validation.jsonl       Validation examples for training
data/test.jsonl             Held-out comparison examples
src/prepare_dataset.py      Reproducible data generation
src/baseline.py             Untouched Qwen predictions
src/train.py                LoRA supervised training
src/evaluate.py             Side-by-side metrics
src/inference.py            Single-sentence prediction
src/check_setup.py          Local hardware and dependency report
src/compare_runs.py         Summary across completed experiments
tests/                      Fast tests without model downloads
results/                    Local outputs (ignored by Git)
```

The data is synthetic and formulaic. It can establish that the pipeline runs and expose simple extraction errors, but it cannot establish production reliability. Use validation for tuning and reserve the test set for final comparison. Before production use, add human-written test data, reliability and latency measurements, failure handling, and a model deployment plan. The fast unit tests and GitHub Actions workflow do not download model weights; the real GPU smoke run remains a separate check.

## Publish from your computer

The repository contains only code and synthetic data; generated model files under `results/` and local environments are ignored. After creating an empty GitHub repository, run the following from this directory:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/llm-finetuning-demo.git
git push -u origin main
```

The local repository is committed and ready for that push. GitHub may prompt you to authenticate. Check the 3B model's separate license before publishing any derived weights; model weights are not in this repository.
