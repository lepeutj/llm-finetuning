# Low-compute LLM fine-tuning: tool calling and structured extraction

This repository provides reproducible QLoRA experiments for small instruction models. The primary experiment trains a model to select tools and emit schema-valid arguments from agent trajectories. The original structured-extraction benchmark remains available as a second experiment.

## Tool-calling experiment

Each generated trajectory stores the available JSON Schema tools, user request, expected action, simulated tool result, and final assistant response. The router learns one of three actions:

```json
{"type":"tool_call","name":"get_weather","arguments":{"city":"Paris","date":"tomorrow"}}
```

```json
{"type":"clarify","question":"Which city should I use?"}
```

```json
{"type":"respond","content":"JSON stands for JavaScript Object Notation."}
```

The bundled tools cover weather, currency conversion, web search, calculation, and file discovery. Their implementations are deterministic simulations: dataset creation and evaluation never invoke a live service or modify files. The generated splits contain 1,200 training, 150 validation, and 300 held-out test trajectories. Test requests use paraphrase templates absent from training.

Run a small end-to-end check:

```bash
MODEL_PROFILE=qwen-0.5b QUANTIZATION=nf4 bash run_tool_experiment.sh smoke
```

Run the complete benchmark:

```bash
MODEL_PROFILE=qwen-0.5b QUANTIZATION=nf4 bash run_tool_experiment.sh full
```

Metrics include valid action rate, action-type accuracy, tool selection accuracy, exact arguments, executable call rate, no-call accuracy, and overall exact match. Adapters and predictions are stored below `results/tool_calling/`.

Recompute metrics from saved predictions without loading the model:

```bash
.venv/bin/python -m src.tool_rescore --model Qwen/Qwen2.5-0.5B-Instruct --quantization nf4
```

Test one request after training:

```bash
.venv/bin/python -m src.tool_inference "What is the weather in Paris tomorrow?" \
  --model Qwen/Qwen2.5-0.5B-Instruct --quantization nf4
```

The trainer learns both the routing action before execution and the final response after receiving a tool result. Run `python -m src.tool_agent` to exercise that complete loop. Replace or extend `src/tool_catalog.py` and the data importer to train on application-specific traces; multi-step planning remains a future extension.

```bash
.venv/bin/python -m src.tool_agent "Convert 125 USD to EUR" \
  --model Qwen/Qwen2.5-0.5B-Instruct --quantization nf4
```

### Hermes dataset experiment with Qwen 1.5B

The external-data experiment downloads the Apache-2.0 [NousResearch Hermes Function Calling V1 dataset](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1). It strictly normalizes the `func_calling_singleturn` subset, rejects malformed calls and schema violations, creates deterministic train/validation/test splits, and mixes in clarification and direct-response examples. The resulting routing distribution is 60% Hermes calls and 40% balanced negatives.

```bash
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct QUANTIZATION=nf4 \
  bash run_hermes_experiment.sh smoke
```

After the smoke test succeeds:

```bash
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct QUANTIZATION=nf4 \
  bash run_hermes_experiment.sh full
```

The prepared files contain 500 training, 100 validation, and 150 test rows. The strict importer uses 450 source calls across the three splits and rejects multi-call, empty-tool, malformed, and schema-invalid source rows. It adds 300 controlled clarification and direct-response examples to obtain a 60/40 routing balance. An import report is written to `data/hermes/`, and results are isolated under `results/hermes/`. The dataset revision is pinned; rerunning the importer uses the Hugging Face cache and produces the same split for the same revision and seed.

## Structured-extraction experiment

This repository compares zero-shot prompting, three-shot prompting, and QLoRA on a held-out information-extraction task. The default model is [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct); `--model` also accepts another compatible Hugging Face model ID or a local Transformers model directory. The text passages are French and English. Instructions and documentation are in English.

The output schema is always `{"name": string, "job": string, "company": string, "city": string, "since": integer|null}`. The goal is to extract **current** employment. An earlier date or employer must not be mistaken for the current one. For example:

```text
Input:  Jean Dupont left Toulouse in 2020. Two years later, Jean Dupont
        joined Airbus in Nantes as a backend developer.
Output: {"name":"Jean Dupont","job":"backend developer","company":"Airbus","city":"Nantes","since":2022}
```

If the passage mentions only the year of a previous job, `since` must be `null`. These examples are synthetic and still have limited linguistic diversity. They are a controlled experiment, not a production benchmark.

## Try it without a local GPU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lepeutj/llm-finetuning/blob/main/notebooks/quickstart_colab.ipynb)

Open the notebook, select a **T4 GPU** runtime, and run all cells. It provisions Python 3.12 independently of Colab's system Python, installs the project in an isolated environment, uses the 0.5B Qwen model with NF4 quantization, runs the full pipeline with one training step, and displays the generated comparison. This is a real end-to-end smoke test; its three evaluation samples are too small to measure whether fine-tuning improves quality.

## Run on any Linux machine

The code is independent of a cloud provider. It runs on a local Linux workstation, GCP, AWS, Azure, RunPod, or another Linux host with Python 3.10–3.12. CUDA mode requires an NVIDIA driver visible through `nvidia-smi`; the Python environment contains its own pinned PyTorch and Hugging Face packages.

```bash
git clone https://github.com/lepeutj/llm-finetuning.git
cd llm-finetuning
bash setup.sh --backend cuda
source .venv/bin/activate
```

For a CPU-only wiring check:

```bash
bash setup.sh --backend cpu
QUANTIZATION=none MODEL_PROFILE=qwen-0.5b bash run_experiment.sh smoke
```

CPU training is expected to be slow. The setup script creates an isolated `.venv`, installs the hardware-specific PyTorch wheel, installs the pinned training stack, runs `pip check`, prepares the dataset, imports the actual training APIs, and tests the selected device. Rebuild an inconsistent environment with `bash setup.sh --backend cuda --recreate`.

### Model selection

Select a tested preset with `MODEL_PROFILE`:

| Profile | Hugging Face model | Suggested hardware |
| --- | --- | --- |
| `qwen-0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` | CPU smoke test or small GPU |
| `qwen-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | Recommended starting point |
| `qwen-3b` | `Qwen/Qwen2.5-3B-Instruct` | 8 GB+ GPU with NF4 |
| `smollm-1.7b` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Alternative small model |
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` | 16 GB+ GPU with NF4 |

Examples:

```bash
MODEL_PROFILE=qwen-0.5b bash run_experiment.sh smoke
MODEL_PROFILE=smollm-1.7b bash run_experiment.sh full
MODEL_PROFILE=mistral-7b QUANTIZATION=nf4 bash run_experiment.sh smoke
```

Any compatible local directory or Hugging Face model ID can override the preset:

```bash
MODEL_ID=/models/my-instruction-model bash run_experiment.sh smoke
```

The model must be supported by `AutoModelForCausalLM`, include a tokenizer chat template, and expose the configured LoRA target modules. Each model receives a separate result directory, and adapters are validated against the exact base model before evaluation.

### Optional GCP target

On GCP, use an Ubuntu Deep Learning VM with a T4 or L4, clone the same repository, and run the same `setup.sh` and `run_experiment.sh` commands. GCP is an execution target only; no project code depends on Google Cloud.

## Experiment commands

Start with the smoke run. It downloads the model, produces three zero-shot and few-shot predictions, performs one optimizer step, reloads the saved adapter, and creates a comparison marked as a smoke test:

```bash
bash run_experiment.sh smoke
```

Once that succeeds, launch the complete experiment:

```bash
bash run_experiment.sh full
```

The full runner executes these commands in order:

```bash
python -m src.baseline --split easy --strategy zero
python -m src.baseline --split hard --strategy zero
python -m src.baseline --split hard --strategy few
python -m src.train
python -m src.evaluate --split easy
python -m src.evaluate --split hard
python -m src.compare_runs
```

The hard split compares zero-shot, few-shot, and LoRA on the same 300 examples. The easy split compares zero-shot and LoRA on 100 examples. The few-shot baseline uses **three examples from the training split only**, one each for a missing year, a relative year, and a prior employer. The validation set is used during training; the test sets are never used for gradient updates or selecting training examples. Results, raw predictions, adapters, and manifests go to a unique directory under `results/` for each dataset, model, precision, and configuration.

For a fast wiring check, use the following commands instead. The resulting `comparison_hard.json` is marked `smoke_test: true`:

```bash
python -m src.baseline --split hard --strategy zero --limit 3
python -m src.baseline --split hard --strategy few --limit 3
python -m src.train --max-train-samples 16 --max-steps 1
python -m src.evaluate --split hard --allow-partial
```

A smoke test verifies model loading, data formatting, adapter saving, and scoring. It cannot establish whether fine-tuning helps. Repeat the full commands without limits for the experiment. `evaluate` refuses partial baselines or a smoke-test adapter unless `--allow-partial` is explicitly supplied.

## Data design and measurements

`prepare_dataset` deterministically generates 1,000 training, 150 validation, 300 hard test, and 100 easy test examples. Full names never cross splits. Hard passages contain earlier employers/locations/years, dates to calculate (`two years later`), current facts spread over sentences, and missing current start years. Each hard-test category has 75 examples. Hard-test sentence templates are held out from training and validation, though the underlying generation rules and value vocabulary remain shared. Every row contains `input`, `output`, and `tags` for analysis. About one quarter of training examples are easy, while the hard test is entirely hard.

`Valid JSON` requires one JSON object with exactly the five keys and their types; `since` accepts an integer or `null`. `Exact match` requires all five values to match. Per-field F1 treats each field value as an exact categorical prediction; missing or malformed responses score zero. `Global F1` is the micro average across fields and therefore equals field accuracy in this single-value setup. `since_missing_accuracy` reports how often a missing current start year is correctly returned as `null`; `since_present_accuracy` scores known years. Metrics are also broken down by tags such as `missing` and `relative`. Matching is case-sensitive and does not normalize accents or whitespace. Scores are produced from model predictions, never entered manually.

Run manifests include a pipeline version. When prompt formatting or metric semantics change, results from older runs are rejected; rerun the baselines and training commands to obtain a comparable experiment.

If zero-shot already performs as well as LoRA, that is a valid conclusion: this task may not justify training. Do not tune on the hard test. Use validation and, for a stronger claim, collect independently written passages and compare on those later.

## Model and training choices

`config.yaml` documents the settings. `project.seed` controls data generation and training randomness; `project.model_revision` can pin a Hugging Face commit. `generation.max_new_tokens` limits the JSON response and `do_sample: false` selects greedy decoding. Training uses TRL supervised fine-tuning with a separate conversational prompt and completion, so the loss is on the assistant JSON response. PEFT freezes the base and trains LoRA weights in the selected attention projections. `num_train_epochs`, `learning_rate`, `lora_r`, `lora_alpha`, `lora_dropout`, batch size, gradient accumulation, and checkpointing can be adjusted in the config. Only the adapter and tokenizer are saved, not a duplicate of the base model.

`--model`, `--revision`, and `--quantization` override the config on all model commands. Use **identical overrides** for baseline, training, and evaluation. For example:

```bash
model='Qwen/Qwen2.5-1.5B-Instruct'
python -m src.baseline --model $model --quantization nf4 --split hard --strategy zero
python -m src.baseline --model $model --quantization nf4 --split hard --strategy few
python -m src.train --model $model --quantization nf4
python -m src.evaluate --model $model --quantization nf4 --split hard
```

The modes are `none` (FP16 on CUDA or FP32 on CPU), `int8` (bitsandbytes LLM.int8), and `nf4`/`fp4` (4-bit base plus LoRA). Quantized modes in this demo require CUDA. Nested quantization is controlled by `quantization.double_quant`. Another model must have a Transformers causal-LM loader and tokenizer chat template; different architectures may require changing `training.lora_target_modules` to their layer names or `all-linear` (which uses more memory). A GGUF file is not accepted by this loader. An adapter belongs to the exact base model it was trained for. Check a model's license before distributing derived weights. [TRL SFT](https://huggingface.co/docs/trl/sft_trainer), [PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora), and [Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes) document the underlying methods.

## Methods for adapting an LLM

| Method | What changes | When it is useful |
| --- | --- | --- |
| Prompting / few-shot examples | No weights; instructions and examples are added to the prompt | First baseline and small experiments |
| Retrieval-augmented generation (RAG) | No weights; external documents are retrieved at inference time | Answers need changing facts or citations |
| Full supervised fine-tuning | All model weights are updated from input/output pairs | Large datasets and ample compute are available |
| LoRA / PEFT | Small low-rank matrices are trained while base weights stay frozen | Task adaptation with lower storage and training memory; used here |
| QLoRA | LoRA with a quantized frozen base model | Lower GPU memory is needed, with extra quantization complexity |
| Preference tuning (for example DPO) | Model learns from preferred versus rejected answers | Behavior and style preferences are the target |

This demo uses supervised fine-tuning through LoRA and a quantized base model. Its baseline may already solve the task, so an improvement is not assumed.

## Scope and verification

`python -m unittest discover -s tests -v` checks deterministic generation, disjoint splits, held-out templates, null years, relative dates, scoring, and run separation without downloading model weights. GitHub Actions runs these checks on Python 3.10 and 3.11. Full CUDA model loading and training must be verified on the GPU computer; no improvement is claimed in advance. The synthetic test is useful for controlled comparison, but production use would require human-written evaluation data, latency and reliability checks, and deployment work.
