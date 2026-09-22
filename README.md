# LLM fine-tuning experiment: extraction with distractors and missing values

This repository compares zero-shot prompting, three-shot prompting, and QLoRA on a held-out information-extraction task. The default model is [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct); `--model` also accepts another compatible Hugging Face model ID or a local Transformers model directory. The text passages are French and English. Instructions and documentation are in English.

The output schema is always `{"name": string, "job": string, "company": string, "city": string, "since": integer|null}`. The goal is to extract **current** employment. An earlier date or employer must not be mistaken for the current one. For example:

```text
Input:  Jean Dupont left Toulouse in 2020. Two years later, Jean Dupont
        joined Airbus in Nantes as a backend developer.
Output: {"name":"Jean Dupont","job":"backend developer","company":"Airbus","city":"Nantes","since":2022}
```

If the passage mentions only the year of a previous job, `since` must be `null`. These examples are synthetic and still have limited linguistic diversity. They are a controlled experiment, not a production benchmark.

## Run on Google Cloud

The recommended first machine is an Ubuntu Deep Learning VM with one NVIDIA T4 (16 GB VRAM), four vCPUs, and at least 80 GB of disk. Qwen 1.5B with NF4 is deliberately conservative for this GPU. A T4 is widely available and sufficient for this experiment; use an L4 only when faster training is worth the additional cost. Confirm that the selected zone has GPU capacity and that the project has GPU quota.

From Cloud Shell or a computer with `gcloud`, this example creates the VM in Belgium. Change the project, zone, or machine name as needed:

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud compute instances create llm-finetuning \
  --zone=europe-west1-b \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=80GB \
  --metadata=install-nvidia-driver=True
```

Google's [Deep Learning VM images](https://cloud.google.com/deep-learning-vm/docs/images) include the ML stack and NVIDIA driver. GPU availability varies by [region and zone](https://cloud.google.com/compute/docs/regions-zones/gpu-regions-zones). After the VM starts, connect and install the repository in an isolated environment:

```bash
gcloud compute ssh llm-finetuning --zone=europe-west1-b
git clone https://github.com/lepeutj/llm-finetuning.git
cd llm-finetuning
bash setup_gcp.sh
source .venv/bin/activate
```

`setup_gcp.sh` first verifies `nvidia-smi`, then creates `.venv`, force-installs the CUDA-enabled PyTorch wheel, installs the pinned Hugging Face stack, runs `pip check`, prepares the dataset, imports the actual training APIs, and allocates a CUDA tensor. This prevents a system Python package or CPU-only PyTorch build from producing a misleading successful installation. `torchvision` and `torchaudio` are absent because this is a text-only project.

If the environment becomes inconsistent, rebuild only the local virtual environment:

```bash
bash setup_gcp.sh --recreate
```

If setup stops before package installation, fix `nvidia-smi` or the VM configuration first. If it reports `none (CPU-only build)`, recreate the environment with the command above. The VM should be stopped when it is idle to avoid ongoing compute charges.

## Small model choices

| Model | Suggested use | Expected tradeoff |
| --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | Fastest pipeline and training demonstration | Lowest memory use, but likely weaker extraction quality |
| `Qwen/Qwen2.5-1.5B-Instruct` | Recommended first complete experiment | Better balance for an 8 GB GPU |
| `Qwen/Qwen2.5-3B-Instruct` | Follow-up comparison | Better base capability, longer runs, tighter memory margin |

The Qwen2.5 instruction checkpoints use the same model family and an Apache 2.0 license. A useful second portfolio experiment is to run the same fixed train/test protocol at all three sizes. This separates gains from model scale, prompting, and QLoRA without introducing another dataset or scoring pipeline.

## Experiment commands

Run each baseline on the **full** held-out set before training:

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
