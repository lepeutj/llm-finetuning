#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${root}/.venv/bin/python"
mode="${1:-smoke}"
profile="${MODEL_PROFILE:-qwen-0.5b}"
quantization="${QUANTIZATION:-nf4}"
case "${profile}" in
  qwen-0.5b) model="Qwen/Qwen2.5-0.5B-Instruct" ;;
  qwen-1.5b) model="Qwen/Qwen2.5-1.5B-Instruct" ;;
  qwen-3b) model="Qwen/Qwen2.5-3B-Instruct" ;;
  smollm-1.7b) model="HuggingFaceTB/SmolLM2-1.7B-Instruct" ;;
  *) echo "Unknown MODEL_PROFILE: ${profile}" >&2; exit 2 ;;
esac
model="${MODEL_ID:-${model}}"; args=(--model "${model}" --quantization "${quantization}")
[[ -x "${python_bin}" ]] || { echo "Missing .venv; run setup.sh first" >&2; exit 1; }
cd "${root}"; "${python_bin}" -m src.prepare_tool_dataset
case "${mode}" in
  smoke)
    "${python_bin}" -m src.tool_baseline "${args[@]}" --limit 20
    "${python_bin}" -m src.tool_train "${args[@]}" --max-train-samples 64 --max-steps 2
    "${python_bin}" -m src.tool_evaluate "${args[@]}" --allow-partial ;;
  full)
    "${python_bin}" -m src.tool_baseline "${args[@]}"
    "${python_bin}" -m src.tool_train "${args[@]}"
    "${python_bin}" -m src.tool_evaluate "${args[@]}" ;;
  *) echo "Usage: bash run_tool_experiment.sh [smoke|full]" >&2; exit 2 ;;
esac
