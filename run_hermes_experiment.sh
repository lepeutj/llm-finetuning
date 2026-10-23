#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${root}/.venv/bin/python"
mode="${1:-smoke}"
model="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
quantization="${QUANTIZATION:-nf4}"
config="${root}/hermes_config.yaml"
args=(--config "${config}" --model "${model}" --quantization "${quantization}")
[[ -x "${python_bin}" ]] || { echo "Missing .venv; run setup.sh first" >&2; exit 1; }
cd "${root}"
"${python_bin}" -m src.prepare_hermes_dataset --config "${config}"
case "${mode}" in
  smoke)
    "${python_bin}" -m src.tool_baseline "${args[@]}" --limit 20
    "${python_bin}" -m src.tool_train "${args[@]}" --max-train-samples 64 --max-steps 2
    "${python_bin}" -m src.tool_evaluate "${args[@]}" --allow-partial ;;
  full)
    "${python_bin}" -m src.tool_baseline "${args[@]}"
    "${python_bin}" -m src.tool_train "${args[@]}"
    "${python_bin}" -m src.tool_evaluate "${args[@]}" ;;
  *) echo "Usage: bash run_hermes_experiment.sh [smoke|full]" >&2; exit 2 ;;
esac
