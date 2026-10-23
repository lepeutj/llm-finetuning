#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${repo_root}/.venv/bin/python"
mode="${1:-smoke}"
profile="${MODEL_PROFILE:-qwen-1.5b}"
quantization="${QUANTIZATION:-nf4}"

case "${profile}" in
    qwen-0.5b) model_id="Qwen/Qwen2.5-0.5B-Instruct" ;;
    qwen-1.5b) model_id="Qwen/Qwen2.5-1.5B-Instruct" ;;
    qwen-3b) model_id="Qwen/Qwen2.5-3B-Instruct" ;;
    smollm-1.7b) model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct" ;;
    mistral-7b) model_id="mistralai/Mistral-7B-Instruct-v0.3" ;;
    *)
        echo "Unknown MODEL_PROFILE: ${profile}" >&2
        echo "Choose qwen-0.5b, qwen-1.5b, qwen-3b, smollm-1.7b, or mistral-7b." >&2
        exit 2
        ;;
esac
model_id="${MODEL_ID:-${model_id}}"
model_args=(--model "${model_id}" --quantization "${quantization}")

if [[ ! -x "${python_bin}" ]]; then
    echo "Missing .venv. Run: bash setup.sh" >&2
    exit 1
fi

cd "${repo_root}"
echo "Model: ${model_id} (${profile})"
echo "Quantization: ${quantization}"
"${python_bin}" -m src.check_setup "${model_args[@]}"

case "${mode}" in
    smoke)
        "${python_bin}" -m src.baseline "${model_args[@]}" --split hard --strategy zero --limit 3
        "${python_bin}" -m src.baseline "${model_args[@]}" --split hard --strategy few --limit 3
        "${python_bin}" -m src.train "${model_args[@]}" --max-train-samples 16 --max-steps 1
        "${python_bin}" -m src.evaluate "${model_args[@]}" --split hard --allow-partial
        ;;
    full)
        "${python_bin}" -m src.baseline "${model_args[@]}" --split easy --strategy zero
        "${python_bin}" -m src.baseline "${model_args[@]}" --split hard --strategy zero
        "${python_bin}" -m src.baseline "${model_args[@]}" --split hard --strategy few
        "${python_bin}" -m src.train "${model_args[@]}"
        "${python_bin}" -m src.evaluate "${model_args[@]}" --split easy
        "${python_bin}" -m src.evaluate "${model_args[@]}" --split hard
        "${python_bin}" -m src.compare_runs
        ;;
    *)
        echo "Usage: bash run_experiment.sh [smoke|full]" >&2
        exit 2
        ;;
esac
