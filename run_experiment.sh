#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${repo_root}/.venv/bin/python"
mode="${1:-smoke}"

if [[ ! -x "${python_bin}" ]]; then
    echo "Missing .venv. Run: bash setup_gcp.sh" >&2
    exit 1
fi

cd "${repo_root}"
"${python_bin}" -m src.check_setup

case "${mode}" in
    smoke)
        "${python_bin}" -m src.baseline --split hard --strategy zero --limit 3
        "${python_bin}" -m src.baseline --split hard --strategy few --limit 3
        "${python_bin}" -m src.train --max-train-samples 16 --max-steps 1
        "${python_bin}" -m src.evaluate --split hard --allow-partial
        ;;
    full)
        "${python_bin}" -m src.baseline --split easy --strategy zero
        "${python_bin}" -m src.baseline --split hard --strategy zero
        "${python_bin}" -m src.baseline --split hard --strategy few
        "${python_bin}" -m src.train
        "${python_bin}" -m src.evaluate --split easy
        "${python_bin}" -m src.evaluate --split hard
        "${python_bin}" -m src.compare_runs
        ;;
    *)
        echo "Usage: bash run_experiment.sh [smoke|full]" >&2
        exit 2
        ;;
esac
