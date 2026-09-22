#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${repo_root}/.venv"
python_bin="${PYTHON_BIN:-python3}"
torch_version="${TORCH_VERSION:-2.6.0}"
torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

if [[ "${1:-}" == "--recreate" ]]; then
    rm -rf -- "${venv_dir}"
elif [[ $# -gt 0 ]]; then
    echo "Usage: bash setup_gcp.sh [--recreate]" >&2
    exit 2
fi

command -v "${python_bin}" >/dev/null || {
    echo "${python_bin} was not found. Use a GCP Deep Learning VM image with Python 3.10+." >&2
    exit 1
}
command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi was not found. Create the VM with an NVIDIA GPU and driver installed." >&2
    exit 1
}
nvidia-smi >/dev/null || {
    echo "The NVIDIA driver cannot access the GPU. Fix the VM driver before installing Python packages." >&2
    exit 1
}

"${python_bin}" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' || {
    echo "Python 3.10, 3.11, or 3.12 is required." >&2
    exit 1
}

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    "${python_bin}" -m venv "${venv_dir}" || {
        echo "Could not create .venv. On Ubuntu, install python3-venv and retry." >&2
        exit 1
    }
fi

venv_python="${venv_dir}/bin/python"
"${venv_python}" -m pip install --upgrade pip

# Force replacement in case the image's system packages exposed a CPU wheel or
# another CUDA build with the same public PyTorch version.
"${venv_python}" -m pip install --force-reinstall \
    "torch==${torch_version}" --index-url "${torch_index_url}"
"${venv_python}" -m pip install -r "${repo_root}/requirements.txt"
"${venv_python}" -m pip check

cd "${repo_root}"
"${venv_python}" -m src.prepare_dataset
"${venv_python}" -m src.check_setup

cat <<EOF

Setup complete.
Activate the environment with: source .venv/bin/activate
Then run the smoke experiment documented in README.md.
EOF
