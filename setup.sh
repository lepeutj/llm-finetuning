#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${repo_root}/.venv"
python_bin="${PYTHON_BIN:-python3}"
torch_version="${TORCH_VERSION:-2.6.0}"
backend="cuda"
recreate=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            backend="${2:-}"
            shift 2
            ;;
        --recreate)
            recreate=true
            shift
            ;;
        *)
            echo "Usage: bash setup.sh [--backend cuda|cpu] [--recreate]" >&2
            exit 2
            ;;
    esac
done

if [[ "${backend}" != "cuda" && "${backend}" != "cpu" ]]; then
    echo "--backend must be cuda or cpu." >&2
    exit 2
fi

command -v "${python_bin}" >/dev/null || {
    echo "${python_bin} was not found. Install Python 3.10, 3.11, or 3.12." >&2
    exit 1
}
"${python_bin}" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' || {
    echo "Python 3.10, 3.11, or 3.12 is required." >&2
    exit 1
}

if [[ "${backend}" == "cuda" ]]; then
    command -v nvidia-smi >/dev/null || {
        echo "nvidia-smi was not found. Install an NVIDIA driver or use --backend cpu." >&2
        exit 1
    }
    nvidia-smi >/dev/null || {
        echo "The NVIDIA driver cannot access the GPU." >&2
        exit 1
    }
    torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
else
    torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
fi

if [[ "${recreate}" == true ]]; then
    rm -rf -- "${venv_dir}"
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    "${python_bin}" -m venv "${venv_dir}" || {
        echo "Could not create .venv. On Debian/Ubuntu, install python3-venv." >&2
        exit 1
    }
fi

venv_python="${venv_dir}/bin/python"
"${venv_python}" -m pip install --upgrade pip
"${venv_python}" -m pip install --force-reinstall \
    "torch==${torch_version}" --index-url "${torch_index_url}"
"${venv_python}" -m pip install -r "${repo_root}/requirements.txt"
"${venv_python}" -m pip check

cd "${repo_root}"
"${venv_python}" -m src.prepare_dataset
if [[ "${backend}" == "cuda" ]]; then
    "${venv_python}" -m src.check_setup
else
    "${venv_python}" -m src.check_setup --quantization none
fi

cat <<EOF

Setup complete for ${backend}.
Activate with: source .venv/bin/activate
Run a quick test with: bash run_experiment.sh smoke
EOF
