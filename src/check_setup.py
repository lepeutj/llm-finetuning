"""Report local Python, PyTorch, CUDA, and quantization readiness."""

import importlib
import importlib.metadata
import platform
import sys

from .common import add_run_arguments, run_directory, selected_config


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    args = parser.parse_args()
    cfg = selected_config(args)
    print(f"Python: {platform.python_version()}")
    print(f"Model: {cfg['project']['model_name']}")
    print(f"Quantization: {cfg['quantization']['mode']}")
    print(f"Results: {run_directory(cfg)}")
    ready = True
    if sys.version_info < (3, 10):
        print("FAIL: Python 3.10 or newer is required for the specified dependencies.")
        ready = False
    packages = ("torch", "transformers", "datasets", "peft", "trl", "accelerate")
    loaded = {}
    for package in packages:
        try:
            loaded[package] = importlib.import_module(package)
            version = importlib.metadata.version(package)
            print(f"{package}: {version} (import OK)")
        except Exception as error:  # Diagnostic command: report binary/API import failures cleanly.
            print(f"{package}: FAIL to import: {error}")
            ready = False
    if "torch" not in loaded:
        raise SystemExit(1)
    torch = loaded["torch"]

    try:
        from peft import LoraConfig  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: F401
        from trl import SFTConfig, SFTTrainer  # noqa: F401
        print("Training API imports: OK")
    except Exception as error:
        print(f"Training API imports: FAIL: {error}")
        ready = False

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda or 'none (CPU-only build)'}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(f"CUDA GPU: {properties.name} ({properties.total_memory / (1024 ** 3):.1f} GiB VRAM)")
        capability = torch.cuda.get_device_capability(0)
        print(f"Compute capability: {capability[0]}.{capability[1]}")
        try:
            probe = torch.ones(1, device="cuda")
            print(f"CUDA tensor test: OK ({probe.device})")
            del probe
        except RuntimeError as error:
            print(f"CUDA tensor test: FAIL: {error}")
            ready = False
    else:
        print("CUDA GPU: unavailable")
        if torch.version.cuda is None:
            print("Install a CUDA-enabled PyTorch wheel in this Python environment before using NF4.")
        else:
            print("PyTorch has CUDA support but cannot access the GPU; check the NVIDIA driver and environment.")
    if cfg["quantization"]["mode"] != "none":
        try:
            importlib.import_module("bitsandbytes")
            print(f"bitsandbytes: {importlib.metadata.version('bitsandbytes')} (import OK)")
        except Exception as error:
            print(f"bitsandbytes: FAIL to import: {error}")
            ready = False
        if not torch.cuda.is_available():
            print("FAIL: This demo requires CUDA for quantized modes; use --quantization none on CPU.")
            ready = False
        elif cfg["quantization"]["mode"] == "int8" and torch.cuda.get_device_capability(0) < (7, 5):
            print("FAIL: LLM.int8 requires a Turing-class NVIDIA GPU or newer.")
            ready = False
        elif cfg["quantization"]["mode"] in ("nf4", "fp4") and torch.cuda.get_device_capability(0) < (6, 0):
            print("FAIL: NF4/FP4 requires a Pascal-class NVIDIA GPU or newer.")
            ready = False
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
