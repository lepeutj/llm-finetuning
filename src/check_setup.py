"""Report local Python, PyTorch, CUDA, and quantization readiness."""

import importlib.util
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
    for package in ("torch", "transformers", "datasets", "peft", "trl", "accelerate"):
        installed = importlib.util.find_spec(package) is not None
        print(f"{package}: {'installed' if installed else 'MISSING'}")
        ready &= installed
    if not importlib.util.find_spec("torch"):
        raise SystemExit(1)
    import torch

    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(f"CUDA GPU: {properties.name} ({properties.total_memory / (1024 ** 3):.1f} GiB VRAM)")
        capability = torch.cuda.get_device_capability(0)
        print(f"Compute capability: {capability[0]}.{capability[1]}")
    else:
        print("CUDA GPU: unavailable")
    if cfg["quantization"]["mode"] != "none":
        installed = importlib.util.find_spec("bitsandbytes") is not None
        print(f"bitsandbytes: {'installed' if installed else 'MISSING'}")
        ready &= installed
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
