# vLLM Custom Build Environment

Per-architecture vLLM builds optimized for specific NVIDIA GPU targets. Built with Flox + Nix.

## CUDA Compatibility

All variants are built against **CUDA 12.9** and require **NVIDIA driver 560+**.

- **Forward compatibility**: CUDA 12.9 builds work with any driver that supports CUDA 12.9 or later
- **No cross-major compatibility**: CUDA 12.x builds are **not** compatible with CUDA 11.x or 13.x runtimes
- **Check your driver**: Run `nvidia-smi` — the "CUDA Version" in the top-right shows the maximum CUDA version your driver supports

```bash
# Verify your driver supports CUDA 12.9
nvidia-smi
# Look for "CUDA Version: 12.9" or higher in the output
```

## Overview

Standard vLLM wheels from PyPI compile CUDA kernels for all compute capabilities. This project creates **per-SM builds** from source, resulting in:

- **Smaller binaries** — Only include CUDA kernels for your target GPU architecture
- **Optimized kernels** — PagedAttention, FlashAttention, CUTLASS scaled_mm, Marlin quantization, MoE kernels compiled for your exact SM
- **Targeted deployment** — Install exactly what your hardware needs

## Version Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| vLLM | 0.11.2 | From nixpkgs |
| PyTorch | 2.9.1 | Pre-built wheel (not compiled from source) |
| CUTLASS | 4.2.1 + 3.9.0 | v4.2.1 primary, v3.9.0 for FlashMLA Blackwell |
| CUDA Toolkit | 12.9 | Via nixpkgs `cudaPackages_12_9`, requires driver 560+ |
| Python | 3.12 | Via nixpkgs |
| Nixpkgs | [`ed142ab`](https://github.com/NixOS/nixpkgs/tree/ed142ab1b3a092c4d149245d0c4126a5d7ea00b0) | Pinned revision |

## Build Matrix

### x86_64-linux (8 variants)

| SM | Architecture | GPUs | Variant |
|----|-------------|------|---------|
| SM70 | Volta | V100 | `vllm-python312-cuda12_9-sm70` |
| SM75 | Turing | T4, RTX 2080 Ti | `vllm-python312-cuda12_9-sm75` |
| SM80 | Ampere DC | A100, A30 | `vllm-python312-cuda12_9-sm80` |
| SM86 | Ampere | RTX 3090, A40 | `vllm-python312-cuda12_9-sm86` |
| SM89 | Ada Lovelace | RTX 4090, L4, L40 | `vllm-python312-cuda12_9-sm89` |
| SM90 | Hopper | H100, H200, L40S | `vllm-python312-cuda12_9-sm90` |
| SM100 | Blackwell DC | B100, B200, GB200 | `vllm-python312-cuda12_9-sm100` |
| SM120 | Blackwell | RTX 5090, RTX PRO 6000 | `vllm-python312-cuda12_9-sm120` |

### aarch64-linux (2 variants)

| SM | Architecture | GPUs | Variant |
|----|-------------|------|---------|
| SM90 | Hopper | Grace Hopper GH200 | `vllm-python312-cuda12_9-sm90-arm64` |
| SM100 | Blackwell DC | Grace Blackwell GB200/GB300 | `vllm-python312-cuda12_9-sm100-arm64` |

### Total: 10 variants (8 x86_64 + 2 aarch64)

## Quick Start

```bash
# Build a specific variant
flox build vllm-python312-cuda12_9-sm90

# The output is in result-vllm-python312-cuda12_9-sm90/
# Test it
./result-vllm-python312-cuda12_9-sm90/bin/python -c "import vllm; print(vllm.__version__)"

# Check GPU support
./result-vllm-python312-cuda12_9-sm90/bin/python -c "import torch; print(torch.cuda.is_available())"
```

## Variant Selection Guide

**Which variant do I need?**

| GPU | Variant |
|-----|---------|
| H100, H200, L40S | `sm90` |
| RTX 4090, L4, L40 | `sm89` |
| RTX 3090, A40 | `sm86` |
| A100, A30 | `sm80` |
| T4, RTX 2080 Ti | `sm75` |
| V100 | `sm70` |
| B100, B200, GB200 | `sm100` |
| RTX 5090, RTX PRO 6000 | `sm120` |
| Grace Hopper GH200 | `sm90-arm64` |
| Grace Blackwell GB200 | `sm100-arm64` |

## Naming Convention

```
vllm-python312-cuda12_9-sm{XX}[-arm64]
```

The CUDA minor version is encoded in the name (e.g., `cuda12_9` for CUDA 12.9). ARM64 variants include the `-arm64` suffix.

## Build Architecture

vLLM builds use a single `overrideAttrs` on the nixpkgs `python312Packages.vllm` package. SM targeting is set at the nixpkgs import level via `config.cudaCapabilities`, which propagates automatically to vllm and all CUDA dependencies (xformers, flashinfer, CUTLASS).

### Key Differences from ONNX Runtime Builds

| Aspect | ONNX Runtime | vLLM |
|--------|-------------|------|
| Override pattern | `.override` + `.overrideAttrs` (two-layer) | `.overrideAttrs` only (single-layer) |
| Source override | Yes (version, src, patches) | No (uses nixpkgs version directly) |
| CMake flags | Extensive filtering + additions | None |
| Vendored deps | 6+ FETCHCONTENT overrides | None |
| CPU ISA variants | Yes (avx2, avx512, etc.) | No (GPU-only engine) |
| Lines per variant | ~155 | ~25 |
| Torch | N/A | Pre-built wheel (no multi-hour rebuild) |

## Build Requirements

- ~20GB disk space per variant
- 16GB+ RAM recommended for CUDA builds
- Builds use `requiredSystemFeatures = [ "big-parallel" ]`

## License

Build configuration: MIT
vLLM: Apache 2.0
