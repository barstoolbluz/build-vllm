# vLLM Custom Build Environment

Per-architecture vLLM builds optimized for specific NVIDIA GPU targets. Built with Flox + Nix.

## CUDA Compatibility

Variants are built against **CUDA 12.9** (driver 560+) and **CUDA 12.8** (driver 550+). Choose based on your installed driver version.

| CUDA Version | Minimum Driver | Variant Infix |
|-------------|---------------|---------------|
| 12.9 | 560+ | `cuda12_9` |
| 12.8 | 550+ | `cuda12_8` |

- **Forward compatibility**: CUDA 12.x builds work with any driver that supports the target CUDA version or later
- **No cross-major compatibility**: CUDA 12.x builds are **not** compatible with CUDA 11.x or 13.x runtimes
- **SM103 (Blackwell Ultra)**: CUDA 12.9 only — nvcc 12.8 does not support SM103
- **Check your driver**: Run `nvidia-smi` — the "CUDA Version" in the top-right shows the maximum CUDA version your driver supports

```bash
# Verify your driver supports your target CUDA version
nvidia-smi
# Look for "CUDA Version: 12.8" or higher in the output
```

## Overview

Standard vLLM wheels from PyPI compile CUDA kernels for all compute capabilities. This project creates **per-SM builds** with custom source-built PyTorch, resulting in:

- **Smaller binaries** — Only include CUDA kernels for your target GPU architecture
- **Optimized kernels** — PagedAttention, FlashAttention, CUTLASS scaled_mm, Marlin quantization, MoE kernels compiled for your exact SM
- **CPU ISA optimization** — PyTorch built from source with CPU-specific instruction sets (AVX, AVX2, AVX-512, ARMv8.2, ARMv9)
- **ABI consistency** — Custom torch propagated via `packageOverrides` ensures xformers, flashinfer, torchvision, and torchaudio all link against the same torch
- **Targeted deployment** — Install exactly what your hardware needs

## Version Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| vLLM | 0.14.1 | Version + src overridden atop 0.14.0 nixpkgs pin |
| PyTorch | 2.9.1 | Custom source build (SM + ISA targeting) |
| CUTLASS | 4.2.1 + 3.9.0 | v4.2.1 primary, v3.9.0 for FlashMLA Blackwell |
| CUDA Toolkit | 12.9 / 12.8 | 12.9 via `cudaPackages_12_9` (driver 560+), 12.8 via `cudaPackages_12_8` (driver 550+) |
| Python | 3.12 | Via nixpkgs |
| Nixpkgs | [`46336d4`](https://github.com/NixOS/nixpkgs/tree/46336d4d6980ae6f136b45c8507b17787eb186a0) | Pinned revision |

## Build Matrix

Each SM/ISA combination is available for both CUDA 12.9 and CUDA 12.8, except SM103 which is CUDA 12.9 only (nvcc 12.8 doesn't support it).

### x86_64-linux

| SM | Architecture | GPUs | avx | avx2 | avx512 | avx512bf16 | avx512vnni | CUDA |
|----|-------------|------|-----|------|--------|------------|------------|------|
| SM61 | Pascal | P40, GTX 1080 Ti | x | x | — | — | — | 12.9, 12.8 |
| SM70 | Volta | V100, Titan V | x | x | x | x | x | 12.9, 12.8 |
| SM75 | Turing | T4, RTX 2080 Ti | x | x | x | x | x | 12.9, 12.8 |
| SM80 | Ampere DC | A100, A30 | x | x | x | x | x | 12.9, 12.8 |
| SM86 | Ampere | RTX 3090, A40 | x | x | x | x | x | 12.9, 12.8 |
| SM89 | Ada Lovelace | RTX 4090, L4, L40 | x | x | x | x | x | 12.9, 12.8 |
| SM90 | Hopper | H100, H200, L40S | x | x | x | x | x | 12.9, 12.8 |
| SM100 | Blackwell DC | B100, B200, GB200 | x | x | x | x | x | 12.9, 12.8 |
| SM103 | Blackwell Ultra | B300, GB300 | x | x | x | x | x | 12.9 only |
| SM120 | Blackwell | RTX 5090, RTX PRO 6000 | x | x | x | x | x | 12.9, 12.8 |

SM61 (Pascal) only gets avx/avx2 — Pascal-era servers predate AVX-512 CPUs.

Variant names are prefixed with `vllm-python312-cuda12_9-` or `vllm-python312-cuda12_8-`.

### aarch64-linux

| SM | Architecture | GPUs | ISA | Variant | CUDA |
|----|-------------|------|-----|---------|------|
| SM80 | Ampere DC | A100 (Altra PCIe) | armv8_2 | `sm80-armv8_2` | 12.9, 12.8 |
| SM90 | Hopper | H100 (Altra PCIe) | armv8_2 | `sm90-armv8_2` | 12.9, 12.8 |
| SM90 | Grace Hopper | GH200 | armv9 | `sm90-armv9` | 12.9, 12.8 |
| SM100 | Grace Blackwell | GB200 | armv9 | `sm100-armv9` | 12.9, 12.8 |
| SM103 | Grace Blackwell Ultra | GB300 | armv9 | `sm103-armv9` | 12.9 only |

### Total: 98 variants (52 CUDA 12.9 + 46 CUDA 12.8)

## Quick Start

```bash
# Build a specific variant (CUDA 12.9, driver 560+)
flox build vllm-python312-cuda12_9-sm90-avx512

# Or use CUDA 12.8 for older drivers (driver 550+)
flox build vllm-python312-cuda12_8-sm90-avx512

# The output is in result-<variant-name>/
# Test it
./result-vllm-python312-cuda12_9-sm90-avx512/bin/python -c "import vllm; print(vllm.__version__)"

# Check GPU support
./result-vllm-python312-cuda12_9-sm90-avx512/bin/python -c "import torch; print(torch.cuda.is_available())"

# Check torch is custom-built (should show SM-specific CUDA arch, ISA flags)
./result-vllm-python312-cuda12_9-sm90-avx512/bin/python -c "import torch; print(torch.__config__.show())"
```

## Variant Selection Guide

**Which variant do I need?**

First, identify your GPU's SM architecture, then choose the CPU ISA matching your server:

| GPU | SM |
|-----|----|
| P40, GTX 1080 Ti | SM61 |
| V100, Titan V | SM70 |
| T4, RTX 2080 Ti | SM75 |
| A100, A30 | SM80 |
| RTX 3090, A40 | SM86 |
| RTX 4090, L4, L40 | SM89 |
| H100, H200, L40S | SM90 |
| B100, B200, GB200 | SM100 |
| B300, GB300 | SM103 |
| RTX 5090, RTX PRO 6000 | SM120 |
| Grace Hopper GH200 | SM90 (armv9) |
| Grace Blackwell GB200 | SM100 (armv9) |

Then choose your CUDA version based on your driver:

| Driver Version | CUDA |
|---------------|------|
| 560+ | `cuda12_9` (recommended) |
| 550–559 | `cuda12_8` |

Note: SM103 (B300, GB300) requires CUDA 12.9.

**CPU ISA guide (x86_64):**

| ISA | CPU Generation | Example CPUs |
|-----|---------------|--------------|
| avx | Sandy Bridge+ | Pre-Haswell servers |
| avx2 | Haswell+ | Most consumer/workstation CPUs |
| avx512 | Skylake-SP+ | Intel datacenter (Xeon Scalable) |
| avx512bf16 | Cooper Lake, Sapphire Rapids+ | Newer Intel datacenter with BF16 |
| avx512vnni | Cascade Lake+ | Intel datacenter with VNNI |

**CPU ISA guide (aarch64):**

| ISA | CPU | Example Systems |
|-----|-----|-----------------|
| armv8_2 | Ampere Altra/Altra Max | Altra servers with PCIe GPUs |
| armv9 | NVIDIA Grace | GH200, GB200, GB300 superchips |

## Naming Convention

```
vllm-python312-cuda{12_9|12_8}-sm{XX}-{isa}
```

The Python version, CUDA minor version, SM architecture, and CPU ISA are encoded in the name. ARM ISA suffixes (`armv8_2`, `armv9`) imply `aarch64-linux` platform.

## Build Architecture

vLLM builds use `python312.override { packageOverrides }` to create a custom Python package set where torch is built from source with SM-specific GPU targeting and CPU ISA flags. This ensures all CUDA Python packages resolve against the custom torch, avoiding ABI conflicts.

Shared helpers in `.flox/pkgs/lib/`:
- `cpu-isa.nix` — CPU ISA flag lookup table
- `custom-torch.nix` — Builds custom PyTorch with `.override { gpuTargets }` + `.overrideAttrs` for ISA flags. Accepts `cudaVersionTag` parameter (default `"cuda12_9"`) for CUDA version differentiation in store paths.

## Build Requirements

- ~20GB disk space per variant (PyTorch source build adds significant intermediate artifacts)
- 16GB+ RAM recommended for CUDA builds
- Builds use `requiredSystemFeatures = [ "big-parallel" ]`
- Parallel CUDA compilation is capped at 16 jobs via `NIX_BUILD_CORES = 16` to prevent swap thrashing on machines with ≤128GB RAM (see [CUDA-BUILD-PARALLELISM.md](CUDA-BUILD-PARALLELISM.md) for details)
- Custom torch build uses `MAX_JOBS=16` and `ninjaFlags = [ "-j16" ]`

## Build Notes

- **Custom PyTorch**: Each variant builds PyTorch from source with SM-specific CUDA targeting and CPU ISA flags. The `packageOverrides` mechanism ensures all downstream packages (xformers, flashinfer, torchvision, torchaudio) link against the custom torch.
- **bitsandbytes single-SM override**: CCCL 2.8.2 (CUDA 12.9) has a missing `_CCCL_PP_SPLICE_WITH_IMPL20` macro that causes compile failures when targeting all 19 SM architectures. Each variant overrides bitsandbytes with `-DCOMPUTE_CAPABILITY=<SM>` to restrict compilation to the target architecture.
- **CUDA 12.8 variants**: Use `cudaPackages_12_8` overlay and `cudaVersionTag = "cuda12_8"` for custom-torch.nix. CCCL 2.7.0 (CUDA 12.8) does **not** have the IMPL20 bug, but the bitsandbytes single-SM override is kept for smaller binaries and consistency.
- **SM61/SM70 (Pascal/Volta)**: `USE_CUDNN=0` — cuDNN 9.11+ dropped support for SM < 7.5
- **SM103 (Blackwell Ultra)**: CUDA 12.9 only — nvcc 12.8 does not support SM103. Family-specific `sm_10x` compilation via CUDA 12.9.

## Branch Strategy

| Branch | vLLM Version | Nixpkgs Pin | PyTorch | Python | Status |
|--------|-------------|-------------|---------|--------|--------|
| `main` | 0.15.1 | `0182a36` | 2.9.1 (source) | 3.13 | Current stable |
| `vllm-0.14.1-python312` | 0.14.1 | `46336d4` | 2.9.1 (source) | 3.12 | Patch release |
| `vllm-0.14.1-python311` | 0.14.1 | `46336d4` | 2.9.1 (source) | 3.11 | Patch release |
| `vllm-0.14.0-python312` | 0.14.0 | `46336d4` | 2.9.1 (source) | 3.12 | Previous release |
| `vllm-0.14.0-python311` | 0.14.0 | `46336d4` | 2.9.1 (source) | 3.11 | Previous release |
| `vllm-0.13.0` | 0.13.0 | `ed142ab` | 2.9.1 | 3.12 | Older release |

## License

Build configuration: MIT
vLLM: Apache 2.0
