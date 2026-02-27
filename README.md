# vLLM Custom Build Environment

Per-architecture vLLM builds optimized for specific NVIDIA GPU targets, with custom source-built PyTorch. Built with Flox + Nix.

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

Standard vLLM wheels from PyPI compile CUDA kernels for all compute capabilities and ship a pre-built PyTorch binary. This project creates **per-SM builds** from source with a **custom-built PyTorch**, resulting in:

- **Smaller binaries** — Only include CUDA kernels for your target GPU architecture
- **Optimized kernels** — PagedAttention, FlashAttention, CUTLASS scaled_mm, Marlin quantization, MoE kernels compiled for your exact SM
- **CPU ISA optimization** — PyTorch built from source with CPU-specific instructions (AVX-512, AVX2, ARMv9 SVE2, etc.)
- **Consistent ABI** — All CUDA Python packages (xformers, flashinfer, torchvision, torchaudio) built against the same custom torch
- **Targeted deployment** — Install exactly what your hardware needs

## Version Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| vLLM | 0.15.1 | From nixpkgs |
| PyTorch | 2.9.1 | Custom source build (SM + ISA targeting) |
| CUTLASS | 4.2.1 + 3.9.0+ | v4.2.1 primary, v3.9.0+ for FlashMLA Blackwell |
| CUDA Toolkit | 12.9 | Via nixpkgs `cudaPackages_12_9`, requires driver 560+ |
| Python | 3.13 | Via nixpkgs |
| Nixpkgs | [`0182a36`](https://github.com/NixOS/nixpkgs/tree/0182a361324364ae3f436a63005877674cf45efb) | Pinned revision |

## Build Matrix

### x86_64-linux (42 variants)

| SM | Architecture | GPUs | AVX | AVX2 | AVX-512 | AVX-512 BF16 | AVX-512 VNNI |
|----|-------------|------|-----|------|---------|-------------|-------------|
| SM61 | Pascal | P40, GTX 1080 Ti | `sm61-avx` | `sm61-avx2` | — | — | — |
| SM75 | Turing | T4, RTX 2080 Ti | `sm75-avx` | `sm75-avx2` | `sm75-avx512` | `sm75-avx512bf16` | `sm75-avx512vnni` |
| SM80 | Ampere DC | A100, A30 | `sm80-avx` | `sm80-avx2` | `sm80-avx512` | `sm80-avx512bf16` | `sm80-avx512vnni` |
| SM86 | Ampere | RTX 3090, A40 | `sm86-avx` | `sm86-avx2` | `sm86-avx512` | `sm86-avx512bf16` | `sm86-avx512vnni` |
| SM89 | Ada Lovelace | RTX 4090, L4, L40 | `sm89-avx` | `sm89-avx2` | `sm89-avx512` | `sm89-avx512bf16` | `sm89-avx512vnni` |
| SM90 | Hopper | H100, H200, L40S | `sm90-avx` | `sm90-avx2` | `sm90-avx512` | `sm90-avx512bf16` | `sm90-avx512vnni` |
| SM100 | Blackwell DC | B100, B200, GB200 | `sm100-avx` | `sm100-avx2` | `sm100-avx512` | `sm100-avx512bf16` | `sm100-avx512vnni` |
| SM103 | Blackwell Ultra | B300, GB300 | `sm103-avx` | `sm103-avx2` | `sm103-avx512` | `sm103-avx512bf16` | `sm103-avx512vnni` |
| SM120 | Blackwell | RTX 5090, RTX PRO 6000 | `sm120-avx` | `sm120-avx2` | `sm120-avx512` | `sm120-avx512bf16` | `sm120-avx512vnni` |

All variant names are prefixed with `vllm-python313-cuda12_9-`.

### aarch64-linux (5 variants)

| SM | Architecture | GPUs | ARM Platform | ISA | Variant |
|----|-------------|------|-------------|-----|---------|
| SM80 | Ampere DC | A100 | Ampere Altra (Neoverse N1) | ARMv8.2 | `sm80-armv8_2` |
| SM90 | Hopper | H100 PCIe | Ampere Altra (Neoverse N1) | ARMv8.2 | `sm90-armv8_2` |
| SM90 | Hopper | GH200 | NVIDIA Grace (Neoverse V2) | ARMv9 | `sm90-armv9` |
| SM100 | Blackwell DC | GB200 | NVIDIA Grace (Neoverse V2) | ARMv9 | `sm100-armv9` |
| SM103 | Blackwell Ultra | GB300 | NVIDIA Grace (Neoverse V2) | ARMv9 | `sm103-armv9` |

### Total: 47 variants (all Python 3.13)

## Quick Start

```bash
# Build a specific variant
flox build vllm-python313-cuda12_9-sm90-avx512

# The output is in result-vllm-python313-cuda12_9-sm90-avx512/
# Test it
./result-vllm-python313-cuda12_9-sm90-avx512/bin/python -c "import vllm; print(vllm.__version__)"

# Check GPU support
./result-vllm-python313-cuda12_9-sm90-avx512/bin/python -c "import torch; print(torch.cuda.is_available())"

# Verify custom torch (should show SM-specific CUDA arch and ISA flags)
./result-vllm-python313-cuda12_9-sm90-avx512/bin/python -c "import torch; print(torch.__config__.show())"
```

## Variant Selection Guide

### Step 1: Choose your GPU

| GPU | SM |
|-----|----|
| P40, GTX 1080 Ti | SM61 |
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
| Grace Blackwell Ultra GB300 | SM103 (armv9) |

### Step 2: Choose your CPU ISA

**x86_64 servers:**
- **avx512bf16** — Sapphire Rapids, Emerald Rapids (best for BF16 inference)
- **avx512vnni** — Cascade Lake, Ice Lake, Sapphire Rapids (best for INT8 inference)
- **avx512** — Skylake-SP, Cascade Lake, Ice Lake (general datacenter)
- **avx2** — Haswell+, any modern x86_64 (broadest compatibility)
- **avx** — Sandy Bridge+ (legacy servers)

**ARM64 servers:**
- **armv9** — NVIDIA Grace (GH200, GB200, GB300): SVE2, BF16, I8MM
- **armv8_2** — Ampere Altra/Altra Max with PCIe GPUs: fp16, dotprod

## Naming Convention

```
vllm-python313-cuda12_9-sm{XX}-{isa}
```

The Python version, CUDA minor version, SM architecture, and CPU ISA are all encoded in the name. ARM ISAs (`armv8_2`, `armv9`) imply aarch64-linux platform.

## Build Architecture

vLLM builds use `python313.override { packageOverrides }` to create a custom Python package set where:
- **torch** is built from source with SM-specific GPU targeting (`gpuTargets`) and CPU ISA flags (`CXXFLAGS`/`CFLAGS`)
- **bitsandbytes** is overridden with single-SM restriction (CCCL 2.8.2 workaround)

All CUDA Python packages (xformers, flashinfer, torchvision, torchaudio) automatically resolve against the custom torch via callPackage, ensuring consistent ABI across the entire package set.

Shared helpers in `.flox/pkgs/lib/`:
- `cpu-isa.nix` — CPU ISA flag definitions (7 ISAs: avx, avx2, avx512, avx512bf16, avx512vnni, armv8_2, armv9)
- `custom-torch.nix` — Custom PyTorch builder with `.override` + `.overrideAttrs`

## Build Requirements

- ~30GB disk space per variant (PyTorch source build + vLLM + CUDA deps)
- 16GB+ RAM recommended for CUDA builds
- Builds use `requiredSystemFeatures = [ "big-parallel" ]`
- Parallel CUDA compilation is capped at 16 jobs via `NIX_BUILD_CORES = 16` and `MAX_JOBS=16` to prevent swap thrashing on machines with ≤128GB RAM (see [CUDA-BUILD-PARALLELISM.md](CUDA-BUILD-PARALLELISM.md) for details)

## Build Notes

- **Custom PyTorch**: PyTorch 2.9.1 is built from source with SM-specific CUDA targeting and CPU ISA optimization. The `packageOverrides` mechanism ensures xformers, flashinfer, and other torch-dependent packages all link against the custom build, avoiding ABI mismatches.
- **SM61 (Pascal)**: Uses `USE_CUDNN=0` — cuDNN 9.11+ dropped support for SM < 7.5. Only avx and avx2 ISAs (Pascal-era servers predate AVX-512).
- **SM103 (Blackwell Ultra)**: Builds with CUDA 12.9 via family-specific `sm_10x` compilation. Native `sm_103` cubins require CUDA 13.0+.
- **bitsandbytes single-SM override**: CCCL 2.8.2 (CUDA 12.9) has a missing `_CCCL_PP_SPLICE_WITH_IMPL20` macro that causes compile failures when targeting all 19 SM architectures. Each variant overrides bitsandbytes with `-DCOMPUTE_CAPABILITY=<SM>` to restrict compilation to the target architecture. Standalone build targets `bitsandbytes-cuda12_9` (Python 3.12) and `bitsandbytes-python313-cuda12_9` (Python 3.13) are provided for independent testing.

## Branch Strategy

| Branch | vLLM Version | Nixpkgs Pin | PyTorch | Python | Status |
|--------|-------------|-------------|---------|--------|--------|
| `main` | 0.15.1 | `0182a36` | 2.9.1 (source) | 3.13 | Current stable |
| `vllm-0.14.0` | 0.14.0 | `46336d4` | 2.9.1 | 3.12 | Previous release |
| `vllm-0.13.0` | 0.13.0 | `ed142ab` | 2.9.1 | 3.12 | Older release |

Each branch has its own Python version and nixpkgs pin. The custom PyTorch branches in `build-pytorch` match by pin: `pytorch-2.9-vllm-0.15.1` (Python 3.13, pin `0182a36`) and `pytorch-2.9-vllm-0.14.0` (Python 3.12, pin `46336d4`).

## License

Build configuration: MIT
vLLM: Apache 2.0
