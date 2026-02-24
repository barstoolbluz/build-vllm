# CLAUDE.md

## Project Overview

This is a Flox-based repository that builds per-architecture vLLM variants optimized for specific NVIDIA GPU targets. Each variant is a Nix expression in `.flox/pkgs/` that configures the nixpkgs `vllm` package with targeted CUDA compute capabilities via `config.cudaCapabilities`.

Unlike ORT builds, vLLM variants require no source/version/patch overrides — the nixpkgs vllm package (v0.14.0) already supports per-SM targeting, and torch is a pre-built binary wheel.

## Common Development Commands

```bash
# Build a specific variant
flox build vllm-python312-cuda12_9-sm90

# Test the built package
./result-vllm-python312-cuda12_9-sm90/bin/python -c "import vllm; print(vllm.__version__)"

# Check GPU support
./result-vllm-python312-cuda12_9-sm90/bin/python -c "import torch; print(torch.cuda.is_available())"

# Test LLM import
./result-vllm-python312-cuda12_9-sm90/bin/python -c "from vllm import LLM; print('OK')"

# Publish to catalog
flox publish vllm-python312-cuda12_9-sm90
```

## Architecture

### Build System

vLLM builds use a single-layer `overrideAttrs` on the nixpkgs `python312Packages.vllm` package. SM targeting is set at the nixpkgs import level via `config.cudaCapabilities`, which propagates automatically to vllm, xformers, flashinfer, and CUTLASS.

Key simplification vs ORT:
- No `.override` + `.overrideAttrs` two-layer pattern
- No `version`/`src`/`patches`/`postPatch` overrides
- No `cmakeFlags` filtering or additions
- No vendored dep `FETCHCONTENT_SOURCE_DIR_*` overrides
- Each `.nix` file is ~37 lines instead of ~155 lines

### Package Naming Convention

```
vllm-python312-cuda12_9-sm{XX}[-arm64].nix
```

The CUDA minor version is encoded in the filename (e.g., `cuda12_9` for CUDA 12.9). ARM64 variants include the `-arm64` suffix.

Examples:
- `vllm-python312-cuda12_9-sm90.nix` (H100, x86_64)
- `vllm-python312-cuda12_9-sm90-arm64.nix` (Grace Hopper, aarch64)

### No CPU-only Variants

vLLM is fundamentally a GPU inference engine. CPU ISA variants (avx2/avx512) provide negligible benefit since vLLM's CPU code is mostly Python orchestration.

### Key Variables Per Variant

- `smCapability`: CUDA compute capability string (e.g., `"9.0"`)
- `cudaCapabilities`: List passed to `config` at nixpkgs import (e.g., `[ "9.0" ]`)
- `variantName`: Package name matching the filename (includes `cuda12_9` prefix)

### Nixpkgs Pin

- Revision: `46336d4d6980ae6f136b45c8507b17787eb186a0`
- vLLM: 0.14.0
- PyTorch: 2.9.1 (pre-built wheel — not compiled from source)
- CUTLASS: v4.2.1 primary + v3.9.0 for FlashMLA Blackwell
- Python: 3.12
- CUDA: 12.9 via `cudaPackages_12_9` overlay — **requires NVIDIA driver 560+**

### Build Parallelism

Each variant sets `NIX_BUILD_CORES = 16` to cap parallel CUDA compilation. CUTLASS and FlashAttention template-heavy compilation units use 3–8 GB of RAM each; unrestricted parallelism on machines with ≤128 GB RAM causes swap thrashing. See [CUDA-BUILD-PARALLELISM.md](CUDA-BUILD-PARALLELISM.md) for full details and tuning guidance.

### bitsandbytes Single-SM Override

CCCL 2.8.2 (shipped with CUDA 12.9) has a bug: `_CCCL_PP_SPLICE_WITH_IMPL20` is missing from `preprocessor.h`, and `IMPL21` incorrectly chains to `IMPL19`. When bitsandbytes builds all 19 default SM architectures, `__CUDA_ARCH_LIST__` expands to 19 comma-separated values, pushing the variadic arg count into the broken `IMPL20`/`IMPL21` range and causing a hard compile error.

Each variant overrides bitsandbytes with `-DCOMPUTE_CAPABILITY=<SM>` (matching the variant's target architecture), restricting compilation to a single SM. This keeps the macro arg count well below the broken range and also produces a smaller binary.

A standalone build target `bitsandbytes-cuda12_9` (`.flox/pkgs/bitsandbytes-cuda12_9.nix`) is available for independent testing: `flox build bitsandbytes-cuda12_9`.

### Branch Strategy

| Branch | vLLM | Nixpkgs Pin | PyTorch |
|--------|------|-------------|---------|
| `main` | 0.15.1 | `0182a36` | 2.10.0 |
| `vllm-0.14.0` | 0.14.0 | `46336d4` | 2.9.1 |
| `vllm-0.13.0` | 0.13.0 | `ed142ab` | 2.9.1 |

All branches share the same CUDA 12.9 toolkit, build matrix, and variant naming convention.

### CUDA Version Documentation

Each `.nix` file includes a two-line header comment:
```nix
# vLLM 0.14.0 for NVIDIA Hopper (SM90: H100, H200, L40S)
# CUDA 12.9 — Requires NVIDIA driver 560+
```

The `meta.description` also includes the CUDA version:
```nix
description = "vLLM 0.14.0 for NVIDIA H100/H200/L40S (SM90) [CUDA 12.9]";
```

## Package Development Guidelines

### Adding a New Variant

1. Copy an existing variant `.nix` file with similar configuration
2. Update `cudaCapabilities`, `smCapability`, `variantName`, header comment, and `meta.description`
3. Set `meta.platforms` to `[ "x86_64-linux" ]` or `[ "aarch64-linux" ]`
4. Ensure the header comment includes the vLLM version, CUDA version, and driver requirement
5. Ensure `variantName` matches the filename
6. Test with `flox build <variant-name>`

### Adding a New CUDA Version

When a new CUDA toolkit is needed:
1. Update the nixpkgs pin to a revision with the target CUDA version
2. Update the overlay in each `.nix` file (e.g., `cudaPackages_13_0`)
3. Rename files to reflect the new CUDA version (e.g., `cuda13_0`)
4. Update `variantName`, header comments, and `meta.description` in each file
5. Update README.md and CLAUDE.md with the new CUDA version and driver requirement

### Updating vLLM Version

Update the nixpkgs pin to a revision containing the target vLLM version. All variants share the same pin, so updating it updates all variants. Update header comments and `meta.description` with the new version.

### Troubleshooting

- **Build failures with SM targeting**: If `config.cudaCapabilities` doesn't propagate to all dependencies, may need to pass `gpuTargets` via override
- **xformers/flashinfer issues**: These dependencies may need their own SM targeting overrides if they don't pick up `config.cudaCapabilities`
- **CUTLASS compilation**: Newer SM architectures may require CUTLASS updates not yet in the pinned nixpkgs
- **Swap thrashing / desktop freeze during builds**: CUDA compilation is memory-intensive. See [CUDA-BUILD-PARALLELISM.md](CUDA-BUILD-PARALLELISM.md) for diagnosis and tuning of `NIX_BUILD_CORES`

## Commit Message Conventions

- Package updates: `vllm: update to latest`
- New packages: `vllm: init`
- New variants: `vllm: add SM{XX} variant`
- Infrastructure changes: Use appropriate prefix (e.g., `workflows:`, `flake:`)
