# CLAUDE.md

## Project Overview

This is a Flox-based repository that builds per-architecture vLLM variants optimized for specific NVIDIA GPU targets. Each variant is a Nix expression in `.flox/pkgs/` that configures the nixpkgs `vllm` package with targeted CUDA compute capabilities via `config.cudaCapabilities` and substitutes a custom source-built PyTorch with SM-specific GPU targeting and CPU ISA optimization.

## Common Development Commands

```bash
# Build a specific variant
flox build vllm-python312-cuda12_9-sm90-avx512

# Test the built package
./result-vllm-python312-cuda12_9-sm90-avx512/bin/python -c "import vllm; print(vllm.__version__)"

# Check GPU support
./result-vllm-python312-cuda12_9-sm90-avx512/bin/python -c "import torch; print(torch.cuda.is_available())"

# Check torch is custom-built (should show SM-specific CUDA arch, ISA flags)
./result-vllm-python312-cuda12_9-sm90-avx512/bin/python -c "import torch; print(torch.__config__.show())"

# Test LLM import
./result-vllm-python312-cuda12_9-sm90-avx512/bin/python -c "from vllm import LLM; print('OK')"

# Publish to catalog
flox publish vllm-python312-cuda12_9-sm90-avx512
```

## Architecture

### Build System

vLLM builds use `python312.override { packageOverrides }` to create a custom Python package set where torch is built from source with SM-specific GPU targeting and CPU ISA flags. This ensures all CUDA Python packages (xformers, flashinfer, torchvision, torchaudio) resolve against the custom torch, avoiding ABI conflicts.

Each variant file:
1. Imports nixpkgs with `config.cudaCapabilities` for SM targeting
2. Creates a custom Python package set via `packageOverrides` that replaces `torch` and `bitsandbytes`
3. Builds vLLM from the custom package set via `python312Custom.pkgs.vllm`

Shared helpers in `.flox/pkgs/lib/`:
- `cpu-isa.nix` — CPU ISA flag lookup table (avx, avx2, avx512, avx512bf16, avx512vnni, armv8_2, armv9)
- `custom-torch.nix` — Builds custom PyTorch with `.override { gpuTargets }` + `.overrideAttrs` for ISA flags. Accepts `cudaVersionTag` parameter (default `"cuda12_9"`) for CUDA version differentiation in store paths.

### Package Naming Convention

```
vllm-python312-cuda{12_9|12_8}-sm{XX}-{isa}.nix
```

The Python version, CUDA minor version, SM architecture, and CPU ISA are encoded in the filename. ARM ISA suffixes (`armv8_2`, `armv9`) imply `aarch64-linux` platform.

Examples:
- `vllm-python312-cuda12_9-sm90-avx512.nix` (H100, x86_64, AVX-512, CUDA 12.9)
- `vllm-python312-cuda12_8-sm90-avx512.nix` (H100, x86_64, AVX-512, CUDA 12.8)
- `vllm-python312-cuda12_9-sm90-avx512bf16.nix` (H100, x86_64, AVX-512 BF16)
- `vllm-python312-cuda12_9-sm90-armv9.nix` (Grace Hopper GH200, aarch64, ARMv9)
- `vllm-python312-cuda12_8-sm90-armv8_2.nix` (H100 PCIe in Ampere Altra, aarch64, ARMv8.2)
- `vllm-python312-cuda12_9-sm103-armv9.nix` (Grace Blackwell Ultra GB300, aarch64, CUDA 12.9 only)

### CPU ISA Variants

**x86_64** (5 ISAs):
- **avx** — Sandy Bridge+ (pre-Haswell servers)
- **avx2** — Haswell+ (consumer/workstation): AVX2 + FMA + F16C
- **avx512** — Skylake-SP+ (datacenter): AVX-512F/DQ/VL/BW + FMA
- **avx512bf16** — Cooper Lake, Sapphire Rapids+: AVX-512 + BF16
- **avx512vnni** — Cascade Lake+: AVX-512 + VNNI

SM61 (Pascal) only gets avx and avx2 — Pascal-era servers predate AVX-512 CPUs.

**aarch64** (2 ISAs):
- **armv8_2** — Ampere Altra/Altra Max (Neoverse N1, ARMv8.2-A): fp16+dotprod
- **armv9** — NVIDIA Grace (Neoverse V2, ARMv9.0-A): SVE2+BF16+I8MM

### SM Architectures

| SM | Capability | GPUs | x86_64 ISAs | aarch64 ISAs |
|----|-----------|------|-------------|-------------|
| SM61 | 6.1 | P40, GTX 1080 Ti | avx, avx2 | — |
| SM70 | 7.0 | V100, Titan V | all 5 | — |
| SM75 | 7.5 | T4, RTX 2080 Ti | all 5 | — |
| SM80 | 8.0 | A100, A30 | all 5 | armv8_2 |
| SM86 | 8.6 | RTX 3090, A40 | all 5 | — |
| SM89 | 8.9 | RTX 4090, L4, L40 | all 5 | — |
| SM90 | 9.0 | H100, H200, L40S, GH200 | all 5 | armv8_2, armv9 |
| SM100 | 10.0 | B100, B200, GB200 | all 5 | armv9 |
| SM103 | 10.3 | B300, GB300 | all 5 | armv9 |
| SM120 | 12.0 | RTX 5090, RTX PRO 6000 | all 5 | — |

**CUDA 12.9: 52 variants** (47 x86_64 + 5 aarch64). **CUDA 12.8: 46 variants** (42 x86_64 + 4 aarch64, no SM103). **Total: 98 variants**, all Python 3.12.

### ARM64 Platform Mapping

- **armv8_2 variants**: Ampere Altra/Altra Max servers with PCIe GPUs (A100, H100 PCIe)
- **armv9 variants**: NVIDIA Grace-based superchips (GH200, GB200, GB300) with NVLink-C2C GPUs

### Key Variables Per Variant

- `smCapability`: CUDA compute capability string (e.g., `"9.0"`)
- `cudaCapabilities`: List passed to `config` at nixpkgs import (e.g., `[ "9.0" ]`)
- `variantName`: Package name matching the filename
- `cpuISA`: CPU ISA record from `lib/cpu-isa.nix`
- `platform`: `"x86_64-linux"` or `"aarch64-linux"`

### Nixpkgs Pin

- Revision: `46336d4d6980ae6f136b45c8507b17787eb186a0`
- vLLM: 0.14.0
- PyTorch: 2.9.1 (custom source build — compiled from source with SM + ISA targeting)
- CUTLASS: v4.2.1 primary + v3.9.0 for FlashMLA Blackwell
- Python: 3.12
- CUDA: 12.9 via `cudaPackages_12_9` overlay (driver 560+), 12.8 via `cudaPackages_12_8` overlay (driver 550+)

### Build Parallelism

Each variant sets `NIX_BUILD_CORES = 16` to cap parallel CUDA compilation. CUTLASS and FlashAttention template-heavy compilation units use 3–8 GB of RAM each; unrestricted parallelism on machines with ≤128 GB RAM causes swap thrashing. The custom torch build also uses `MAX_JOBS=16` and `ninjaFlags = [ "-j16" ]`. See [CUDA-BUILD-PARALLELISM.md](CUDA-BUILD-PARALLELISM.md) for full details and tuning guidance.

### bitsandbytes Single-SM Override

CCCL 2.8.2 (shipped with CUDA 12.9) has a bug: `_CCCL_PP_SPLICE_WITH_IMPL20` is missing from `preprocessor.h`, and `IMPL21` incorrectly chains to `IMPL19`. When bitsandbytes builds all 19 default SM architectures, `__CUDA_ARCH_LIST__` expands to 19 comma-separated values, pushing the variadic arg count into the broken `IMPL20`/`IMPL21` range and causing a hard compile error.

Each variant overrides bitsandbytes with `-DCOMPUTE_CAPABILITY=<SM>` (matching the variant's target architecture) via `packageOverrides`, restricting compilation to a single SM. This keeps the macro arg count well below the broken range and also produces a smaller binary. CCCL 2.7.0 (CUDA 12.8) does **not** have this bug, but the single-SM override is kept for smaller binaries and consistency.

### Custom PyTorch Integration

The `packageOverrides` mechanism replaces torch across the entire Python package set. This is critical because vllm, xformers, flashinfer, torchvision, and torchaudio all take `torch` as a callPackage argument. Replacing torch only in vllm's `propagatedBuildInputs` would leave xformers/flashinfer linked against the original torch, causing ABI conflicts.

The custom torch build (`lib/custom-torch.nix`) uses:
- `.override { gpuTargets = [ smCapability ]; }` for SM-specific CUDA targeting
- `.overrideAttrs` for CPU ISA flags (`CXXFLAGS`/`CFLAGS`), `MAX_JOBS`, and `ninjaFlags`
- Optional `extraPreConfigure` for SM-specific workarounds (e.g., `USE_CUDNN=0` for SM61/SM70)
- Optional `cudaVersionTag` (default `"cuda12_9"`) for pname differentiation between CUDA versions

### SM-Specific Notes

- **SM61 (Pascal)**: `USE_CUDNN=0` — cuDNN 9.11+ dropped support for SM < 7.5
- **SM70 (Volta)**: `USE_CUDNN=0` — cuDNN 9.11+ dropped support for SM < 7.5
- **SM103 (Blackwell Ultra)**: CUDA 12.9 only — nvcc 12.8 does not support SM103. Builds with CUDA 12.9 via family-specific `sm_10x` compilation. Native `sm_103` cubins require CUDA 13.0+, but family-specific targets provide forward-compatible PTX.

### Branch Strategy

| Branch | vLLM | Nixpkgs Pin | PyTorch | Python |
|--------|------|-------------|---------|--------|
| `main` | 0.15.1 | `0182a36` | 2.9.1 (source) | 3.13 |
| `vllm-0.14.0` | 0.14.0 | `46336d4` | 2.9.1 (source) | 3.12 |
| `vllm-0.13.0` | 0.13.0 | `ed142ab` | 2.9.1 | 3.12 |

### CUDA Version Documentation

Each `.nix` file includes a three-line header comment:
```nix
# vLLM 0.14.0 for NVIDIA Hopper (SM90: H100, H200, L40S) — AVX-512
# CUDA 12.9 — Requires NVIDIA driver 560+
# Custom PyTorch 2.9.1 built from source (SM90 + AVX-512)
```

CUDA 12.8 variants use:
```nix
# CUDA 12.8 — Requires NVIDIA driver 550+
```

The `meta.description` also includes the CUDA version and ISA:
```nix
description = "vLLM 0.14.0 for NVIDIA H100/H200/L40S (SM90) [CUDA 12.9, custom PyTorch AVX-512]";
description = "vLLM 0.14.0 for NVIDIA H100/H200/L40S (SM90) [CUDA 12.8, custom PyTorch AVX-512]";
```

CUDA 12.8 variant files also pass `cudaVersionTag = "cuda12_8"` to the custom-torch.nix import.

## Package Development Guidelines

### Adding a New Variant

1. Copy an existing variant `.nix` file with similar configuration
2. Update `cudaCapabilities`, `smCapability`, `variantName`, `cpuISA`, `platform`, header comment, and `meta.description`
3. Ensure the header comment includes the vLLM version, CUDA version, driver requirement, and PyTorch source info
4. Ensure `variantName` matches the filename
5. For SM < 7.5, add `extraPreConfigure = "export USE_CUDNN=0"` to the torch import
6. For CUDA 12.8 variants, use `cudaPackages_12_8` overlay, add `cudaVersionTag = "cuda12_8"` to the torch import, and update driver requirement to 550+
7. SM103 is CUDA 12.9 only — do not create CUDA 12.8 variants for SM103
8. Test with `flox build <variant-name>`

### Adding a New CPU ISA

1. Add the ISA definition to `lib/cpu-isa.nix`
2. Create variant files using the new ISA key
3. Update documentation

### Adding a New CUDA Version

When adding a new CUDA toolkit alongside existing ones:
1. Ensure the nixpkgs pin includes `cudaPackages_XX_Y` for the target version
2. Copy existing variant files, updating the overlay (e.g., `cudaPackages_13_0`)
3. Update filenames to reflect the new CUDA version (e.g., `cuda13_0`)
4. Update `variantName`, header comments, `meta.description`, and driver requirement
5. Add `cudaVersionTag = "cudaXX_Y"` to the custom-torch.nix import
6. Skip SM architectures not supported by the target nvcc version
7. Update README.md and CLAUDE.md with the new CUDA version, variant counts, and driver requirement

### Updating vLLM Version

Update the nixpkgs pin to a revision containing the target vLLM version. All variants share the same pin, so updating it updates all variants. Update header comments and `meta.description` with the new version.

### Troubleshooting

- **Build failures with SM targeting**: If `config.cudaCapabilities` doesn't propagate to all dependencies, may need to pass `gpuTargets` via override
- **xformers/flashinfer issues**: These dependencies should automatically resolve against the custom torch via `packageOverrides`. If not, add explicit overrides to the `packageOverrides` block
- **CUTLASS compilation**: Newer SM architectures may require CUTLASS updates not yet in the pinned nixpkgs
- **Swap thrashing / desktop freeze during builds**: CUDA compilation is memory-intensive. See [CUDA-BUILD-PARALLELISM.md](CUDA-BUILD-PARALLELISM.md) for diagnosis and tuning of `NIX_BUILD_CORES`
- **packageOverrides fixpoint**: If any package in the set has a circular dependency on torch's attributes, it could cause infinite recursion. This is unlikely since nixpkgs already handles this pattern
- **SM103 Triton issues**: If Triton fails to compile for `sm_103a`, it may need a source build with SM103 support (upstream Triton issue)

## Commit Message Conventions

- Package updates: `vllm: update to latest`
- New packages: `vllm: init`
- New variants: `vllm: add SM{XX} variant`
- Infrastructure changes: Use appropriate prefix (e.g., `workflows:`, `flake:`)
