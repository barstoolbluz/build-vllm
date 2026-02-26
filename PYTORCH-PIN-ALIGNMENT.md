# PyTorch Build Recipes: Align to vLLM Nixpkgs Pin

## Goal

Update the `pytorch-2.9` branch in `build-pytorch` to use the **same nixpkgs pin** as `build-vllm/main`. Both repos already build PyTorch 2.9.1 with CUDA 12.9 — the only difference is the nixpkgs snapshot. Aligning the pin enables Phase 3: substituting a custom-built PyTorch into vLLM builds.

## Current State

| Repo | Branch | Nixpkgs Pin | PyTorch | CUDA | Python |
|------|--------|-------------|---------|------|--------|
| `build-vllm` | `main` | **`0182a36`** | 2.9.1 (wheel) | 12.9 | 3.12, 3.13 |
| `build-pytorch` | `pytorch-2.9` | `6a030d5` | 2.9.1 | 12.9 | 3.13 |

Both repos build the same PyTorch version (2.9.1) with the same CUDA version (12.9) using the same overlay (`cudaPackages_12_9`). They just use different nixpkgs snapshots, so the dependency trees don't match exactly.

## Target State

| Repo | Branch | Nixpkgs Pin | PyTorch | CUDA | Python |
|------|--------|-------------|---------|------|--------|
| `build-pytorch` | `pytorch-2.9` | **`0182a36`** | 2.9.1 | 12.9 | 3.13 |

Same branch, same PyTorch version, same CUDA version. Just a pin bump.

## What the Pin Contains

Nixpkgs revision [`0182a361324364ae3f436a63005877674cf45efb`](https://github.com/NixOS/nixpkgs/tree/0182a361324364ae3f436a63005877674cf45efb):

- **PyTorch 2.9.1** — verified: `nix-instantiate --eval --expr '(...).python313Packages.torch.version'` → `"2.9.1"`
- **vLLM 0.15.1** — the downstream consumer
- **CUDA 12.9** via `cudaPackages_12_9` overlay
- **Python 3.13** via `python313Packages`
- **cuDNN**: check `nixpkgs_pinned.cudaPackages_12_9.cudnn.version` after importing

## Changes Required

### 1. Update the nixpkgs pin in every `.nix` file (required)

All 66 `.nix` files in `.flox/pkgs/` use the same pin. Global find-and-replace:

```
OLD: 6a030d535719c5190187c4cec156f335e95e3211
NEW: 0182a361324364ae3f436a63005877674cf45efb
```

Since both pins ship PyTorch 2.9.1 with CUDA 12.9, this is a low-risk change — it updates the surrounding nixpkgs dependency tree (other system libraries, Python packages, etc.) without changing the core PyTorch or CUDA version.

This is the only strictly required change. It ensures the custom-built torch and vLLM share identical dependency closures (same glibc, libstdc++, Python, CUDA toolkit, etc.), which is necessary for ABI-safe torch substitution in Phase 3.

### 2. Optionally rename `python3Packages` to `python313Packages`

The `pytorch-2.9` recipes use `nixpkgs_pinned.python3Packages.torch`. On pin `0182a36`, `python3` resolves to Python 3.13.11, so `python3Packages` and `python313Packages` are the **same package set**. This rename is not functionally required — it's a readability improvement that makes the Python version explicit and matches the naming convention used in the vLLM recipes (`python313Packages.vllm`).

If desired, global find-and-replace across all 66 `.nix` files:

```
OLD: nixpkgs_pinned.python3Packages.torch
NEW: nixpkgs_pinned.python313Packages.torch
```

### 3. Update README.md

Update the version matrix to reflect the new nixpkgs pin (`0182a36`). The PyTorch version (2.9.1), CUDA version (12.9), and variant names are all unchanged.

### 4. Handle potential build differences

Since the PyTorch version is unchanged, build breakage risk is low. Watch for:

- **Changed nixpkgs packaging**: The nixpkgs `torch` derivation may have different build inputs, patches, or flags between the two snapshots even for the same version. If the build fails, compare the derivation between pins.
- **Changed `.override` parameters**: If the `pytorch.override` function signature changed between snapshots (e.g., `gpuTargets` renamed), the two-stage override will fail.
- **CPU-only and Darwin builds**: These filter out CUDA dependencies from `buildInputs` — if attribute names changed, the filter expressions may need updating.

### 5. Build verification

```bash
# Test one GPU variant (SM90 is the reference)
flox build pytorch-python313-cuda12_9-sm90-avx512

# Verify PyTorch version
./result-pytorch-python313-cuda12_9-sm90-avx512/bin/python -c \
  "import torch; print(torch.__version__)"
# Expected: 2.9.1

# Verify CUDA version
./result-pytorch-python313-cuda12_9-sm90-avx512/bin/python -c \
  "import torch; print(torch.version.cuda)"
# Expected: 12.9

# Verify Python version
./result-pytorch-python313-cuda12_9-sm90-avx512/bin/python --version
# Expected: Python 3.13.x

# Test a CPU-only variant
flox build pytorch-python313-cpu-avx512

# Test Darwin if on macOS
flox build pytorch-python313-darwin-mps
```

## Summary of All Changes

| What | From | To | Required? |
|------|------|----|-----------|
| Nixpkgs pin | `6a030d5` | `0182a36` | **Yes** — needed for ABI-safe substitution |
| Python package set | `python3Packages.torch` | `python313Packages.torch` | No — cosmetic (`python3` = 3.13 on both pins) |
| PyTorch version | 2.9.1 | 2.9.1 (unchanged) | — |
| CUDA version | 12.9 | 12.9 (unchanged) | — |
| CUDA overlay | `cudaPackages_12_9` | `cudaPackages_12_9` (unchanged) | — |
| Filenames | `cuda12_9` | `cuda12_9` (unchanged) | — |
| Recipe structure | — | unchanged | — |

## Files Affected

All 66 `.nix` files in `.flox/pkgs/` need one required global replacement:

1. **Pin hash** (required): `6a030d535719c5190187c4cec156f335e95e3211` → `0182a361324364ae3f436a63005877674cf45efb`
2. **Python package set** (optional): `python3Packages` → `python313Packages`

README.md needs the pin reference updated.

No new files. No deleted files. No structural changes. No new branch.

## Why This Matters

Once both repos share the same nixpkgs pin, we can do Phase 3: substitute the custom-built torch into vLLM's dependency graph via Nix's `override` mechanism. The substitution replaces the pre-built torch wheel with our SM-targeted, CPU-ISA-optimized from-source build, so vLLM gets a PyTorch that matches its exact hardware target — same CUDA kernels, same CPU instruction set, same Python version, same dependency tree.
