# PyTorch Build Recipes: Align to vLLM Nixpkgs Pin

## Goal

Add CUDA 12.9 PyTorch 2.10.0 variants to `build-pytorch` that use the **same nixpkgs pin** as `build-vllm/main`. This enables Phase 3: substituting a custom-built PyTorch into vLLM builds (replacing the pre-built wheel from nixpkgs).

## Current State

| Repo | Branch | Nixpkgs Pin | PyTorch | CUDA | Python | Recipe Pattern |
|------|--------|-------------|---------|------|--------|----------------|
| `build-vllm` | `main` | `0182a36` | 2.10.0 (wheel) | 12.9 | 3.12, 3.13 | Self-contained fetchTarball |
| `build-pytorch` | `main` | Flox-managed | 2.8.0 | 12.8 | 3.13 | Flox function args |
| `build-pytorch` | `pytorch-2.9` | `6a030d5` | 2.9.1 | 12.9 | 3.13 | Self-contained fetchTarball |
| `build-pytorch` | `pytorch-2.10` | `6a030d5` / `2017d6d` | 2.10.0 | 13.0 / 13.1 | 3.13 | Self-contained fetchTarball + source override + magma/CCCL patches |

## Why Not Use the Existing `pytorch-2.10` Branch?

The existing `pytorch-2.10` branch builds PyTorch 2.10.0 for CUDA **13.0/13.1** (128 variants). Those recipes are fundamentally different from what we need:

- They use a nixpkgs pin (`6a030d5`) that does **not** have PyTorch 2.10.0 natively — they override `torch.src` to fetch 2.10.0 from GitHub
- They include complex overlays: magma CUDA 13 clockrate patch, CCCL compatibility symlinks, `FindCUDAToolkit.cmake` workaround
- They set `allowBroken = true` and clear `patches = []`
- They use a `pytorch210-` pname prefix

By contrast, the vLLM nixpkgs pin (`0182a36`) **already ships PyTorch 2.10.0 natively** — no source override needed. The recipes will be as simple as the `pytorch-2.9` ones: just a pin swap.

## Target State

A new branch with CUDA 12.9 variants using the vLLM pin:

| Repo | Branch | Nixpkgs Pin | PyTorch | CUDA | Python |
|------|--------|-------------|---------|------|--------|
| `build-pytorch` | `pytorch-2.10-cuda12_9` | **`0182a36`** | **2.10.0** | **12.9** | 3.13 |

## What the Pin Contains

Nixpkgs revision [`0182a361324364ae3f436a63005877674cf45efb`](https://github.com/NixOS/nixpkgs/tree/0182a361324364ae3f436a63005877674cf45efb) provides:

- **PyTorch 2.10.0** natively (no source override needed)
- **CUDA 12.9** via `cudaPackages_12_9` overlay
- **CUTLASS v4.2.1** primary + v3.9.0+ for FlashMLA Blackwell
- **Python 3.13** via `python313Packages`
- **cuDNN**: check `nixpkgs_pinned.cudaPackages_12_9.cudnn.version` after importing

This is the same pin used by `build-vllm/main`, where PyTorch 2.10.0 is a pre-built wheel dependency of vLLM 0.15.1.

## Changes Required

### 1. Create a new branch from `pytorch-2.9`

The `pytorch-2.9` branch is the right starting point — it already uses CUDA 12.9, the `cudaPackages_12_9` overlay, `cuda12_9` in filenames/variantNames, and the simple self-contained fetchTarball recipe pattern.

```bash
git checkout pytorch-2.9
git checkout -b pytorch-2.10-cuda12_9
```

### 2. Update the nixpkgs pin in every `.nix` file

All 66 `.nix` files in `.flox/pkgs/` use the same pin. Global find-and-replace:

```
OLD: 6a030d535719c5190187c4cec156f335e95e3211
NEW: 0182a361324364ae3f436a63005877674cf45efb
```

This single change upgrades the underlying PyTorch from 2.9.1 to 2.10.0 (since the new pin ships 2.10.0 natively). No source overrides, no new overlays, no patches.

### 3. Switch from `python3Packages` to `python313Packages`

The `pytorch-2.9` recipes use `nixpkgs_pinned.python3Packages.torch`. The vLLM Python 3.13 variants use `nixpkgs_pinned.python313Packages.vllm`, which internally depends on `python313Packages.torch`. For torch substitution to work, the PyTorch builds must use the same Python package set so the derivation paths match.

Global find-and-replace across all 66 `.nix` files:

```
OLD: nixpkgs_pinned.python3Packages.torch
NEW: nixpkgs_pinned.python313Packages.torch
```

This affects two patterns:

**GPU variants** (two-stage override):
```nix
# OLD:
(nixpkgs_pinned.python3Packages.torch.override {
  cudaSupport = true;
  gpuTargets = [ gpuArchSM ];
}).overrideAttrs (oldAttrs: { ... })

# NEW:
(nixpkgs_pinned.python313Packages.torch.override {
  cudaSupport = true;
  gpuTargets = [ gpuArchSM ];
}).overrideAttrs (oldAttrs: { ... })
```

**CPU-only and Darwin variants** (single-stage override):
```nix
# OLD:
nixpkgs_pinned.python3Packages.torch.overrideAttrs (oldAttrs: { ... })

# NEW:
nixpkgs_pinned.python313Packages.torch.overrideAttrs (oldAttrs: { ... })
```

**Note**: If `python3` on the new pin already resolves to Python 3.13, this is a no-op. But using the explicit `python313Packages` makes the intent unambiguous and guarantees alignment with vLLM regardless of the pin's default Python version.

### 4. Verify the package attribute exists on the new pin

Before doing a full build, confirm `python313Packages.torch` exists:

```bash
nix eval --expr '(import (builtins.fetchTarball {
  url = "https://github.com/NixOS/nixpkgs/archive/0182a361324364ae3f436a63005877674cf45efb.tar.gz";
}) {
  config = { allowUnfree = true; cudaSupport = true; };
  overlays = [ (final: prev: { cudaPackages = final.cudaPackages_12_9; }) ];
}).python313Packages.torch.version'
```

Expected: `"2.10.0"` (or `2.10.0.postX`).

### 5. Update metadata

The `pytorch-2.9` `.nix` files do **not** reference PyTorch version in their code — they only reference CUDA 12.9 in the `preConfigure` echo block and `meta.longDescription`. Since CUDA 12.9 is unchanged, these values are already correct.

What does need updating:

- **`meta.longDescription`**: Each file's longDescription says `CUDA: 12.9 with compute capability X.Y` — this is still correct, no change needed.
- **`meta.longDescription` driver values**: Each file lists a per-SM minimum driver (e.g., "Driver: NVIDIA 525+ required" for SM90). These are GPU architecture minimums and are independent of the CUDA toolkit. **Leave these as-is** — they reflect the minimum driver for the GPU architecture, not the CUDA toolkit. The CUDA 12.9 toolkit minimum (driver 560+) is a separate, higher requirement documented at the branch/README level.
- **README.md**: Update the version matrix table to reflect PyTorch 2.10.0, nixpkgs pin `0182a36`, and minimum driver 560+ (matching the vLLM documentation for CUDA 12.9 on this pin).

### 6. Handle potential build breakage

The pin swap upgrades PyTorch from 2.9.1 to 2.10.0. Since the new pin's nixpkgs already packages 2.10.0 with its full dependency tree, most things should "just work." Watch for:

- **New or changed dependencies**: The nixpkgs packaging of 2.10.0 may add/remove build inputs vs 2.9.1. Since we're using `overrideAttrs` (not replacing the dependency list), new deps are picked up automatically.
- **Changed `.override` parameters**: If the `pytorch.override` function signature changed between the two nixpkgs revisions (e.g., `gpuTargets` renamed or removed), the two-stage override will fail. Check by attempting a build.
- **cuDNN compatibility**: Verify the cuDNN version in the new pin satisfies PyTorch 2.10's requirements.
- **CPU-only and Darwin builds**: These filter out CUDA dependencies from `buildInputs` — if the attribute names changed, the filter expressions may need updating.

### 7. Build verification

```bash
# Stage new files (required for flox to discover them)
git add .flox/pkgs/

# Test one GPU variant (SM90 is the reference)
flox build pytorch-python313-cuda12_9-sm90-avx512

# Verify PyTorch version
./result-pytorch-python313-cuda12_9-sm90-avx512/bin/python -c \
  "import torch; print(torch.__version__)"
# Expected: 2.10.0

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

| What | From (`pytorch-2.9`) | To (`pytorch-2.10-cuda12_9`) |
|------|----------------------|------------------------------|
| Nixpkgs pin | `6a030d5` | `0182a36` |
| PyTorch version (from pin) | 2.9.1 | 2.10.0 |
| CUDA version | 12.9 (unchanged) | 12.9 (unchanged) |
| CUDA overlay | `cudaPackages_12_9` (unchanged) | `cudaPackages_12_9` (unchanged) |
| Python package set | `python3Packages.torch` | `python313Packages.torch` |
| Filenames | `cuda12_9` (unchanged) | `cuda12_9` (unchanged) |
| Recipe complexity | Simple (no source overrides) | Simple (no source overrides) |

## Files Affected

All 66 `.nix` files in `.flox/pkgs/` on the `pytorch-2.9` branch need two global replacements:

1. **Pin hash**: `6a030d535719c5190187c4cec156f335e95e3211` → `0182a361324364ae3f436a63005877674cf45efb`
2. **Python package set**: `python3Packages` → `python313Packages`

Documentation files (README.md, etc.) need version and pin reference updates.

No new files. No deleted files. No structural changes.

## Why This Matters

Once `build-pytorch` has a branch on the same nixpkgs pin as `build-vllm`, we can do Phase 3: substitute the custom-built torch into vLLM's dependency graph via Nix's `override` mechanism. The substitution replaces the pre-built torch wheel with our SM-targeted, CPU-ISA-optimized from-source build, so vLLM gets a PyTorch that matches its exact hardware target — same CUDA kernels, same CPU instruction set, same Python version, same dependency tree.
