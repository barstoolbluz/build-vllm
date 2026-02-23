# CUDA Build Parallelism and Swap Thrashing

## Problem

Building vLLM variants (e.g., `flox build vllm-python312-cuda12_9-sm90`) crashes the desktop environment. The shell, window manager (GNOME), and terminal all become unresponsive, requiring a session restart.

This happened twice on 2026-02-22 while building `vllm-python312-cuda12_9-sm90`.

## Root Cause

The Nix daemon config (`/etc/nix/flox.conf`) sets `cores = 0`, which means "use all available cores." On a 64-core machine, this passes `-j64` to CMake/ninja during the vLLM CUDA compilation phase.

vLLM 0.13.0 has **406 CUDA compilation units**, including:

- Flash Attention 2 (SM80 fallback kernels)
- Flash Attention 3 (Hopper SM90 kernels)
- FlashMLA (SM90 + SM100 kernels)
- CUTLASS c3x scaled_mm / grouped_mm (SM90) — worst offenders for RAM
- GPTQ Marlin quantization kernels
- MoE Marlin kernels

Each `nvcc` invocation for CUTLASS/FlashAttention template-heavy files uses **3-8 GB of RAM**. At `-j64`, peak memory demand reaches 200-400 GB.

### Machine specs

- 125 GB RAM
- 127 GB swap (triple NVMe PCIe 4.0 RAID 0)
- 64 cores
- 458 GB root filesystem

### Why the desktop dies

1. RAM fills at 125 GB
2. Kernel pages out to swap — every process that touches a swapped-out page blocks on I/O
3. GNOME Shell, terminal, Xorg, DBus all get swapped out because nvcc processes are the most active allocators
4. Input latency goes to seconds or minutes; the desktop appears frozen
5. The OOM killer does **not** fire — total virtual memory (252 GB) may not be fully exhausted
6. System is functionally dead without being technically dead; thrashing is worse than a clean OOM kill

Fast NVMe swap doesn't solve this. Even at millions of IOPS, random page fault latency (microseconds) is three orders of magnitude slower than RAM (nanoseconds). The page reclaim path (kswapd) becomes the system bottleneck.

## Fix

Added `NIX_BUILD_CORES = 16;` to the `overrideAttrs` block in each variant `.nix` file. This caps parallel CUDA compilation at 16 jobs.

At 16 parallel nvcc processes (16 x 3-8 GB = 48-128 GB peak), the build stays within 125 GB physical RAM and should not touch swap meaningfully.

### Example diff

```nix
nixpkgs_pinned.python312Packages.vllm.overrideAttrs (oldAttrs: {
    pname = variantName;
    requiredSystemFeatures = [ "big-parallel" ];
    # Limit parallel CUDA compilation — CUTLASS/FlashAttn templates use 3-8GB each;
    # unrestricted -j64 on this machine causes swap thrashing that kills the desktop
    NIX_BUILD_CORES = 16;
    ...
})
```

### Applied to

- [x] `vllm-python312-cuda12_9-sm90.nix`
- [ ] Other 9 variants (apply the same line before building them)

## Tuning

If 16 jobs still causes pressure (e.g., on a machine with less RAM), reduce further. Conservative formula:

```
max_jobs = floor(available_RAM_GB / 8)
```

For 125 GB RAM: `floor(125 / 8) = 15`, so 16 is right at the edge. 12 would be more conservative.

If builds are too slow, you can increase toward 24 on this machine — the worst-case CUTLASS templates are ~8 GB, but most kernels are 3-4 GB, so average usage per job is lower than the worst case.

## Diagnosis Commands

If this happens again, check after reboot:

```bash
# Check for OOM kills in kernel log
dmesg | grep -i -E "oom|killed|out of memory"

# Check systemd journal
journalctl -k --no-pager -n 100 | grep -i -E "oom|kill"

# Check current memory
free -h

# Check for zombie/leftover build processes
ps aux | grep -E "nix-build|nix-daemon|nvcc|flox" | grep -v grep

# Check the Nix build log for the last build attempt
ls -lt /nix/var/log/nix/drvs/ | head -5
# Then decompress and inspect:
bzcat /nix/var/log/nix/drvs/<dir>/<file>.bz2 | tail -20
```

## Alternative Approaches Not Taken

- **Changing `cores` in `/etc/nix/flox.conf`**: Would affect all Nix builds system-wide, not just vLLM. Left at `cores = 0` for non-CUDA builds that benefit from full parallelism.
- **Using `MAKEFLAGS` or `cmakeFlags`**: `NIX_BUILD_CORES` is the standard Nix mechanism and is cleaner.
- **Reducing `max-jobs`**: Not relevant here since only one derivation (vllm itself) is building at a time during the CUDA compilation phase.
