# vLLM 0.13.0 for NVIDIA Blackwell (SM120: RTX 5090, RTX PRO 6000)
# CUDA 12.9 — Requires NVIDIA driver 560+
{ pkgs ? import <nixpkgs> {} }:
let
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/ed142ab1b3a092c4d149245d0c4126a5d7ea00b0.tar.gz";
  }) {
    config = {
      allowUnfree = true;
      cudaSupport = true;
      cudaCapabilities = [ "12.0" ];
    };
    overlays = [ (final: prev: { cudaPackages = final.cudaPackages_12_9; }) ];
  };
  inherit (nixpkgs_pinned) lib;

  # ── Variant-specific configuration ──────────────────────────────────
  smCapability = "12.0";
  variantName = "vllm-python312-cuda12_9-sm120";
  # ────────────────────────────────────────────────────────────────────
in
  nixpkgs_pinned.python312Packages.vllm.overrideAttrs (oldAttrs: {
    pname = variantName;
    requiredSystemFeatures = [ "big-parallel" ];
    # Limit parallel CUDA compilation — CUTLASS/FlashAttn templates use 3-8GB each;
    # unrestricted -j64 on this machine causes swap thrashing that kills the desktop
    NIX_BUILD_CORES = 16;
    # Remove bitsandbytes — incompatible with CUDA 12.9 CCCL headers + GCC 15
    # BnB quantization (NF4/INT8) unavailable; all other vLLM features work
    propagatedBuildInputs = builtins.filter (dep:
      lib.getName dep != "bitsandbytes"
    ) (oldAttrs.propagatedBuildInputs or []);
    meta = oldAttrs.meta // {
      description = "vLLM 0.13.0 for NVIDIA RTX 5090/RTX PRO 6000 (SM120) [CUDA 12.9]";
      platforms = [ "x86_64-linux" ];
    };
  })
