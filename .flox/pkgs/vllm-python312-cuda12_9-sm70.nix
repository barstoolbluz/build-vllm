# vLLM 0.15.1 for NVIDIA Volta (SM70: V100)
# CUDA 12.9 — Requires NVIDIA driver 560+
{ pkgs ? import <nixpkgs> {} }:
let
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/0182a361324364ae3f436a63005877674cf45efb.tar.gz";
  }) {
    config = {
      allowUnfree = true;
      cudaSupport = true;
      cudaCapabilities = [ "7.0" ];
    };
    overlays = [ (final: prev: { cudaPackages = final.cudaPackages_12_9; }) ];
  };
  inherit (nixpkgs_pinned) lib;

  # ── Variant-specific configuration ──────────────────────────────────
  smCapability = "7.0";
  variantName = "vllm-python312-cuda12_9-sm70";
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
      description = "vLLM 0.15.1 for NVIDIA V100 (SM70) [CUDA 12.9]";
      platforms = [ "x86_64-linux" ];
    };
  })
