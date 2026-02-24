# vLLM 0.14.0 for NVIDIA Grace Blackwell (SM100: GB200, GB300)
# CUDA 12.9 — Requires NVIDIA driver 560+
{ pkgs ? import <nixpkgs> {} }:
let
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/46336d4d6980ae6f136b45c8507b17787eb186a0.tar.gz";
  }) {
    config = {
      allowUnfree = true;
      cudaSupport = true;
      cudaCapabilities = [ "10.0" ];
    };
    overlays = [ (final: prev: { cudaPackages = final.cudaPackages_12_9; }) ];
  };
  inherit (nixpkgs_pinned) lib;

  # ── Variant-specific configuration ──────────────────────────────────
  smCapability = "10.0";
  variantName = "vllm-python312-cuda12_9-sm100-arm64";
  # ── bitsandbytes: restrict to single SM (CCCL 2.8.2 IMPL20 bug) ────
  customBitsandbytes = nixpkgs_pinned.python312Packages.bitsandbytes.overrideAttrs (oldAttrs: {
    cmakeFlags = (oldAttrs.cmakeFlags or []) ++ [
      "-DCOMPUTE_CAPABILITY=${builtins.replaceStrings ["."] [""] smCapability}"
    ];
  });
  # ────────────────────────────────────────────────────────────────────
in
  nixpkgs_pinned.python312Packages.vllm.overrideAttrs (oldAttrs: {
    pname = variantName;
    requiredSystemFeatures = [ "big-parallel" ];
    # Limit parallel CUDA compilation — CUTLASS/FlashAttn templates use 3-8GB each;
    # unrestricted -j64 on this machine causes swap thrashing that kills the desktop
    NIX_BUILD_CORES = 16;
    # Replace bitsandbytes with single-SM build (CCCL 2.8.2 IMPL20 macro bug)
    propagatedBuildInputs = builtins.map (dep:
      if lib.getName dep == "bitsandbytes" then customBitsandbytes else dep
    ) (oldAttrs.propagatedBuildInputs or []);
    meta = oldAttrs.meta // {
      description = "vLLM 0.14.0 for NVIDIA Grace Blackwell GB200/GB300 (SM100) [CUDA 12.9]";
      platforms = [ "aarch64-linux" ];
    };
  })
