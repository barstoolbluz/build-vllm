# vLLM 0.15.1 for NVIDIA Ada Lovelace (SM89: RTX 4090, L4, L40)
# CUDA 12.9 — Requires NVIDIA driver 560+
{ pkgs ? import <nixpkgs> {} }:
let
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/0182a361324364ae3f436a63005877674cf45efb.tar.gz";
  }) {
    config = {
      allowUnfree = true;
      cudaSupport = true;
      cudaCapabilities = [ "8.9" ];
    };
    overlays = [ (final: prev: { cudaPackages = final.cudaPackages_12_9; }) ];
  };
  inherit (nixpkgs_pinned) lib;

  # ── Variant-specific configuration ──────────────────────────────────
  smCapability = "8.9";
  variantName = "vllm-python313-cuda12_9-sm89";
  # ── bitsandbytes: restrict to single SM (CCCL 2.8.2 IMPL20 bug) ────
  customBitsandbytes = nixpkgs_pinned.python313Packages.bitsandbytes.overrideAttrs (oldAttrs: {
    cmakeFlags = (oldAttrs.cmakeFlags or []) ++ [
      "-DCOMPUTE_CAPABILITY=${builtins.replaceStrings ["."] [""] smCapability}"
    ];
  });
  # ────────────────────────────────────────────────────────────────────
in
  nixpkgs_pinned.python313Packages.vllm.overrideAttrs (oldAttrs: {
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
      description = "vLLM 0.15.1 for NVIDIA RTX 4090/L4/L40 (SM89) [CUDA 12.9]";
      platforms = [ "x86_64-linux" ];
    };
  })
