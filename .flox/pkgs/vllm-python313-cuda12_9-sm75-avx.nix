# vLLM 0.15.1 for NVIDIA Turing (SM75: T4, RTX 2080 Ti) — AVX
# CUDA 12.9 — Requires NVIDIA driver 560+
# Custom PyTorch 2.9.1 built from source (SM75 + AVX)
{ pkgs ? import <nixpkgs> {} }:
let
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/0182a361324364ae3f436a63005877674cf45efb.tar.gz";
  }) {
    config = {
      allowUnfree = true;
      cudaSupport = true;
      cudaCapabilities = [ "7.5" ];
    };
    overlays = [ (final: prev: { cudaPackages = final.cudaPackages_12_9; }) ];
  };
  inherit (nixpkgs_pinned) lib;

  # ── Variant-specific configuration ──────────────────────────────────
  smCapability = "7.5";
  variantName = "vllm-python313-cuda12_9-sm75-avx";
  cpuISA = (import ./lib/cpu-isa.nix).avx;
  platform = "x86_64-linux";

  # ── Custom Python package set with from-source torch ────────────────
  python313Custom = nixpkgs_pinned.python313.override {
    packageOverrides = self: super: {
      torch = import ./lib/custom-torch.nix {
        inherit lib smCapability cpuISA platform;
        torchBase = super.torch;
      };
      bitsandbytes = super.bitsandbytes.overrideAttrs (oldAttrs: {
        cmakeFlags = (oldAttrs.cmakeFlags or []) ++ [
          "-DCOMPUTE_CAPABILITY=${builtins.replaceStrings ["."] [""] smCapability}"
        ];
      });
    };
  };
  # ────────────────────────────────────────────────────────────────────
in
  python313Custom.pkgs.vllm.overrideAttrs (oldAttrs: {
    pname = variantName;
    requiredSystemFeatures = [ "big-parallel" ];
    NIX_BUILD_CORES = 16;
    meta = oldAttrs.meta // {
      description = "vLLM 0.15.1 for NVIDIA T4/RTX 2080 Ti (SM75) [CUDA 12.9, custom PyTorch AVX]";
      platforms = [ platform ];
    };
  })
