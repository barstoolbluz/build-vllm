# vLLM 0.15.1 for NVIDIA Turing (SM75: T4, RTX 2080 Ti) — AVX-512 BF16
# CUDA 12.8 — Requires NVIDIA driver 550+
# Custom PyTorch 2.9.1 built from source (SM75 + AVX-512 BF16)
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
    overlays = [ (final: prev: { cudaPackages = final.cudaPackages_12_8; }) ];
  };
  inherit (nixpkgs_pinned) lib;

  # ── Variant-specific configuration ──────────────────────────────────
  smCapability = "7.5";
  variantName = "vllm-python313-cuda12_8-sm75-avx512bf16";
  cpuISA = (import ./lib/cpu-isa.nix).avx512bf16;
  platform = "x86_64-linux";

  # ── Custom Python package set with from-source torch ────────────────
  python313Custom = nixpkgs_pinned.python313.override {
    packageOverrides = self: super: {
      torch = import ./lib/custom-torch.nix {
        inherit lib smCapability cpuISA platform;
        torchBase = super.torch;
        cudaVersionTag = "cuda12_8";
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
      description = "vLLM 0.15.1 for NVIDIA T4/RTX 2080 Ti (SM75) [CUDA 12.8, custom PyTorch AVX-512 BF16]";
      platforms = [ platform ];
    };
  })
