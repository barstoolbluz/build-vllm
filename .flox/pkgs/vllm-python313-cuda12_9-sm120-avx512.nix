# vLLM 0.15.1 for NVIDIA Blackwell (SM120: RTX 5090, RTX PRO 6000) — AVX-512
# CUDA 12.9 — Requires NVIDIA driver 560+
# Custom PyTorch 2.9.1 built from source (SM120 + AVX-512)
{ pkgs ? import <nixpkgs> {} }:
let
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/0182a361324364ae3f436a63005877674cf45efb.tar.gz";
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
  variantName = "vllm-python313-cuda12_9-sm120-avx512";
  cpuISA = (import ./lib/cpu-isa.nix).avx512;
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
      description = "vLLM 0.15.1 for NVIDIA RTX 5090/RTX PRO 6000 (SM120) [CUDA 12.9, custom PyTorch AVX-512]";
      platforms = [ platform ];
    };
  })
