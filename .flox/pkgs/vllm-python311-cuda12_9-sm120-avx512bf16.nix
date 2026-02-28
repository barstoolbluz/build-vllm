# vLLM 0.14.1 for NVIDIA Blackwell (SM120: RTX 5090, RTX PRO 6000) — AVX-512 BF16
# CUDA 12.9 — Requires NVIDIA driver 560+
# Custom PyTorch 2.9.1 built from source (SM120 + AVX-512 BF16)
{ pkgs ? import <nixpkgs> {} }:
let
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/46336d4d6980ae6f136b45c8507b17787eb186a0.tar.gz";
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
  variantName = "vllm-python311-cuda12_9-sm120-avx512bf16";
  cpuISA = (import ./lib/cpu-isa.nix).avx512bf16;
  platform = "x86_64-linux";

  # ── Custom Python package set with from-source torch ────────────────
  python311Custom = nixpkgs_pinned.python311.override {
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
  python311Custom.pkgs.vllm.overrideAttrs (oldAttrs: {
    pname = variantName;
    version = "0.14.1";
    src = nixpkgs_pinned.fetchFromGitHub {
      owner = "vllm-project";
      repo = "vllm";
      tag = "v0.14.1";
      hash = "sha256-qoC3RpjnqbMR3JwkJfquIyuXhLyW+uGG+zSCCek4G2U=";
    };
    requiredSystemFeatures = [ "big-parallel" ];
    NIX_BUILD_CORES = 16;
    meta = oldAttrs.meta // {
      description = "vLLM 0.14.1 for NVIDIA RTX 5090/RTX PRO 6000 (SM120) [CUDA 12.9, custom PyTorch AVX-512 BF16]";
      platforms = [ platform ];
    };
  })
