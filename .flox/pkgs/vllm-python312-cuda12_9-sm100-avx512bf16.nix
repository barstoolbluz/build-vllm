# vLLM 0.14.1 for NVIDIA Blackwell DC (SM100: B100, B200, GB200) — AVX-512 BF16
# CUDA 12.9 — Requires NVIDIA driver 560+
# Custom PyTorch 2.9.1 built from source (SM100 + AVX-512 BF16)
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
  variantName = "vllm-python312-cuda12_9-sm100-avx512bf16";
  cpuISA = (import ./lib/cpu-isa.nix).avx512bf16;
  platform = "x86_64-linux";

  # ── Custom Python package set with from-source torch ────────────────
  python312Custom = nixpkgs_pinned.python312.override {
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
  python312Custom.pkgs.vllm.overrideAttrs (oldAttrs: {
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
      description = "vLLM 0.14.1 for NVIDIA B100/B200/GB200 (SM100) [CUDA 12.9, custom PyTorch AVX-512 BF16]";
      platforms = [ platform ];
    };
  })
