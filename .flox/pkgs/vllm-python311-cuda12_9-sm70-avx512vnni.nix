# vLLM 0.14.1 for NVIDIA Volta (SM70: V100, Titan V) — AVX-512 VNNI
# CUDA 12.9 — Requires NVIDIA driver 560+
# Custom PyTorch 2.9.1 built from source (SM70 + AVX-512 VNNI)
{ pkgs ? import <nixpkgs> {} }:
let
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/46336d4d6980ae6f136b45c8507b17787eb186a0.tar.gz";
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
  variantName = "vllm-python311-cuda12_9-sm70-avx512vnni";
  cpuISA = (import ./lib/cpu-isa.nix).avx512vnni;
  platform = "x86_64-linux";

  # ── Custom Python package set with from-source torch ────────────────
  python311Custom = nixpkgs_pinned.python311.override {
    packageOverrides = self: super: {
      torch = import ./lib/custom-torch.nix {
        inherit lib smCapability cpuISA platform;
        torchBase = super.torch;
        extraPreConfigure = "export USE_CUDNN=0";
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
      description = "vLLM 0.14.1 for NVIDIA V100/Titan V (SM70) [CUDA 12.9, custom PyTorch AVX-512 VNNI]";
      platforms = [ platform ];
    };
  })
