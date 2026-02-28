# vLLM 0.14.0 for NVIDIA Blackwell DC (SM100: B100, B200, GB200) — AVX-512 BF16
# CUDA 12.8 — Requires NVIDIA driver 550+
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
    overlays = [ (final: prev: { cudaPackages = final.cudaPackages_12_8; }) ];
  };
  inherit (nixpkgs_pinned) lib;

  # ── Variant-specific configuration ──────────────────────────────────
  smCapability = "10.0";
  variantName = "vllm-python311-cuda12_8-sm100-avx512bf16";
  cpuISA = (import ./lib/cpu-isa.nix).avx512bf16;
  platform = "x86_64-linux";

  # ── Custom Python package set with from-source torch ────────────────
  python311Custom = nixpkgs_pinned.python311.override {
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
  python311Custom.pkgs.vllm.overrideAttrs (oldAttrs: {
    pname = variantName;
    requiredSystemFeatures = [ "big-parallel" ];
    NIX_BUILD_CORES = 16;
    meta = oldAttrs.meta // {
      description = "vLLM 0.14.0 for NVIDIA B100/B200/GB200 (SM100) [CUDA 12.8, custom PyTorch AVX-512 BF16]";
      platforms = [ platform ];
    };
  })
