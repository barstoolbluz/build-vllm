# vLLM 0.15.1 for NVIDIA Hopper (SM90: H100, H200, L40S) — ARMv8.2
# CUDA 12.8 — Requires NVIDIA driver 550+
# Custom PyTorch 2.9.1 built from source (SM90 + ARMv8.2)
{ pkgs ? import <nixpkgs> {} }:
let
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/0182a361324364ae3f436a63005877674cf45efb.tar.gz";
  }) {
    config = {
      allowUnfree = true;
      cudaSupport = true;
      cudaCapabilities = [ "9.0" ];
    };
    overlays = [ (final: prev: { cudaPackages = final.cudaPackages_12_8; }) ];
  };
  inherit (nixpkgs_pinned) lib;

  # ── Variant-specific configuration ──────────────────────────────────
  smCapability = "9.0";
  variantName = "vllm-python313-cuda12_8-sm90-armv8_2";
  cpuISA = (import ./lib/cpu-isa.nix).armv8_2;
  platform = "aarch64-linux";

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
      description = "vLLM 0.15.1 for NVIDIA H100/H200/L40S (SM90) [CUDA 12.8, custom PyTorch ARMv8.2]";
      platforms = [ platform ];
    };
  })
