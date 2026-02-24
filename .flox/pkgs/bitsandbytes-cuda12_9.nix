# bitsandbytes for CUDA 12.9 — SM90 single-arch build
# CCCL 2.8.2 (CUDA 12.9) has a missing _CCCL_PP_SPLICE_WITH_IMPL20 macro;
# building all 19 SM architectures triggers the bug. Restricting to a single
# SM via COMPUTE_CAPABILITY keeps the macro arg count below the broken range.
# Standalone build target for testing; vLLM variants inline the same pattern.
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
    overlays = [ (final: prev: { cudaPackages = final.cudaPackages_12_9; }) ];
  };
in
  nixpkgs_pinned.python312Packages.bitsandbytes.overrideAttrs (oldAttrs: {
    cmakeFlags = (oldAttrs.cmakeFlags or []) ++ [
      "-DCOMPUTE_CAPABILITY=90"
    ];
  })
