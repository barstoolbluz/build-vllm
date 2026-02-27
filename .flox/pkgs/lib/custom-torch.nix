# Builds custom PyTorch from source with SM-specific GPU targeting and CPU ISA flags.
# Used inside packageOverrides to replace torch across the entire Python package set.
#
# Arguments:
#   lib:                nixpkgs lib
#   torchBase:          the original python3XXPackages.torch (from `super` in packageOverrides)
#   smCapability:       GPU arch in dotted format, e.g., "9.0"
#   cpuISA:             CPU ISA record from cpu-isa.nix (e.g., { name = "avx512"; flags = [...]; })
#   platform:           "x86_64-linux" or "aarch64-linux"
#   extraPreConfigure:  additional shell commands for preConfigure (e.g., "export USE_CUDNN=0")
#   cudaVersionTag:     tag for pname differentiation (e.g., "cuda12_9", "cuda12_8")
{ lib, torchBase, smCapability, cpuISA, platform, extraPreConfigure ? "", cudaVersionTag ? "cuda12_9" }:

(torchBase.override {
  cudaSupport = true;
  gpuTargets = [ smCapability ];
}).overrideAttrs (oldAttrs: {
  pname = "pytorch-custom-${cudaVersionTag}-sm${builtins.replaceStrings ["."] [""] smCapability}-${cpuISA.name}";

  passthru = oldAttrs.passthru // {
    gpuArch = smCapability;
    blasProvider = "cublas";
    inherit (cpuISA) name;
  };

  ninjaFlags = [ "-j16" ];
  requiredSystemFeatures = [ "big-parallel" ];

  preConfigure = (oldAttrs.preConfigure or "") + ''
    export CXXFLAGS="${lib.concatStringsSep " " cpuISA.flags} $CXXFLAGS"
    export CFLAGS="${lib.concatStringsSep " " cpuISA.flags} $CFLAGS"
    export MAX_JOBS=16
  '' + extraPreConfigure;

  meta = oldAttrs.meta // {
    platforms = [ platform ];
  };
})
