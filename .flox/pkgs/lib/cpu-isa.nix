{
  # ── x86_64 ISAs ───────────────────────────────────────────────────────
  avx = {
    name = "avx";
    flags = [ "-mavx" ];
  };
  avx2 = {
    name = "avx2";
    flags = [ "-mavx2" "-mfma" "-mf16c" ];
  };
  avx512 = {
    name = "avx512";
    flags = [ "-mavx512f" "-mavx512dq" "-mavx512vl" "-mavx512bw" "-mfma" ];
  };
  avx512bf16 = {
    name = "avx512bf16";
    flags = [ "-mavx512f" "-mavx512dq" "-mavx512vl" "-mavx512bw" "-mavx512bf16" "-mfma" ];
  };
  avx512vnni = {
    name = "avx512vnni";
    flags = [ "-mavx512f" "-mavx512dq" "-mavx512vl" "-mavx512bw" "-mavx512vnni" "-mfma" ];
  };
  # ── aarch64 ISAs ──────────────────────────────────────────────────────
  armv8_2 = {
    name = "armv8_2";
    flags = [ "-march=armv8.2-a+fp16+dotprod" ];
  };
  armv9 = {
    name = "armv9";
    flags = [ "-march=armv9-a+sve2+bf16+i8mm" ];
  };
}
