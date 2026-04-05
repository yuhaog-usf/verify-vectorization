# Verification: KokkosFloat4 vs CUDA float4

## Test Environment
- Machine: RIKEN H100
- GPU: NVIDIA H100
- CUDA: 12.6
- Arch: sm_90
- Optimization: -O3

## How to Reproduce
```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
bash run_verify.sh
```

## Results

### PTX Assembly (compiler intermediate representation)
Both kernels generate identical vectorized 128-bit instructions:
```
CUDA float4:    ld.global.nc.v4.f32  /  st.global.v4.f32
KokkosFloat4:   ld.global.nc.v4.f32  /  st.global.v4.f32
```

### SASS Assembly (actual GPU machine code)
Both kernels generate identical 128-bit memory operations:
```
CUDA float4:    LDG.E.128.CONSTANT  /  STG.E.128
KokkosFloat4:   LDG.E.128.CONSTANT  /  STG.E.128
```

### Bandwidth Benchmark (100 iterations, 64MB data)
```
CUDA float4:    4.064 ms total, 3302.76 GB/s
KokkosFloat4:   4.057 ms total, 3308.65 GB/s
Difference:     -0.18%
```

## Conclusion

`KokkosFloat4` (custom `alignas(16)` struct) and CUDA built-in `float4` are **identical** in:
1. **PTX**: same `v4.f32` vectorized load/store instructions
2. **SASS**: same `LDG.E.128` / `STG.E.128` 128-bit memory operations
3. **Performance**: within 0.18% (noise level)

The `nvcc` compiler recognizes `alignas(16)` structs with 4 floats and generates the same vectorized instructions as the built-in `float4` type. No code changes needed for the Kokkos port.
