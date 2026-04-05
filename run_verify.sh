#!/bin/bash
# Verification script: compare float4 vs KokkosFloat4 vectorization
# Usage: bash run_verify.sh
# Run this on the H100 machine after scp-ing the verify_vectorization/ folder

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================="
echo " Step 1: Generate PTX (assembly comparison)"
echo "============================================="

# Compile to PTX for H100 (sm_90)
nvcc -ptx -arch=sm_90 -O3 test_float4_ptx.cu -o test_float4_ptx.ptx
echo "PTX generated: test_float4_ptx.ptx"
echo ""

# Extract vectorized load/store instructions for CUDA float4 kernel
echo "--- CUDA float4 kernel: vectorized instructions ---"
echo "(Searching between kernel_cuda_float4 and kernel_kokkos_float4)"
sed -n '/\.visible .entry.*kernel_cuda_float4/,/\.visible .entry.*kernel_kokkos_float4/p' test_float4_ptx.ptx \
    | grep -E "ld\.global|st\.global" | head -20
echo ""

# Extract vectorized load/store instructions for KokkosFloat4 kernel
echo "--- KokkosFloat4 kernel: vectorized instructions ---"
echo "(Searching after kernel_kokkos_float4)"
sed -n '/\.visible .entry.*kernel_kokkos_float4/,$p' test_float4_ptx.ptx \
    | grep -E "ld\.global|st\.global" | head -20
echo ""

echo "KEY: Look for 'ld.global.v4.f32' (vectorized) vs 'ld.global.f32' (scalar)"
echo "     If both show v4, they are equivalent. If KokkosFloat4 shows scalar loads, it's slower."
echo ""

echo "============================================="
echo " Step 2: Generate SASS (actual GPU assembly)"
echo "============================================="

# Compile to binary, then dump SASS
nvcc -arch=sm_90 -O3 test_float4_ptx.cu -o test_float4_binary
cuobjdump -sass test_float4_binary > test_float4_sass.txt 2>/dev/null || true

if [ -f test_float4_sass.txt ]; then
    echo "SASS generated: test_float4_sass.txt"
    echo ""
    echo "--- CUDA float4 kernel: memory instructions ---"
    sed -n '/kernel_cuda_float4/,/kernel_kokkos_float4/p' test_float4_sass.txt \
        | grep -iE "LDG|STG" | head -10
    echo ""
    echo "--- KokkosFloat4 kernel: memory instructions ---"
    sed -n '/kernel_kokkos_float4/,/\.nv/p' test_float4_sass.txt \
        | grep -iE "LDG|STG" | head -10
    echo ""
    echo "KEY: Look for 'LDG.E.128' (vectorized 128-bit) vs 'LDG.E' (32-bit scalar)"
else
    echo "cuobjdump not found or failed, skipping SASS analysis"
fi

echo ""
echo "============================================="
echo " Step 3: Bandwidth benchmark"
echo "============================================="
./test_float4_binary

echo ""
echo "============================================="
echo " Done! Summary:"
echo "============================================="
echo " - PTX file:  $SCRIPT_DIR/test_float4_ptx.ptx"
echo " - SASS file: $SCRIPT_DIR/test_float4_sass.txt"
echo " - If bandwidth numbers are within 1-2%, vectorization is equivalent."
echo " - Check PTX/SASS above to confirm both use 128-bit loads/stores."
