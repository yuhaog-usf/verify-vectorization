#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---- KokkosFloat4: custom struct (same as in your Kokkos port) ----
struct alignas(16) KokkosFloat4
{
    float x;
    float y;
    float z;
    float w;
};

// ---- Kernel using CUDA built-in float4 ----
__global__ void kernel_cuda_float4(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * 4;
    if (base + 3 < N)
    {
        // Vectorized 128-bit load
        float4 val = reinterpret_cast<const float4*>(input)[tid];
        // Simple operation on each component
        val.x *= 2.0f;
        val.y *= 2.0f;
        val.z *= 2.0f;
        val.w *= 2.0f;
        // Vectorized 128-bit store
        reinterpret_cast<float4*>(output)[tid] = val;
    }
}

// ---- Kernel using KokkosFloat4 (custom struct) ----
__global__ void kernel_kokkos_float4(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * 4;
    if (base + 3 < N)
    {
        // Vectorized 128-bit load (same pattern, but with custom struct)
        KokkosFloat4 val = reinterpret_cast<const KokkosFloat4*>(input)[tid];
        // Same operation
        val.x *= 2.0f;
        val.y *= 2.0f;
        val.z *= 2.0f;
        val.w *= 2.0f;
        // Vectorized 128-bit store
        reinterpret_cast<KokkosFloat4*>(output)[tid] = val;
    }
}

// ---- Benchmark both kernels ----
int main()
{
    const int N = 1 << 24;  // 16M floats (~64 MB)
    const int bytes = N * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_input[i] = (float)i;

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N / 4 + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    // Warmup
    kernel_cuda_float4<<<blocks, threads>>>(d_input, d_output, N);
    kernel_kokkos_float4<<<blocks, threads>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    // Benchmark CUDA float4
    const int ITERS = 100;
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
        kernel_cuda_float4<<<blocks, threads>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float bw_cuda = (2.0f * bytes * ITERS) / (ms / 1000.0f) / 1e9;
    printf("CUDA float4:    %.3f ms total, %.2f GB/s\n", ms, bw_cuda);

    // Benchmark KokkosFloat4
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
        kernel_kokkos_float4<<<blocks, threads>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float bw_kokkos = (2.0f * bytes * ITERS) / (ms / 1000.0f) / 1e9;
    printf("KokkosFloat4:   %.3f ms total, %.2f GB/s\n", ms, bw_kokkos);

    printf("\nDifference: %.2f%%\n", 100.0f * (bw_cuda - bw_kokkos) / bw_cuda);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
