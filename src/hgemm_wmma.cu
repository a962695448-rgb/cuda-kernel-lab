// HGEMM：FP16 输入、FP32 累加，C = A * B（行主序）。
// v0 用普通 CUDA 核心（不走 Tensor Core），v1 用 WMMA 调用 Tensor Core，
// 对照 cuBLAS GemmEx（同样是 FP16 输入 + FP32 累加）。要求 M、N 是 64 的倍数，K 是 32 的倍数。
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "common.cuh"

#define CUBLAS_CHECK(call)                                                   \
  do {                                                                       \
    cublasStatus_t st_ = (call);                                             \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(st_), __FILE__, \
                   __LINE__);                                                \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

using namespace nvcuda;

// v0：共享内存分块，FP16 读入后转成 FP32 在 CUDA 核心上乘加，作为“不用 Tensor Core”的对照。
template <int T>
__global__ void hgemm_v0_smem(int M, int N, int K, const half* A,
                              const half* B, float* C) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * T + ty;
  int col = blockIdx.x * T + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += T) {
    As[ty][tx] = __half2float(A[row * K + k0 + tx]);
    Bs[ty][tx] = __half2float(B[(k0 + ty) * N + col]);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  C[row * N + col] = acc;
}

// v1：一个 block 4 个 warp（2×2），负责 C 的 64×64；每个 warp 算 32×32，
// 即 2×2 个 16×16 的 WMMA 片段。A、B 小块用 16 字节（8 个 half）一次搬进共享内存。
template <int BM, int BN, int BK>
__global__ void __launch_bounds__(128)
    hgemm_v1_wmma(int M, int N, int K, const half* A, const half* B, float* C) {
  constexpr int WM = 32, WN = 32;
  constexpr int kWarpsN = BN / WN;
  __shared__ __align__(32) half As[BM * BK];
  __shared__ __align__(32) half Bs[BK * BN];
  const int warp_id = threadIdx.x / 32;
  const int wr = warp_id / kWarpsN, wc = warp_id % kWarpsN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  const half* a_blk = A + size_t(blockIdx.y) * BM * K;
  const half* b_blk = B + blockIdx.x * BN;
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = threadIdx.x; i < BM * BK / 8; i += blockDim.x) {
      int r = i / (BK / 8), c = (i % (BK / 8)) * 8;
      *reinterpret_cast<int4*>(&As[r * BK + c]) =
          *reinterpret_cast<const int4*>(&a_blk[size_t(r) * K + k0 + c]);
    }
    for (int i = threadIdx.x; i < BK * BN / 8; i += blockDim.x) {
      int r = i / (BN / 8), c = (i % (BN / 8)) * 8;
      *reinterpret_cast<int4*>(&Bs[r * BN + c]) =
          *reinterpret_cast<const int4*>(&b_blk[size_t(k0 + r) * N + c]);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        wmma::load_matrix_sync(a[i], &As[(wr * WM + i * 16) * BK + kk], BK);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::load_matrix_sync(b[j], &Bs[kk * BN + wc * WN + j * 16], BN);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float* c = C + size_t(blockIdx.y * BM + wr * WM + i * 16) * N +
                 blockIdx.x * BN + wc * WN + j * 16;
      wmma::store_matrix_sync(c, acc[i][j], N, wmma::mem_row_major);
    }
}

static void cublas_hgemm(cublasHandle_t h, int M, int N, int K, const half* A,
                         const half* B, float* C) {
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                            CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C,
                            CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}

int main(int argc, char** argv) {
  print_device_info();
  auto sizes = parse_int_list(arg_str(argc, argv, "--sizes", "1024,2048,4096"));
  const int warmup = arg_int(argc, argv, "--warmup", 5);
  const int iters = arg_int(argc, argv, "--iters", 20);
  const std::string csv = arg_str(argc, argv, "--csv", "results/hgemm.csv");
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  for (int S : sizes) {
    if (S % 64 != 0) continue;
    const int M = S, N = S, K = S;
    std::vector<float> fA(size_t(M) * K), fB(size_t(K) * N);
    fill_random(fA, 1);
    fill_random(fB, 2);
    std::vector<half> hA(fA.size()), hB(fB.size());
    for (size_t i = 0; i < fA.size(); ++i) hA[i] = __float2half(fA[i]);
    for (size_t i = 0; i < fB.size(); ++i) hB[i] = __float2half(fB[i]);
    std::vector<float> hC(size_t(M) * N), hRef(size_t(M) * N);
    half *dA, *dB;
    float *dC, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRef, hRef.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(half), cudaMemcpyHostToDevice));
    cublas_hgemm(handle, M, N, K, dA, dB, dRef);
    CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, hRef.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const double flops = 2.0 * M * N * K;
    double cublas_tflops = 0.0;
    struct Variant {
      std::string name;
      std::function<void()> launch;
    };
    std::vector<Variant> variants = {
        {"cublas_tensorcore", [&] { cublas_hgemm(handle, M, N, K, dA, dB, dC); }},
        {"v0_cuda_core",
         [&] {
           dim3 block(32, 32), grid(N / 32, M / 32);
           hgemm_v0_smem<32><<<grid, block>>>(M, N, K, dA, dB, dC);
         }},
        {"v1_wmma",
         [&] {
           constexpr int BM = 64, BN = 64, BK = 32;
           dim3 grid(N / BN, M / BM);
           hgemm_v1_wmma<BM, BN, BK><<<grid, 128>>>(M, N, K, dA, dB, dC);
         }},
    };
    std::printf("\n== HGEMM (fp16 in, fp32 acc) M=N=K=%d ==\n%-18s %10s %10s %9s %12s %s\n",
                S, "version", "ms", "TFLOPS", "%cuBLAS", "max_abs_err", "check");
    for (auto& v : variants) {
      if (v.name != "cublas_tensorcore" && !selected(argc, argv, v.name)) continue;
      CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
      v.launch();
      CHECK_LAST_KERNEL();
      CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
      CompareResult cr = compare(hC.data(), hRef.data(), hC.size(), 1e-2, 1e-3);
      float ms = time_kernel_ms(v.launch, warmup, iters);
      double tflops = flops / (ms * 1e-3) / 1e12;
      if (v.name == "cublas_tensorcore") cublas_tflops = tflops;
      double pct = 100.0 * tflops / cublas_tflops;
      std::printf("%-18s %10.3f %10.2f %8.1f%% %12.2e %s\n", v.name.c_str(), ms,
                  tflops, pct, cr.max_abs_err, cr.ok ? "PASS" : "FAIL");
      char row[256];
      std::snprintf(row, sizeof(row), "hgemm,%s,%d,%d,%d,%.4f,%.2f,%.1f,%.3e,%s",
                    v.name.c_str(), M, N, K, ms, tflops, pct, cr.max_abs_err,
                    cr.ok ? "PASS" : "FAIL");
      append_csv(csv, "op,version,M,N,K,ms,tflops,pct_cublas,max_abs_err,check", row);
    }
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dRef));
  }
  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}
