// SGEMM 逐步优化：C = alpha * A * B + beta * C（行主序，FP32）。
// v0 朴素 → v1 共享内存分块 → v2 一维线程分块 → v3 二维线程分块
// → v4 float4 向量化 + A 转置存放 → v5 双缓冲，最后与 cuBLAS 对比。
// 为了把注意力放在优化本身，要求 M、N、K 都是 128 的倍数。
#include <cublas_v2.h>

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

// v0：每个线程算 C 的一个元素，所有数据都直接从显存读。
// threadIdx.x 对应列，所以同一个 warp 读 B 的一行是合并访存，读 A 是广播。
__global__ void sgemm_v0_naive(int M, int N, int K, float alpha,
                               const float* A, const float* B, float beta,
                               float* C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

// v1：一个 block 先把 A、B 的 T×T 小块搬进共享内存，再在片上做乘加。
// 每个元素从显存读的次数从 K 次降到 K/T 次。
template <int T>
__global__ void sgemm_v1_smem(int M, int N, int K, float alpha,
                              const float* A, const float* B, float beta,
                              float* C) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * T + ty;
  int col = blockIdx.x * T + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += T) {
    As[ty][tx] = A[row * K + k0 + tx];
    Bs[ty][tx] = B[(k0 + ty) * N + col];
    __syncthreads();
#pragma unroll
    for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// v2：每个线程负责同一列上的 TM 个结果。从共享内存读一次 B 的值，
// 可以和 TM 个 A 的值相乘，共享内存访问次数降为原来的约 1/TM。
template <int BM, int BN, int BK, int TM>
__global__ void __launch_bounds__((BM * BN) / TM)
    sgemm_v2_1d_tile(int M, int N, int K, float alpha, const float* A,
                     const float* B, float beta, float* C) {
  const int c_row = blockIdx.y, c_col = blockIdx.x;
  const int thread_col = threadIdx.x % BN;
  const int thread_row = threadIdx.x / BN;
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  A += c_row * BM * K;
  B += c_col * BN;
  C += c_row * BM * N + c_col * BN;
  const int inner_col_a = threadIdx.x % BK, inner_row_a = threadIdx.x / BK;
  const int inner_col_b = threadIdx.x % BN, inner_row_b = threadIdx.x / BN;
  float results[TM] = {0.f};
  for (int bk = 0; bk < K; bk += BK) {
    As[inner_row_a * BK + inner_col_a] = A[inner_row_a * K + inner_col_a];
    Bs[inner_row_b * BN + inner_col_b] = B[inner_row_b * N + inner_col_b];
    __syncthreads();
    A += BK;
    B += BK * N;
#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
      float b = Bs[dot * BN + thread_col];
#pragma unroll
      for (int r = 0; r < TM; ++r)
        results[r] += As[(thread_row * TM + r) * BK + dot] * b;
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < TM; ++r) {
    int idx = (thread_row * TM + r) * N + thread_col;
    C[idx] = alpha * results[r] + beta * C[idx];
  }
}

// v3：每个线程计算 TM×TN（8×8）的小块。每步从共享内存读 TM+TN 个数，
// 做 TM*TN 次乘加：读 16 个数换 64 次 FMA，计算访存比远高于 v2。
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    sgemm_v3_2d_tile(int M, int N, int K, float alpha, const float* A,
                     const float* B, float beta, float* C) {
  constexpr int kThreads = (BM * BN) / (TM * TN);
  const int c_row = blockIdx.y, c_col = blockIdx.x;
  const int thread_col = threadIdx.x % (BN / TN);
  const int thread_row = threadIdx.x / (BN / TN);
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  A += c_row * BM * K;
  B += c_col * BN;
  C += c_row * BM * N + c_col * BN;
  const int inner_row_a = threadIdx.x / BK, inner_col_a = threadIdx.x % BK;
  constexpr int kStrideA = kThreads / BK;
  const int inner_row_b = threadIdx.x / BN, inner_col_b = threadIdx.x % BN;
  constexpr int kStrideB = kThreads / BN;
  float results[TM * TN] = {0.f};
  float reg_m[TM];
  float reg_n[TN];
  for (int bk = 0; bk < K; bk += BK) {
#pragma unroll
    for (int off = 0; off < BM; off += kStrideA)
      As[(inner_row_a + off) * BK + inner_col_a] =
          A[(inner_row_a + off) * K + inner_col_a];
#pragma unroll
    for (int off = 0; off < BK; off += kStrideB)
      Bs[(inner_row_b + off) * BN + inner_col_b] =
          B[(inner_row_b + off) * N + inner_col_b];
    __syncthreads();
    A += BK;
    B += BK * N;
#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
      for (int i = 0; i < TM; ++i) reg_m[i] = As[(thread_row * TM + i) * BK + dot];
#pragma unroll
      for (int i = 0; i < TN; ++i) reg_n[i] = Bs[dot * BN + thread_col * TN + i];
#pragma unroll
      for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n) results[m * TN + n] += reg_m[m] * reg_n[n];
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      int idx = (thread_row * TM + m) * N + thread_col * TN + n;
      C[idx] = alpha * results[m * TN + n] + beta * C[idx];
    }
}

// v4：全局内存用 float4 一次读 16 字节；A 小块在共享内存里转置存放（As[k][m]），
// 内层循环读 A 时变成连续地址，编译器可以生成 128 位的共享内存加载。
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    sgemm_v4_vectorize(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C) {
  const int c_row = blockIdx.y, c_col = blockIdx.x;
  const int thread_col = threadIdx.x % (BN / TN);
  const int thread_row = threadIdx.x / (BN / TN);
  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BN];
  A += c_row * BM * K;
  B += c_col * BN;
  C += c_row * BM * N + c_col * BN;
  const int inner_row_a = threadIdx.x / (BK / 4);
  const int inner_col_a = threadIdx.x % (BK / 4);
  const int inner_row_b = threadIdx.x / (BN / 4);
  const int inner_col_b = threadIdx.x % (BN / 4);
  float results[TM * TN] = {0.f};
  float reg_m[TM];
  float reg_n[TN];
  for (int bk = 0; bk < K; bk += BK) {
    float4 a4 = reinterpret_cast<const float4*>(
        &A[inner_row_a * K + inner_col_a * 4])[0];
    As[(inner_col_a * 4 + 0) * BM + inner_row_a] = a4.x;
    As[(inner_col_a * 4 + 1) * BM + inner_row_a] = a4.y;
    As[(inner_col_a * 4 + 2) * BM + inner_row_a] = a4.z;
    As[(inner_col_a * 4 + 3) * BM + inner_row_a] = a4.w;
    reinterpret_cast<float4*>(&Bs[inner_row_b * BN + inner_col_b * 4])[0] =
        reinterpret_cast<const float4*>(&B[inner_row_b * N + inner_col_b * 4])[0];
    __syncthreads();
    A += BK;
    B += BK * N;
#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
      for (int i = 0; i < TM; ++i) reg_m[i] = As[dot * BM + thread_row * TM + i];
#pragma unroll
      for (int i = 0; i < TN; ++i) reg_n[i] = Bs[dot * BN + thread_col * TN + i];
#pragma unroll
      for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n) results[m * TN + n] += reg_m[m] * reg_n[n];
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      float4* cp = reinterpret_cast<float4*>(
          &C[(thread_row * TM + m) * N + thread_col * TN + n]);
      float4 c = cp[0];
      c.x = alpha * results[m * TN + n + 0] + beta * c.x;
      c.y = alpha * results[m * TN + n + 1] + beta * c.y;
      c.z = alpha * results[m * TN + n + 2] + beta * c.z;
      c.w = alpha * results[m * TN + n + 3] + beta * c.w;
      cp[0] = c;
    }
}

// v5：共享内存开两份（双缓冲）。计算当前小块的同时，把下一块先读进寄存器，
// 算完再写进另一份缓冲区，让显存读取的延迟被计算掩盖；每轮只需一次 __syncthreads。
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    sgemm_v5_double_buffer(int M, int N, int K, float alpha, const float* A,
                           const float* B, float beta, float* C) {
  const int c_row = blockIdx.y, c_col = blockIdx.x;
  const int thread_col = threadIdx.x % (BN / TN);
  const int thread_row = threadIdx.x / (BN / TN);
  __shared__ float As[2][BK * BM];
  __shared__ float Bs[2][BK * BN];
  A += c_row * BM * K;
  B += c_col * BN;
  C += c_row * BM * N + c_col * BN;
  const int inner_row_a = threadIdx.x / (BK / 4);
  const int inner_col_a = threadIdx.x % (BK / 4);
  const int inner_row_b = threadIdx.x / (BN / 4);
  const int inner_col_b = threadIdx.x % (BN / 4);

  auto load_a = [&](int bk) {
    return reinterpret_cast<const float4*>(
        &A[inner_row_a * K + bk + inner_col_a * 4])[0];
  };
  auto load_b = [&](int bk) {
    return reinterpret_cast<const float4*>(
        &B[(bk + inner_row_b) * N + inner_col_b * 4])[0];
  };
  auto store_tiles = [&](int buf, float4 a4, float4 b4) {
    As[buf][(inner_col_a * 4 + 0) * BM + inner_row_a] = a4.x;
    As[buf][(inner_col_a * 4 + 1) * BM + inner_row_a] = a4.y;
    As[buf][(inner_col_a * 4 + 2) * BM + inner_row_a] = a4.z;
    As[buf][(inner_col_a * 4 + 3) * BM + inner_row_a] = a4.w;
    reinterpret_cast<float4*>(&Bs[buf][inner_row_b * BN + inner_col_b * 4])[0] = b4;
  };

  float results[TM * TN] = {0.f};
  float reg_m[TM];
  float reg_n[TN];
  store_tiles(0, load_a(0), load_b(0));
  __syncthreads();
  int write_buf = 1;
  for (int bk = 0; bk < K; bk += BK) {
    const int read_buf = write_buf ^ 1;
    const bool has_next = bk + BK < K;
    float4 next_a, next_b;
    if (has_next) {  // 先发出下一块的显存读取，结果暂存在寄存器
      next_a = load_a(bk + BK);
      next_b = load_b(bk + BK);
    }
#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
        reg_m[i] = As[read_buf][dot * BM + thread_row * TM + i];
#pragma unroll
      for (int i = 0; i < TN; ++i)
        reg_n[i] = Bs[read_buf][dot * BN + thread_col * TN + i];
#pragma unroll
      for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n) results[m * TN + n] += reg_m[m] * reg_n[n];
    }
    if (has_next) store_tiles(write_buf, next_a, next_b);
    __syncthreads();
    write_buf ^= 1;
  }
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      float4* cp = reinterpret_cast<float4*>(
          &C[(thread_row * TM + m) * N + thread_col * TN + n]);
      float4 c = cp[0];
      c.x = alpha * results[m * TN + n + 0] + beta * c.x;
      c.y = alpha * results[m * TN + n + 1] + beta * c.y;
      c.z = alpha * results[m * TN + n + 2] + beta * c.z;
      c.w = alpha * results[m * TN + n + 3] + beta * c.w;
      cp[0] = c;
    }
}

// 行主序的 C = A*B 等价于列主序的 C^T = B^T * A^T，所以把 A、B 交换后调用 cuBLAS。
static void cublas_sgemm(cublasHandle_t h, int M, int N, int K, float alpha,
                         const float* A, const float* B, float beta, float* C) {
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N,
                           A, K, &beta, C, N));
}

int main(int argc, char** argv) {
  print_device_info();
  auto sizes = parse_int_list(arg_str(argc, argv, "--sizes", "1024,2048,4096"));
  const int warmup = arg_int(argc, argv, "--warmup", 5);
  const int iters = arg_int(argc, argv, "--iters", 20);
  const std::string csv = arg_str(argc, argv, "--csv", "results/sgemm.csv");

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  // 默认数学模式不启用 TF32，保证和我们的纯 FP32 kernel 是公平对比。
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  for (int S : sizes) {
    if (S % 128 != 0) {
      std::fprintf(stderr, "size %d is not a multiple of 128, skipped\n", S);
      continue;
    }
    const int M = S, N = S, K = S;
    const float alpha = 1.f, beta = 0.f;
    std::vector<float> hA(size_t(M) * K), hB(size_t(K) * N), hRef(size_t(M) * N),
        hC(size_t(M) * N);
    fill_random(hA, 1);
    fill_random(hB, 2);
    float *dA, *dB, *dC, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRef, hRef.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dRef, 0, hRef.size() * sizeof(float)));
    cublas_sgemm(handle, M, N, K, alpha, dA, dB, beta, dRef);
    CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, hRef.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const double flops = 2.0 * M * N * K;
    double cublas_gflops = 0.0;
    struct Variant {
      std::string name;
      std::function<void()> launch;
    };
    std::vector<Variant> variants = {
        {"cublas", [&] { cublas_sgemm(handle, M, N, K, alpha, dA, dB, beta, dC); }},
        {"v0_naive",
         [&] {
           dim3 block(32, 32), grid(ceil_div(N, 32), ceil_div(M, 32));
           sgemm_v0_naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
         }},
        {"v1_smem",
         [&] {
           dim3 block(32, 32), grid(N / 32, M / 32);
           sgemm_v1_smem<32><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
         }},
        {"v2_1d_tile",
         [&] {
           constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
           dim3 grid(N / BN, M / BM);
           sgemm_v2_1d_tile<BM, BN, BK, TM>
               <<<grid, (BM * BN) / TM>>>(M, N, K, alpha, dA, dB, beta, dC);
         }},
        {"v3_2d_tile",
         [&] {
           constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
           dim3 grid(N / BN, M / BM);
           sgemm_v3_2d_tile<BM, BN, BK, TM, TN>
               <<<grid, (BM * BN) / (TM * TN)>>>(M, N, K, alpha, dA, dB, beta, dC);
         }},
        {"v4_vectorize",
         [&] {
           constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
           dim3 grid(N / BN, M / BM);
           sgemm_v4_vectorize<BM, BN, BK, TM, TN>
               <<<grid, (BM * BN) / (TM * TN)>>>(M, N, K, alpha, dA, dB, beta, dC);
         }},
        {"v5_double_buffer",
         [&] {
           constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
           dim3 grid(N / BN, M / BM);
           sgemm_v5_double_buffer<BM, BN, BK, TM, TN>
               <<<grid, (BM * BN) / (TM * TN)>>>(M, N, K, alpha, dA, dB, beta, dC);
         }},
    };

    std::printf("\n== SGEMM M=N=K=%d ==\n%-18s %10s %10s %9s %12s %s\n", S,
                "version", "ms", "GFLOPS", "%cuBLAS", "max_abs_err", "check");
    for (auto& v : variants) {
      if (v.name != "cublas" && !selected(argc, argv, v.name)) continue;
      CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
      v.launch();
      CHECK_LAST_KERNEL();
      CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
      CompareResult cr = compare(hC.data(), hRef.data(), hC.size(), 5e-3, 1e-3);
      float ms = time_kernel_ms(v.launch, warmup, iters);
      double gflops = flops / (ms * 1e-3) / 1e9;
      if (v.name == "cublas") cublas_gflops = gflops;
      double pct = cublas_gflops > 0 ? 100.0 * gflops / cublas_gflops : 0.0;
      std::printf("%-18s %10.3f %10.1f %8.1f%% %12.2e %s\n", v.name.c_str(), ms,
                  gflops, pct, cr.max_abs_err, cr.ok ? "PASS" : "FAIL");
      char row[256];
      std::snprintf(row, sizeof(row), "sgemm,%s,%d,%d,%d,%.4f,%.1f,%.1f,%.3e,%s",
                    v.name.c_str(), M, N, K, ms, gflops, pct, cr.max_abs_err,
                    cr.ok ? "PASS" : "FAIL");
      append_csv(csv, "op,version,M,N,K,ms,gflops,pct_cublas,max_abs_err,check", row);
    }
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dRef));
  }
  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}
