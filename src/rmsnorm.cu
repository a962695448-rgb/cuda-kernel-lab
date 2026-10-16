// RMSNorm（LLaMA 等大模型使用）：y = x / sqrt(mean(x^2) + eps) * w，按行归一化。
// v0 一个线程一行 → v1 一个 block 一行 + 共享内存树形归约 → v2 一个 block 一行 + float4 + warp shuffle。
// 访存受限：有效带宽按读 x、写 y 计算（w 很小，忽略）。
#include "common.cuh"
#include "warp_utils.cuh"

__global__ void rmsnorm_v0_thread_row(const float* x, const float* w, float* y, int rows,
                                      int hidden, float eps) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* xr = x + size_t(r) * hidden;
  float ss = 0.f;
  for (int i = 0; i < hidden; ++i) ss += xr[i] * xr[i];
  const float scale = rsqrtf(ss / hidden + eps);
  for (int i = 0; i < hidden; ++i) y[size_t(r) * hidden + i] = xr[i] * scale * w[i];
}

// v1：block 内线程跨步读取（合并访存），平方和用共享内存树形归约。
__global__ void rmsnorm_v1_block_smem(const float* x, const float* w, float* y, int rows,
                                      int hidden, float eps) {
  extern __shared__ float s[];
  const float* xr = x + size_t(blockIdx.x) * hidden;
  float* yr = y + size_t(blockIdx.x) * hidden;
  float ss = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) ss += xr[i] * xr[i];
  s[threadIdx.x] = ss;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(s[0] / hidden + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) yr[i] = xr[i] * scale * w[i];
}

// v2：float4 一次读 4 个数，平方和用 warp shuffle 归约（只用 32 个 float 的共享内存）。
// 要求 hidden 是 4 的倍数。
__global__ void rmsnorm_v2_vec4_shuffle(const float* x, const float* w, float* y, int rows,
                                        int hidden, float eps) {
  const float4* xr = reinterpret_cast<const float4*>(x + size_t(blockIdx.x) * hidden);
  const float4* w4 = reinterpret_cast<const float4*>(w);
  float4* yr = reinterpret_cast<float4*>(y + size_t(blockIdx.x) * hidden);
  const int h4 = hidden / 4;
  float ss = 0.f;
  for (int i = threadIdx.x; i < h4; i += blockDim.x) {
    float4 v = xr[i];
    ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  ss = block_reduce_sum(ss);
  const float scale = rsqrtf(ss / hidden + eps);
  for (int i = threadIdx.x; i < h4; i += blockDim.x) {
    float4 v = xr[i], g = w4[i];
    yr[i] = make_float4(v.x * scale * g.x, v.y * scale * g.y, v.z * scale * g.z,
                        v.w * scale * g.w);
  }
}

static void rmsnorm_cpu(const float* x, const float* w, float* y, int rows, int hidden,
                        float eps) {
  for (int r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int i = 0; i < hidden; ++i) ss += double(x[size_t(r) * hidden + i]) * x[size_t(r) * hidden + i];
    double scale = 1.0 / std::sqrt(ss / hidden + eps);
    for (int i = 0; i < hidden; ++i)
      y[size_t(r) * hidden + i] = float(x[size_t(r) * hidden + i] * scale * w[i]);
  }
}

int main(int argc, char** argv) {
  print_device_info();
  const int rows = arg_int(argc, argv, "--rows", 8192);
  const int hidden = arg_int(argc, argv, "--hidden", 4096);
  const int warmup = arg_int(argc, argv, "--warmup", 5);
  const int iters = arg_int(argc, argv, "--iters", 20);
  const std::string csv = arg_str(argc, argv, "--csv", "results/rmsnorm.csv");
  const float eps = 1e-6f;
  if (hidden % 4 != 0) {
    std::fprintf(stderr, "hidden must be a multiple of 4\n");
    return 1;
  }
  const double copy_bw = measure_copy_bandwidth_gbs();
  std::printf("# device-to-device copy bandwidth: %.1f GB/s\n", copy_bw);

  const size_t n = size_t(rows) * hidden;
  std::vector<float> hx(n), hw(hidden), hy(n), href(n);
  fill_random(hx, 4);
  fill_random(hw, 5, 0.5f, 1.5f);
  rmsnorm_cpu(hx.data(), hw.data(), href.data(), rows, hidden, eps);
  float *dx, *dw, *dy;
  CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dw, hidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, hw.data(), hidden * sizeof(float), cudaMemcpyHostToDevice));

  const int threads = 256;
  struct Variant {
    std::string name;
    std::function<void()> launch;
  };
  std::vector<Variant> variants = {
      {"v0_thread_row",
       [&] { rmsnorm_v0_thread_row<<<ceil_div(rows, 128), 128>>>(dx, dw, dy, rows, hidden, eps); }},
      {"v1_block_smem",
       [&] {
         rmsnorm_v1_block_smem<<<rows, threads, threads * sizeof(float)>>>(dx, dw, dy, rows,
                                                                            hidden, eps);
       }},
      {"v2_vec4_shuffle",
       [&] { rmsnorm_v2_vec4_shuffle<<<rows, threads>>>(dx, dw, dy, rows, hidden, eps); }},
  };

  std::printf("\n== RMSNorm rows=%d hidden=%d ==\n%-18s %10s %10s %8s %12s %s\n", rows, hidden,
              "version", "ms", "GB/s", "%copy", "max_abs_err", "check");
  for (auto& v : variants) {
    if (!selected(argc, argv, v.name)) continue;
    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    v.launch();
    CHECK_LAST_KERNEL();
    CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    CompareResult cr = compare(hy.data(), href.data(), n, 1e-5, 1e-4);
    float ms = time_kernel_ms(v.launch, warmup, iters);
    double gbs = 2.0 * n * 4 / (ms * 1e-3) / 1e9;
    std::printf("%-18s %10.3f %10.1f %7.1f%% %12.2e %s\n", v.name.c_str(), ms, gbs,
                100.0 * gbs / copy_bw, cr.max_abs_err, cr.ok ? "PASS" : "FAIL");
    char row[256];
    std::snprintf(row, sizeof(row), "rmsnorm,%s,%d,%d,%.4f,%.1f,%.1f,%.3e,%s", v.name.c_str(),
                  rows, hidden, ms, gbs, 100.0 * gbs / copy_bw, cr.max_abs_err,
                  cr.ok ? "PASS" : "FAIL");
    append_csv(csv, "op,version,rows,hidden,ms,gbs,pct_copy_bw,max_abs_err,check", row);
  }
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dw));
  CUDA_CHECK(cudaFree(dy));
  return 0;
}
