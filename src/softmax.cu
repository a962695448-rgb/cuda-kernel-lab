// 按行 softmax：y[r][c] = exp(x[r][c] - max_r) / sum_c exp(x[r][c] - max_r)。
// v0 一个线程一行（三遍读显存）→ v1 一个 block 一行（三遍，但读取合并）
// → v2 online softmax（一遍同时求 max 和 sum，共两遍）→ v3 一个 warp 一行，整行缓存在寄存器（只读一遍）。
// 访存受限：用有效带宽 = (读 x + 写 y) 字节数 / 时间 衡量。
#include "common.cuh"
#include "warp_utils.cuh"

// v0：线程 r 独自处理第 r 行。相邻线程访问相隔一整行的地址，访存不合并。
__global__ void softmax_v0_thread_row(const float* x, float* y, int rows, int cols) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* xr = x + size_t(r) * cols;
  float* yr = y + size_t(r) * cols;
  float m = -FLT_MAX;
  for (int c = 0; c < cols; ++c) m = fmaxf(m, xr[c]);
  float s = 0.f;
  for (int c = 0; c < cols; ++c) s += expf(xr[c] - m);
  for (int c = 0; c < cols; ++c) yr[c] = expf(xr[c] - m) / s;
}

// v1：一个 block 负责一行，线程按列跨步读取（合并访存），用 block 归约求 max 和 sum。
__global__ void softmax_v1_block_row(const float* x, float* y, int rows, int cols) {
  const float* xr = x + size_t(blockIdx.x) * cols;
  float* yr = y + size_t(blockIdx.x) * cols;
  float m = -FLT_MAX;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, xr[c]);
  m = block_reduce_max(m);
  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) s += expf(xr[c] - m);
  s = block_reduce_sum(s);
  const float inv = 1.f / s;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = expf(xr[c] - m) * inv;
}

// online softmax 的合并规则：(m1, d1) 与 (m2, d2) 合并为
// m = max(m1, m2)，d = d1 * exp(m1 - m) + d2 * exp(m2 - m)。FlashAttention 也基于这个规则。
struct MD {
  float m, d;
};
__device__ __forceinline__ MD md_merge(MD a, MD b) {
  float m = fmaxf(a.m, b.m);
  return {m, a.d * expf(a.m - m) + b.d * expf(b.m - m)};
}
__device__ __forceinline__ MD warp_reduce_md(MD v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    MD o{__shfl_xor_sync(0xffffffff, v.m, offset), __shfl_xor_sync(0xffffffff, v.d, offset)};
    v = md_merge(v, o);
  }
  return v;
}

// v2：每个线程一遍扫描就同时维护“当前最大值”和“按当前最大值缩放的和”，
// 统计阶段从两遍读降为一遍，总共读两遍 x。
__global__ void softmax_v2_online(const float* x, float* y, int rows, int cols) {
  const float* xr = x + size_t(blockIdx.x) * cols;
  float* yr = y + size_t(blockIdx.x) * cols;
  MD md{-FLT_MAX, 0.f};
  for (int c = threadIdx.x; c < cols; c += blockDim.x) md = md_merge(md, {xr[c], 1.f});
  md = warp_reduce_md(md);
  __shared__ MD warp_md[32];
  __shared__ MD row_md;
  const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
  if (lane == 0) warp_md[wid] = md;
  __syncthreads();
  if (wid == 0) {
    MD v = lane < blockDim.x / 32 ? warp_md[lane] : MD{-FLT_MAX, 0.f};
    v = warp_reduce_md(v);
    if (lane == 0) row_md = v;
  }
  __syncthreads();
  const float m = row_md.m, inv = 1.f / row_md.d;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = expf(xr[c] - m) * inv;
}

// v3：一个 warp 负责一行，每个 lane 把 COLS/32 个数读进寄存器后全程在寄存器里计算：
// x 只从显存读一遍，y 写一遍，达到访存下限。只适用于编译期已知、且每 lane 寄存器放得下的列数。
template <int COLS>
__global__ void softmax_v3_warp_register(const float* x, float* y, int rows) {
  constexpr int kPerLane = COLS / 32;
  const int warps_per_block = blockDim.x / 32;
  const int r = blockIdx.x * warps_per_block + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (r >= rows) return;
  const float* xr = x + size_t(r) * COLS;
  float* yr = y + size_t(r) * COLS;
  float v[kPerLane];
  float m = -FLT_MAX;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    v[i] = xr[i * 32 + lane];  // 第 i 轮整个 warp 读连续的 32 个数，合并访存
    m = fmaxf(m, v[i]);
  }
  m = warp_reduce_max(m);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    v[i] = expf(v[i] - m);
    s += v[i];
  }
  s = warp_reduce_sum(s);
  const float inv = 1.f / s;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) yr[i * 32 + lane] = v[i] * inv;
}

static void softmax_cpu(const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + size_t(r) * cols;
    double m = -1e300, s = 0.0;
    for (int c = 0; c < cols; ++c) m = std::max(m, double(xr[c]));
    for (int c = 0; c < cols; ++c) s += std::exp(double(xr[c]) - m);
    for (int c = 0; c < cols; ++c) y[size_t(r) * cols + c] = float(std::exp(double(xr[c]) - m) / s);
  }
}

int main(int argc, char** argv) {
  print_device_info();
  // 默认 65536×1024×4 B = 256 MB，大于 RTX 5090 的 96 MB L2，测到的是显存带宽而不是 L2 带宽。
  const int rows = arg_int(argc, argv, "--rows", 65536);
  const int cols = 1024;  // v3 是按 1024 列实例化的模板，这里固定列数
  const int warmup = arg_int(argc, argv, "--warmup", 5);
  const int iters = arg_int(argc, argv, "--iters", 20);
  const std::string csv = arg_str(argc, argv, "--csv", "results/softmax.csv");
  const double copy_bw = measure_copy_bandwidth_gbs();
  std::printf("# device-to-device copy bandwidth: %.1f GB/s\n", copy_bw);

  const size_t n = size_t(rows) * cols;
  std::vector<float> hx(n), hy(n), href(n);
  fill_random(hx, 3, -8.f, 8.f);
  softmax_cpu(hx.data(), href.data(), rows, cols);
  float *dx, *dy;
  CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  struct Variant {
    std::string name;
    std::function<void()> launch;
  };
  std::vector<Variant> variants = {
      {"v0_thread_row",
       [&] { softmax_v0_thread_row<<<ceil_div(rows, 128), 128>>>(dx, dy, rows, cols); }},
      {"v1_block_row", [&] { softmax_v1_block_row<<<rows, 256>>>(dx, dy, rows, cols); }},
      {"v2_online", [&] { softmax_v2_online<<<rows, 256>>>(dx, dy, rows, cols); }},
      {"v3_warp_register",
       [&] { softmax_v3_warp_register<1024><<<ceil_div(rows, 4), 128>>>(dx, dy, rows); }},
  };

  std::printf("\n== Softmax rows=%d cols=%d ==\n%-18s %10s %10s %8s %12s %s\n", rows, cols,
              "version", "ms", "GB/s", "%copy", "max_abs_err", "check");
  for (auto& v : variants) {
    if (!selected(argc, argv, v.name)) continue;
    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    v.launch();
    CHECK_LAST_KERNEL();
    CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    CompareResult cr = compare(hy.data(), href.data(), n, 1e-6, 1e-4);
    float ms = time_kernel_ms(v.launch, warmup, iters);
    double gbs = 2.0 * n * 4 / (ms * 1e-3) / 1e9;
    std::printf("%-18s %10.3f %10.1f %7.1f%% %12.2e %s\n", v.name.c_str(), ms, gbs,
                100.0 * gbs / copy_bw, cr.max_abs_err, cr.ok ? "PASS" : "FAIL");
    char row[256];
    std::snprintf(row, sizeof(row), "softmax,%s,%d,%d,%.4f,%.1f,%.1f,%.3e,%s", v.name.c_str(),
                  rows, cols, ms, gbs, 100.0 * gbs / copy_bw, cr.max_abs_err,
                  cr.ok ? "PASS" : "FAIL");
    append_csv(csv, "op,version,rows,cols,ms,gbs,pct_copy_bw,max_abs_err,check", row);
  }
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  return 0;
}
