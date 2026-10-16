// 矩阵转置 out[c][r] = in[r][c]（M×N → N×M）。用来演示合并访存和共享内存 bank conflict。
// v0 朴素（读合并、写不合并）→ v1 共享内存 32×32 分块（读写都合并，但有 32 路 bank conflict）
// → v2 分块 + 每行多填 1 个 float（[32][33]），消除 bank conflict。
#include "common.cuh"

constexpr int kTile = 32;
constexpr int kRowsPerPass = 8;  // block = 32×8 线程，每个线程搬 4 个元素

__global__ void transpose_v0_naive(const float* in, float* out, int M, int N) {
  int c = blockIdx.x * kTile + threadIdx.x;
  int r0 = blockIdx.y * kTile + threadIdx.y;
  for (int j = 0; j < kTile; j += kRowsPerPass) {
    int r = r0 + j;
    if (r < M && c < N) out[size_t(c) * M + r] = in[size_t(r) * N + c];  // 写入跨度为 M，不合并
  }
}

// PAD = 0：tile[x][y] 按列读取时，同一 warp 的 32 个线程访问相隔 32 个 float 的地址，
// 全部落在同一个 bank，形成 32 路冲突。PAD = 1 时地址错开一个 bank，冲突消失。
template <int PAD>
__global__ void transpose_smem(const float* in, float* out, int M, int N) {
  __shared__ float tile[kTile][kTile + PAD];
  int c = blockIdx.x * kTile + threadIdx.x;
  int r = blockIdx.y * kTile + threadIdx.y;
  for (int j = 0; j < kTile; j += kRowsPerPass)
    if (r + j < M && c < N) tile[threadIdx.y + j][threadIdx.x] = in[size_t(r + j) * N + c];
  __syncthreads();
  // 交换 block 坐标后再按行写出，写入也是合并的
  c = blockIdx.y * kTile + threadIdx.x;
  r = blockIdx.x * kTile + threadIdx.y;
  for (int j = 0; j < kTile; j += kRowsPerPass)
    if (r + j < N && c < M) out[size_t(r + j) * M + c] = tile[threadIdx.x][threadIdx.y + j];
}

int main(int argc, char** argv) {
  print_device_info();
  const int M = arg_int(argc, argv, "--m", 8192);
  const int N = arg_int(argc, argv, "--n", 8192);
  const int warmup = arg_int(argc, argv, "--warmup", 5);
  const int iters = arg_int(argc, argv, "--iters", 20);
  const std::string csv = arg_str(argc, argv, "--csv", "results/transpose.csv");
  const double copy_bw = measure_copy_bandwidth_gbs();
  std::printf("# device-to-device copy bandwidth: %.1f GB/s\n", copy_bw);

  const size_t n = size_t(M) * N;
  std::vector<float> hin(n), hout(n), href(n);
  fill_random(hin, 6);
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) href[size_t(c) * M + r] = hin[size_t(r) * N + c];
  float *din, *dout;
  CUDA_CHECK(cudaMalloc(&din, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(din, hin.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(kTile, kRowsPerPass), grid(ceil_div(N, kTile), ceil_div(M, kTile));
  struct Variant {
    std::string name;
    std::function<void()> launch;
  };
  std::vector<Variant> variants = {
      {"v0_naive", [&] { transpose_v0_naive<<<grid, block>>>(din, dout, M, N); }},
      {"v1_smem_conflict", [&] { transpose_smem<0><<<grid, block>>>(din, dout, M, N); }},
      {"v2_smem_padded", [&] { transpose_smem<1><<<grid, block>>>(din, dout, M, N); }},
  };

  std::printf("\n== Transpose %dx%d ==\n%-18s %10s %10s %8s %s\n", M, N, "version", "ms",
              "GB/s", "%copy", "check");
  for (auto& v : variants) {
    if (!selected(argc, argv, v.name)) continue;
    CUDA_CHECK(cudaMemset(dout, 0, n * sizeof(float)));
    v.launch();
    CHECK_LAST_KERNEL();
    CUDA_CHECK(cudaMemcpy(hout.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = std::memcmp(hout.data(), href.data(), n * sizeof(float)) == 0;
    float ms = time_kernel_ms(v.launch, warmup, iters);
    double gbs = 2.0 * n * 4 / (ms * 1e-3) / 1e9;
    std::printf("%-18s %10.3f %10.1f %7.1f%% %s\n", v.name.c_str(), ms, gbs,
                100.0 * gbs / copy_bw, ok ? "PASS" : "FAIL");
    char row[256];
    std::snprintf(row, sizeof(row), "transpose,%s,%d,%d,%.4f,%.1f,%.1f,%s", v.name.c_str(), M,
                  N, ms, gbs, 100.0 * gbs / copy_bw, ok ? "PASS" : "FAIL");
    append_csv(csv, "op,version,M,N,ms,gbs,pct_copy_bw,check", row);
  }
  CUDA_CHECK(cudaFree(din));
  CUDA_CHECK(cudaFree(dout));
  return 0;
}
