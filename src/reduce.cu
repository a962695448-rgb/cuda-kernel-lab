// 求和归约：把 N 个 float 加成一个数。典型的访存受限算子，用带宽（GB/s）衡量。
// v0 交错寻址（warp 分歧）→ v1 顺序寻址（无分歧、无 bank conflict）
// → v2 网格跨步 + float4 + warp shuffle，对照 CUB 的 DeviceReduce::Sum。
#include <cub/cub.cuh>

#include "common.cuh"
#include "warp_utils.cuh"

// v0：stride 从 1 开始翻倍，只有 tid 是 2*stride 倍数的线程干活。
// 同一个 warp 里一半线程闲着（分歧），共享内存访问也随 stride 变大产生 bank conflict。
__global__ void reduce_v0_interleaved(const float* in, float* out, int n) {
  extern __shared__ float s[];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;
  s[tid] = i < n ? in[i] : 0.f;
  __syncthreads();
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, s[0]);
}

// v1：stride 从 blockDim/2 开始减半，前 stride 个连续线程干活：
// warp 要么全干要么全不干，相邻线程访问相邻地址，没有 bank conflict。
__global__ void reduce_v1_sequential(const float* in, float* out, int n) {
  extern __shared__ float s[];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;
  s[tid] = i < n ? in[i] : 0.f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, s[0]);
}

// v2：只启动“刚好填满 GPU”的 block 数，每个线程用网格跨步循环累加很多个 float4，
// 最后 warp shuffle + 少量共享内存完成 block 内归约，每个 block 只做一次 atomicAdd。
__global__ void reduce_v2_shuffle(const float* in, float* out, int n) {
  float sum = 0.f;
  const int n4 = n / 4;
  const float4* in4 = reinterpret_cast<const float4*>(in);
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    float4 v = in4[i];
    sum += (v.x + v.y) + (v.z + v.w);
  }
  for (int i = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    sum += in[i];
  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) atomicAdd(out, sum);
}

int main(int argc, char** argv) {
  print_device_info();
  const int n = arg_int(argc, argv, "--n", 1 << 26);
  const int warmup = arg_int(argc, argv, "--warmup", 5);
  const int iters = arg_int(argc, argv, "--iters", 20);
  const std::string csv = arg_str(argc, argv, "--csv", "results/reduce.csv");
  const double copy_bw = measure_copy_bandwidth_gbs();
  std::printf("# device-to-device copy bandwidth: %.1f GB/s\n", copy_bw);

  // 取 [0,1) 的数，和约为 n/2，便于用相对误差判断正确性。
  std::vector<float> h(n);
  fill_random(h, 7, 0.f, 1.f);
  double ref = 0.0;
  for (float x : h) ref += x;
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, size_t(n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), size_t(n) * sizeof(float), cudaMemcpyHostToDevice));

  int num_sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  void* cub_tmp = nullptr;
  size_t cub_tmp_bytes = 0;
  CUDA_CHECK(cub::DeviceReduce::Sum(cub_tmp, cub_tmp_bytes, d_in, d_out, n));
  CUDA_CHECK(cudaMalloc(&cub_tmp, cub_tmp_bytes));

  const int threads = 256;
  struct Variant {
    std::string name;
    std::function<void()> launch;
  };
  std::vector<Variant> variants = {
      {"cub",
       [&] { CUDA_CHECK(cub::DeviceReduce::Sum(cub_tmp, cub_tmp_bytes, d_in, d_out, n)); }},
      {"v0_interleaved",
       [&] {
         CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));
         reduce_v0_interleaved<<<ceil_div(n, threads), threads, threads * sizeof(float)>>>(
             d_in, d_out, n);
       }},
      {"v1_sequential",
       [&] {
         CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));
         reduce_v1_sequential<<<ceil_div(n, threads), threads, threads * sizeof(float)>>>(
             d_in, d_out, n);
       }},
      {"v2_shuffle",
       [&] {
         CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));
         reduce_v2_shuffle<<<num_sms * 8, threads>>>(d_in, d_out, n);
       }},
  };

  std::printf("\n== Reduce sum, n=%d (%.0f MB) ==\n%-16s %10s %10s %8s %12s %s\n", n,
              n * 4.0 / (1 << 20), "version", "ms", "GB/s", "%copy", "rel_err", "check");
  for (auto& v : variants) {
    if (v.name != "cub" && !selected(argc, argv, v.name)) continue;
    v.launch();
    CHECK_LAST_KERNEL();
    float got = 0.f;
    CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    double rel = std::fabs(got - ref) / ref;
    bool ok = rel < 1e-4;
    float ms = time_kernel_ms(v.launch, warmup, iters);
    double gbs = n * 4.0 / (ms * 1e-3) / 1e9;
    std::printf("%-16s %10.3f %10.1f %7.1f%% %12.2e %s\n", v.name.c_str(), ms, gbs,
                100.0 * gbs / copy_bw, rel, ok ? "PASS" : "FAIL");
    char row[256];
    std::snprintf(row, sizeof(row), "reduce,%s,%d,%.4f,%.1f,%.1f,%.3e,%s", v.name.c_str(), n,
                  ms, gbs, 100.0 * gbs / copy_bw, rel, ok ? "PASS" : "FAIL");
    append_csv(csv, "op,version,n,ms,gbs,pct_copy_bw,rel_err,check", row);
  }
  CUDA_CHECK(cudaFree(cub_tmp));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
