// 公共工具：错误检查、计时、随机初始化、结果比对、CSV 输出。
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,   \
                   __LINE__, cudaGetErrorString(err_));                       \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define CHECK_LAST_KERNEL() CUDA_CHECK(cudaGetLastError())

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// 用 cudaEvent 计时：先预热，再取多次运行的中位数（毫秒），减少偶发抖动的影响。
inline float time_kernel_ms(const std::function<void()>& launch, int warmup = 5,
                            int iters = 20) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> samples(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&samples[i], start, stop));
  }
  CHECK_LAST_KERNEL();
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::sort(samples.begin(), samples.end());
  return samples[iters / 2];
}

inline void fill_random(std::vector<float>& v, unsigned seed, float lo = -1.f,
                        float hi = 1.f) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& x : v) x = dist(gen);
}

struct CompareResult {
  double max_abs_err = 0.0;
  double max_rel_err = 0.0;
  bool ok = true;
};

// 逐元素比较：|a-b| <= atol + rtol*|b| 视为通过。
inline CompareResult compare(const float* got, const float* ref, size_t n,
                             double atol, double rtol) {
  CompareResult r;
  for (size_t i = 0; i < n; ++i) {
    double diff = std::fabs(double(got[i]) - double(ref[i]));
    double rel = diff / (std::fabs(double(ref[i])) + 1e-12);
    r.max_abs_err = std::max(r.max_abs_err, diff);
    r.max_rel_err = std::max(r.max_rel_err, rel);
    if (!(diff <= atol + rtol * std::fabs(double(ref[i])))) r.ok = false;
  }
  return r;
}

// 用 device-to-device 拷贝测出本卡实际可达的显存带宽（GB/s），作为访存型算子的对照上限。
inline double measure_copy_bandwidth_gbs(size_t bytes = size_t(256) << 20) {
  void *a = nullptr, *b = nullptr;
  CUDA_CHECK(cudaMalloc(&a, bytes));
  CUDA_CHECK(cudaMalloc(&b, bytes));
  float ms = time_kernel_ms(
      [&] { CUDA_CHECK(cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice)); });
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  return 2.0 * bytes / (ms * 1e-3) / 1e9;  // 读 + 写
}

inline void print_device_info() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  std::printf("# device: %s, sm_%d%d, SMs=%d, smem/block=%zu KB\n", p.name,
              p.major, p.minor, p.multiProcessorCount,
              p.sharedMemPerBlockOptin / 1024);
}

// 结果追加写入 CSV，便于后续画图和写报告。
inline void append_csv(const std::string& path, const std::string& header,
                       const std::string& row) {
  FILE* f = std::fopen(path.c_str(), "r");
  bool exists = f != nullptr;
  if (f) std::fclose(f);
  f = std::fopen(path.c_str(), "a");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return;
  }
  if (!exists) std::fprintf(f, "%s\n", header.c_str());
  std::fprintf(f, "%s\n", row.c_str());
  std::fclose(f);
}

inline int arg_int(int argc, char** argv, const char* name, int def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
  return def;
}

inline bool has_flag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], name) == 0) return true;
  return false;
}

inline std::vector<int> parse_int_list(const std::string& s) {
  std::vector<int> out;
  size_t start = 0;
  while (start < s.size()) {
    size_t end = s.find(',', start);
    if (end == std::string::npos) end = s.size();
    out.push_back(std::atoi(s.substr(start, end - start).c_str()));
    start = end + 1;
  }
  return out;
}

// 只跑名字匹配 --only 的版本（给 Nsight Compute 单独抓某个 kernel 用）。
inline bool selected(int argc, char** argv, const std::string& name) {
  std::string only = "";
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], "--only") == 0) only = argv[i + 1];
  return only.empty() || name.find(only) != std::string::npos;
}

inline std::string arg_str(int argc, char** argv, const char* name,
                           const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
  return def;
}
