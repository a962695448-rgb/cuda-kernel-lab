// warp / block 级归约工具：用 __shfl_xor_sync 在寄存器之间交换数据，不经过共享内存。
#pragma once

#include <cfloat>

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, offset);
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
  return v;
}

// block 内求和：先 warp 内归约，每个 warp 的结果写进共享内存，再由第 0 个 warp 归约。
// 返回值对 block 内所有线程都有效。blockDim.x 必须是 32 的倍数且不超过 1024。
__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float warp_vals[32];
  __shared__ float result;
  const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
  v = warp_reduce_sum(v);
  if (lane == 0) warp_vals[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = lane < blockDim.x / 32 ? warp_vals[lane] : 0.f;
    v = warp_reduce_sum(v);
    if (lane == 0) result = v;
  }
  __syncthreads();
  return result;
}

__device__ __forceinline__ float block_reduce_max(float v) {
  __shared__ float warp_vals[32];
  __shared__ float result;
  const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
  v = warp_reduce_max(v);
  if (lane == 0) warp_vals[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = lane < blockDim.x / 32 ? warp_vals[lane] : -FLT_MAX;
    v = warp_reduce_max(v);
    if (lane == 0) result = v;
  }
  __syncthreads();
  return result;
}
