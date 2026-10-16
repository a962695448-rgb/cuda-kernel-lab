# cuda-kernel-lab

手写常见深度学习算子的 CUDA 实现，每个算子从朴素版本逐步优化，并且都附带：
- 正确性校验：和 cuBLAS、CUB 或 CPU 双精度参考逐元素比对；
- 性能基准：GEMM 报告 GFLOPS 和占 cuBLAS 的百分比，访存型算子报告 GB/s 和占实测拷贝带宽的百分比；
- Nsight Compute 指标：用来解释每一步为什么变快。

| 算子 | 版本 | 涉及的优化 |
|---|---|---|
| SGEMM（FP32） | v0–v5 + cuBLAS | 共享内存分块、一维/二维线程分块（8×8）、float4 向量化、A 转置存放、双缓冲 |
| HGEMM（FP16 输入，FP32 累加） | CUDA core / WMMA + cuBLAS | Tensor Core（WMMA）、warp 级分块 |
| Reduce | v0–v2 + CUB | 消除 warp 分歧、顺序寻址、网格跨步循环、warp shuffle |
| Softmax | v0–v3 | 合并访存、online softmax（FlashAttention 的基础）、寄存器缓存 |
| RMSNorm | v0–v2 | block 归约、float4、warp shuffle |
| 转置 | v0–v2 | 合并访存、共享内存 padding 消除 bank conflict |

## 运行

需要 CUDA 12.x、CMake ≥ 3.20 和 NVIDIA GPU。默认按 RTX 4090（sm_89）编译。

```bash
bash scripts/run_all.sh          # 编译 + 全部基准 → results/SUMMARY.md
bash scripts/ncu_profile.sh      # Nsight Compute 指标 → results/ncu/
./build/sgemm --sizes 4096 --only v5_double_buffer   # 单独跑某个版本
```

其他 GPU 用 `bash scripts/run_all.sh 80`（A100）这样的方式指定架构。

## 结果

在 RTX 4090 上实测后填写，见 `results/SUMMARY.md`。

## 学习笔记

`notes/` 目录下每个算子一篇：版本对比、必须会算的数、面试常见问题和练习。从 [00-导读](notes/00-导读.md) 开始读。

## 参考

- Simon Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*（SGEMM v1–v4 的思路来源）
- Mark Harris, *Optimizing Parallel Reduction in CUDA*（NVIDIA）
- Milakov & Gimelshein, *Online normalizer calculation for softmax*（2018）
- NVIDIA, *CUDA C++ Programming Guide* —— WMMA 一节
