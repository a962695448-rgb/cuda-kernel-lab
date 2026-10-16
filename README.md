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

需要 CUDA 12.x、CMake ≥ 3.20 和 NVIDIA GPU。`run_all.sh` 会自动识别显卡架构（RTX 4090 为 sm_89，RTX 5090 为 sm_120，后者需要 CUDA ≥ 12.8）。

```bash
bash scripts/run_all.sh          # 编译 + 全部基准 → results/SUMMARY.md
bash scripts/ncu_profile.sh      # Nsight Compute 指标 → results/ncu/
./build/sgemm --sizes 4096 --only v5_double_buffer   # 单独跑某个版本
```

也可以手动指定架构，例如 `bash scripts/run_all.sh 80`（A100）。

## 结果

RTX 5090（sm_120，32 GB GDDR7，96 MB L2），CUDA 12.8，2026-10-16 实测。所有版本都通过了正确性校验；完整数据见 [`results/SUMMARY.md`](results/SUMMARY.md)。

**计算受限：GEMM（4096³）**

| 算子 | 最好的版本 | 性能 | 占 cuBLAS |
|---|---|---|---|
| SGEMM FP32 | v5 双缓冲 | 56.0 TFLOPS | **80.9%**（v0 朴素版只有 9.3%） |
| HGEMM FP16 | v1 WMMA Tensor Core | 115.1 TFLOPS | 63.9%（比 CUDA 核心版快 12.1 倍） |

**访存受限**（数据量都大于 L2；参照值是实测拷贝带宽 1521 GB/s，理论带宽 1792 GB/s）

| 算子 | 朴素版本 | 最好的版本 | 说明 |
|---|---|---|---|
| Reduce（256 MB） | 808 GB/s | **1,639 GB/s** | 与 CUB 持平（1,631 GB/s） |
| Softmax（65536×1024） | 270 GB/s | **1,530 GB/s** | 达到拷贝带宽 |
| RMSNorm（8192×4096） | 259 GB/s | **1,536 GB/s** | 达到拷贝带宽 |
| 转置（8192²） | 806 GB/s | **1,462 GB/s** | 拷贝带宽的 96% |

**几个和教科书不一样的发现**（详见各篇笔记）

- 1024³ 时，SGEMM 的 128×128 分块版本反而比 64×64 的慢：block 数（64 个）少于 SM 数（170 个），GPU 填不满。
- 大矩阵转置里，bank conflict 的代价被显存带宽掩盖（只差 1.6%）；把数据缩小到能放进 L2 后，padding 版本快了 **1.73 倍**。
- Softmax 从三遍读降到一遍读，在显存层面几乎没有收益：每行只有 4 KB，后两遍读取都命中了缓存。

> 这台云服务器关闭了 GPU 性能计数器（ncu 报 `ERR_NVGPUCTRPERM`），所以暂时没有 Nsight Compute 指标，改用控制变量实验验证结论。

## 学习笔记

`notes/` 目录下每个算子一篇：版本对比、必须会算的数、面试常见问题和练习。从 [00-导读](notes/00-导读.md) 开始读。

## 参考

- Simon Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*（SGEMM v1–v4 的思路来源）
- Mark Harris, *Optimizing Parallel Reduction in CUDA*（NVIDIA）
- Milakov & Gimelshein, *Online normalizer calculation for softmax*（2018）
- NVIDIA, *CUDA C++ Programming Guide* —— WMMA 一节
