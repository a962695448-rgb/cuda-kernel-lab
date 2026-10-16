# 02 HGEMM 与 Tensor Core（WMMA）

代码：`src/hgemm_wmma.cu`。输入 FP16，用 FP32 累加。

## 版本对比

| 版本 | 做法 | 用 Tensor Core 吗 |
|---|---|---|
| v0_cuda_core | 共享内存分块，FP16 读入后转成 FP32 在普通 CUDA 核心上算 | 否 |
| v1_wmma | 一个 block 4 个 warp，每个 warp 算 32×32（2×2 个 16×16 片段），调用 `wmma::mma_sync` | 是 |
| cublas_tensorcore | `cublasGemmEx`，FP16 输入，`CUBLAS_COMPUTE_32F` | 是 |

## 核心概念

- **Tensor Core**：专门做小矩阵乘加 D = A·B + C 的硬件单元。一条指令完成一个 warp 级的矩阵乘，吞吐比 CUDA 核心高一个数量级。
- **WMMA API**：`nvcuda::wmma` 是 CUDA C++ 对 Tensor Core 的封装。
  - `fragment`：一个 warp 共同持有的矩阵片段；每个线程只拿其中一部分，具体分布对程序员不透明。
  - `load_matrix_sync` / `mma_sync` / `store_matrix_sync`：整个 warp 一起调用（所以带 `_sync`）。
  - 约束：片段 16×16×16；指针要 256 位（32 字节）对齐；FP16 的 leading dimension 要是 8 的倍数。代码里 `__align__(32)` 和 BK=32 就是为了满足这些约束。
- **为什么 FP16 输入、FP32 累加**：FP16 只有 10 位尾数，几千次累加后误差会很大；用 FP32 累加器可以保住精度。大模型训练和推理的 GEMM 大多这样做。

## 面试高频问题

**Q1：WMMA 和 PTX 的 `mma.sync`、CUTLASS 有什么区别？**
- WMMA 最好上手，但片段在线程间的分布不透明，做不了精细的寄存器布局和 swizzle 优化。
- PTX 的 `mma.sync` / `ldmatrix` 能精确控制每个线程拿哪些元素，CUTLASS 和 FlashAttention 都基于这一层。
- Hopper（H100）上还有 `wgmma`（warpgroup 级）和 TMA（张量内存加速器）。

**Q2：你的 WMMA 版本为什么比 cuBLAS 慢？**（实测后用数据回答）
- 没有双缓冲或 `cp.async` 流水线，加载和计算是串行的。
- 共享内存没做 swizzle，`load_matrix_sync` 读取时可能有 bank conflict。
- block 只有 64×64，每读一次数据能做的计算量偏少。

**Q3：之前的 Hadamard 项目里，WMMA 为什么没比 warp 实现快？**
- 可能的原因（回答前先对照 Hadamard 项目的性能报告确认）：维度最大只有 256，每个元素的计算量很小，整体偏访存受限，Tensor Core 的算力优势发挥不出来。
- 另外，WMMA 要求固定的 16×16 片段，数据要先整理成矩阵形式，这部分额外开销可能抵消收益。
- 这个例子能说明"先判断瓶颈，再选优化手段"，可以用来把两个项目串起来讲。

## 实测结果（RTX 5090，CUDA 12.8，2026-10-16 实测；数据来自 `results/`）

M=N=K=4096：

| 版本 | TFLOPS | 占 cuBLAS |
|---|---|---|
| cuBLAS（Tensor Core） | 180.0 | 100% |
| v0 CUDA 核心 | 9.5 | 5.3% |
| v1 WMMA | **115.1** | **63.9%** |

- 同样是 FP16 输入，改用 Tensor Core 后提速 **12.1 倍**（9.5 → 115.1 TFLOPS）。
- WMMA 版本离 cuBLAS 还差 36%。主要原因见 Q2：没有流水线，共享内存没做 swizzle，block 只有 64×64。
- 1024³ 时只达到 cuBLAS 的 53.6%，同样受 block 数量不足的影响：1024²/64² = 256 个 block，对 170 个 SM 只有约 1.5 波。

## 练习

1. 把 BK 从 32 改成 64，看共享内存用量和性能的变化。
2. 用 ncu 看 v1 的 `sm__pipe_tensor_op_hmma_cycles_active`，确认 Tensor Core 真的在工作。
