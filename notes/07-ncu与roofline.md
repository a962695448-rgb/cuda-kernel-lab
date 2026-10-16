# 07 用 Nsight Compute 和 roofline 回答“为什么快”

> **实测环境说明**：2026-10-16 租用的 RTX 5090 云服务器在宿主机层面关闭了 GPU 性能计数器，ncu 报 `ERR_NVGPUCTRPERM`，所以本仓库暂时没有 ncu 指标。替代做法是控制变量实验，比如把数据缩小到能放进 L2 来暴露 bank conflict（见 06）。换一台允许 profiling 的机器后，运行 `scripts/ncu_profile.sh` 即可补齐。

面试里说"我优化了 3 倍"不够，要能回答："瓶颈在哪？你怎么知道的？"本篇用到的工具：`scripts/ncu_profile.sh`。

## 两个工具

| 工具 | 回答什么 | 命令 |
|---|---|---|
| Nsight Systems（nsys） | 时间线：哪个 kernel 慢、CPU 和 GPU 有没有空等、拷贝和计算有没有重叠 | `nsys profile -o out ./build/sgemm` |
| Nsight Compute（ncu） | 单个 kernel 内部：算力和带宽各用了多少、warp 卡在哪里 | `ncu --set full -k regex:sgemm_v5 ./build/sgemm --only v5` |

先用 nsys 找到最慢的 kernel，再用 ncu 深挖它。

## 脚本抓的指标怎么读

| 指标 | 含义 | 怎么用 |
|---|---|---|
| `sm__throughput…pct_of_peak` | 计算单元利用率 | GEMM 越接近 100% 越好 |
| `dram__throughput…pct_of_peak` | 显存带宽利用率 | 访存型算子越接近 100% 越好 |
| `sm__warps_active…`（achieved occupancy） | 实际驻留的 warp 比例 | 太低会导致延迟藏不住；但 occupancy 只是手段，不是目标 |
| `l1tex__data_bank_conflicts…` | 共享内存 bank conflict 次数 | 转置 v1 和 v2 对比；SGEMM v4 的转置写入 |
| `launch__registers_per_thread` | 每个线程的寄存器数 | 解释 occupancy 的上限 |
| `…stalled_long_scoreboard` | warp 在等显存数据 | 双缓冲应该让它下降 |
| `…stalled_barrier` | warp 卡在 `__syncthreads` | 同步太多或负载不均 |

## roofline 怎么讲

- 横轴是计算访存比（FLOP/B），纵轴是达到的 FLOP/s。
- 屋顶由两条线组成：斜线是"带宽 × AI"，平线是峰值算力，两条线在拐点（5090 约 58 FLOP/B）相交。
- 每个 kernel 是图上的一个点：在斜线下方说明访存受限，在平线下方说明计算受限；离屋顶越近，效率越高。
- `ncu --set full` 生成的报告里有现成的 roofline 图（GPU Speed Of Light 页），下载 `.ncu-rep` 文件后，用本地的 Nsight Compute 图形界面打开就能看到。

## 面试回答模板

> 这个 kernel 的计算访存比约为 X，低于拐点，所以是访存受限。ncu 显示 DRAM 吞吐达到峰值的 Y%，而 SM 吞吐只有 Z%，说明瓶颈在带宽。我通过 A（比如 float4 向量化、减少读取遍数）把显存访问量从 B 降到了 C，带宽利用率从 D% 提到 E%。

**X、Y、Z 必须换成你自己实测的数字。**
