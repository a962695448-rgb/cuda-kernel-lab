GPU：NVIDIA GeForce RTX 5090, 610.43.02, 32607 MiB, 3090 MHz

## SGEMM（FP32）

| version | M | ms | gflops | pct_cublas | check |
|---|---|---|---|---|---|
| cublas | 1024 | 0.0492 | 43690.7 | 100.0 | PASS |
| v0_naive | 1024 | 0.3796 | 5657.9 | 13.0 | PASS |
| v1_smem | 1024 | 0.2550 | 8422.3 | 19.3 | PASS |
| v2_1d_tile | 1024 | 0.1041 | 20623.5 | 47.2 | PASS |
| v3_2d_tile | 1024 | 0.1238 | 17340.8 | 39.7 | PASS |
| v4_vectorize | 1024 | 0.1198 | 17924.4 | 41.0 | PASS |
| v5_double_buffer | 1024 | 0.0850 | 25266.9 | 57.8 | PASS |
| cublas | 2048 | 0.2508 | 68513.4 | 100.0 | PASS |
| v0_naive | 2048 | 2.6878 | 6391.7 | 9.3 | PASS |
| v1_smem | 2048 | 1.7875 | 9611.2 | 14.0 | PASS |
| v2_1d_tile | 2048 | 0.6561 | 26185.0 | 38.2 | PASS |
| v3_2d_tile | 2048 | 0.4001 | 42935.9 | 62.7 | PASS |
| v4_vectorize | 2048 | 0.3471 | 49499.4 | 72.2 | PASS |
| v5_double_buffer | 2048 | 0.3202 | 53660.3 | 78.3 | PASS |
| cublas | 4096 | 1.9841 | 69271.4 | 100.0 | PASS |
| v0_naive | 4096 | 21.3023 | 6451.8 | 9.3 | PASS |
| v1_smem | 4096 | 14.7070 | 9345.2 | 13.5 | PASS |
| v2_1d_tile | 4096 | 4.8043 | 28607.6 | 41.3 | PASS |
| v3_2d_tile | 4096 | 2.8411 | 48376.0 | 69.8 | PASS |
| v4_vectorize | 4096 | 2.7268 | 50402.7 | 72.8 | PASS |
| v5_double_buffer | 4096 | 2.4528 | 56034.2 | 80.9 | PASS |

## HGEMM（FP16 输入，FP32 累加）

| version | M | ms | tflops | pct_cublas | check |
|---|---|---|---|---|---|
| cublas_tensorcore | 1024 | 0.0248 | 86.70 | 100.0 | PASS |
| v0_cuda_core | 1024 | 0.2488 | 8.63 | 10.0 | PASS |
| v1_wmma | 1024 | 0.0462 | 46.44 | 53.6 | PASS |
| cublas_tensorcore | 2048 | 0.1079 | 159.21 | 100.0 | PASS |
| v0_cuda_core | 2048 | 1.7420 | 9.86 | 6.2 | PASS |
| v1_wmma | 2048 | 0.1751 | 98.09 | 61.6 | PASS |
| cublas_tensorcore | 4096 | 0.7634 | 180.04 | 100.0 | PASS |
| v0_cuda_core | 4096 | 14.4648 | 9.50 | 5.3 | PASS |
| v1_wmma | 4096 | 1.1946 | 115.05 | 63.9 | PASS |

## Reduce 求和

| version | n | ms | gbs | pct_copy_bw | check |
|---|---|---|---|---|---|
| cub | 67108864 | 0.1645 | 1631.4 | 107.2 | PASS |
| v0_interleaved | 67108864 | 0.3321 | 808.2 | 53.1 | PASS |
| v1_sequential | 67108864 | 0.3313 | 810.2 | 53.3 | PASS |
| v2_shuffle | 67108864 | 0.1638 | 1639.0 | 107.7 | PASS |

## Softmax

| version | rows | cols | ms | gbs | pct_copy_bw | check |
|---|---|---|---|---|---|---|
| v0_thread_row | 65536 | 1024 | 1.9919 | 269.5 | 17.7 | PASS |
| v1_block_row | 65536 | 1024 | 0.3510 | 1529.7 | 100.5 | PASS |
| v2_online | 65536 | 1024 | 0.3526 | 1522.7 | 100.0 | PASS |
| v3_warp_register | 65536 | 1024 | 0.3591 | 1495.2 | 98.2 | PASS |

## RMSNorm

| version | rows | hidden | ms | gbs | pct_copy_bw | check |
|---|---|---|---|---|---|---|
| v0_thread_row | 8192 | 4096 | 1.0358 | 259.2 | 17.0 | PASS |
| v1_block_smem | 8192 | 4096 | 0.1751 | 1533.3 | 100.7 | PASS |
| v2_vec4_shuffle | 8192 | 4096 | 0.1748 | 1535.5 | 100.9 | PASS |

## 矩阵转置

| version | M | N | ms | gbs | pct_copy_bw | check |
|---|---|---|---|---|---|---|
| v0_naive | 8192 | 8192 | 0.6662 | 805.9 | 53.0 | PASS |
| v1_smem_conflict | 8192 | 8192 | 0.3734 | 1437.9 | 94.5 | PASS |
| v2_smem_padded | 8192 | 8192 | 0.3673 | 1461.6 | 96.1 | PASS |

## 补充实验：转置 2048×2048（16 MB，可放进 L2）

| version | M | N | ms | gbs | check |
|---|---|---|---|---|---|
| v0_naive | 2048 | 2048 | 0.0440 | 763.2 | PASS |
| v1_smem_conflict | 2048 | 2048 | 0.0144 | 2325.0 | PASS |
| v2_smem_padded | 2048 | 2048 | 0.0083 | 4033.0 | PASS |

## 补充实验：Reduce 2²² 个 float（16 MB，可放进 L2）

| version | n | ms | gbs | check |
|---|---|---|---|---|
| cub | 4194304 | 0.0074 | 2269.6 | PASS |
| v0_interleaved | 4194304 | 0.0265 | 633.2 | PASS |
| v1_sequential | 4194304 | 0.0263 | 637.0 | PASS |
| v2_shuffle | 4194304 | 0.0081 | 2080.5 | PASS |

