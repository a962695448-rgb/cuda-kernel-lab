#!/usr/bin/env bash
# 用 Nsight Compute 抓关键指标，回答“为什么快/为什么慢”。
# 每个版本单独跑一次（--only），只抓 1 次 kernel 启动，导出 CSV 便于写报告。
# 需要先跑过 scripts/run_all.sh 完成编译。
# 注意：很多云 GPU 虚拟机在宿主机层面关闭了性能计数器，会报 ERR_NVGPUCTRPERM，在虚拟机内部无法解决。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/ncu

# 计算吞吐、显存吞吐、占用率、共享内存 bank conflict、每个 warp 的主要停顿原因
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
launch__registers_per_thread,\
smsp__average_warp_latency_issue_stalled_long_scoreboard,\
smsp__average_warp_latency_issue_stalled_barrier"

profile() {  # $1=可执行文件 $2=--only 版本名 $3=kernel 名正则 $4...=额外参数
  local bin=$1 ver=$2 kre=$3; shift 3
  ncu --metrics "$METRICS" --launch-count 1 -k "regex:${kre}" --csv \
      "./build/$bin" --only "$ver" --warmup 0 --iters 1 --csv /tmp/ncu_ignore.csv "$@" \
      > "results/ncu/${bin}_${ver}.csv" || echo "ncu failed for $bin $ver"
}

for v in v0_naive v1_smem v2_1d_tile v3_2d_tile v4_vectorize v5_double_buffer; do
  profile sgemm "$v" "sgemm_${v}" --sizes 4096
done
profile hgemm_wmma v1_wmma "hgemm_v1_wmma" --sizes 4096
for v in v0_interleaved v1_sequential v2_shuffle; do profile reduce "$v" "reduce_${v}"; done
for v in v1_block_row v2_online v3_warp_register; do profile softmax "$v" "softmax_${v}"; done
# 两个版本的函数名都是 transpose_smem（模板参数不参与匹配），靠 --only 区分
profile transpose v1_smem_conflict "transpose_smem"
profile transpose v2_smem_padded "transpose_smem"

# 另存一份完整报告（含 roofline 图），可以下载到本地用 Nsight Compute 图形界面打开
ncu --set full --launch-count 1 -k regex:sgemm_v5 -o results/ncu/sgemm_v5_full \
    ./build/sgemm --only v5_double_buffer --sizes 4096 --warmup 0 --iters 1 --csv /tmp/ncu_ignore.csv || true
echo "done: results/ncu/"
