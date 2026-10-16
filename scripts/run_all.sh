#!/usr/bin/env bash
# 在 GPU 服务器上：编译 → 跑全部基准 → 生成 results/SUMMARY.md。
# 用法：bash scripts/run_all.sh [CUDA 架构，例如 89、120；不填则自动识别当前显卡]
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH="${1:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '. ')}"
echo "CUDA architecture: sm_${ARCH}"

nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv | tee results/gpu_info.csv
nvcc --version | tail -n 2 | tee results/nvcc_version.txt

cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2>&1 | tee results/build.log

rm -f results/*.csv.tmp
for f in sgemm hgemm reduce softmax rmsnorm transpose; do rm -f "results/$f.csv"; done

./build/sgemm --sizes 1024,2048,4096 | tee results/sgemm.txt
./build/hgemm_wmma --sizes 1024,2048,4096 | tee results/hgemm.txt
./build/reduce | tee results/reduce.txt
./build/softmax | tee results/softmax.txt
./build/rmsnorm | tee results/rmsnorm.txt
./build/transpose | tee results/transpose.txt

python3 scripts/summarize.py > results/SUMMARY.md
echo "done: results/SUMMARY.md"
