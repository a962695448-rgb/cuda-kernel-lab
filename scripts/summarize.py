"""把 results/*.csv 汇总成 Markdown 表格（只用标准库）。"""
import csv
import pathlib

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

SECTIONS = [
    ("sgemm.csv", "SGEMM（FP32）", ["version", "M", "ms", "gflops", "pct_cublas", "check"]),
    ("hgemm.csv", "HGEMM（FP16 输入，FP32 累加）", ["version", "M", "ms", "tflops", "pct_cublas", "check"]),
    ("reduce.csv", "Reduce 求和", ["version", "n", "ms", "gbs", "pct_copy_bw", "check"]),
    ("softmax.csv", "Softmax", ["version", "rows", "cols", "ms", "gbs", "pct_copy_bw", "check"]),
    ("rmsnorm.csv", "RMSNorm", ["version", "rows", "hidden", "ms", "gbs", "pct_copy_bw", "check"]),
    ("transpose.csv", "矩阵转置", ["version", "M", "N", "ms", "gbs", "pct_copy_bw", "check"]),
]


def main() -> None:
    gpu = RESULTS / "gpu_info.csv"
    if gpu.exists():
        print("GPU：" + gpu.read_text(encoding="utf-8").strip().splitlines()[-1] + "\n")
    for fname, title, cols in SECTIONS:
        path = RESULTS / fname
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        print(f"## {title}\n")
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for r in rows:
            print("| " + " | ".join(r.get(c, "") for c in cols) + " |")
        print()


if __name__ == "__main__":
    main()
