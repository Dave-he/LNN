#!/usr/bin/env python3
"""105K 档 CfC 的 CUDA 实测 (Orin Nano Super)。

对标 MTLiquid (arXiv 2609.33232) 报告的 105K 参数 / 0.40 MB 单模型多任务设定:
该文在单卡 RTX 4090 Ti 上训练, 未给边缘硬件实测。本脚本在本机 Jetson Orin Nano
Super 上补上这块证据 —— CUDA 与 CPU 双路径, 报参数量 / 峰值显存 / 延迟 / 吞吐。

数据全部为合成 (自写生成器), 不接任何真实病人数据或硬件, 符合本仓的合成仿真约束。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.environ.get("LNN_ROOT", "/workspace/LNN"))

import numpy as np
import torch
import torch.nn as nn

from lnn.core.bench_suite import CfCNetwork

N_FEATURES = 8
HIDDEN = 160          # -> 107,364 params, 对齐 MTLiquid 的 105K 档 (+2.25%)
SEQ_LEN = 48
N_TASKS = 2


def make_synthetic_multitask(n_samples: int, seed: int = 0):
    """合成两个相关联的分类任务: 共享潜在因子 + 各自独立噪声标签。

    数据形态贴近生理监测 (多变量 + 标签不平衡), 但完全由种子决定, 无外部数据依赖。
    """
    rng = np.random.default_rng(seed)
    # 潜在因子 f[t] 是被两个任务共享的慢变过程 —— 这正是多任务能省参数的来源
    latent = rng.normal(size=(n_samples, 1, 1))
    f = np.zeros((n_samples, SEQ_LEN, 1), dtype=np.float32)
    for t in range(1, SEQ_LEN):
        f[:, t] = 0.92 * f[:, t - 1] + 0.26 * rng.normal(size=(n_samples, 1))
    x = np.zeros((n_samples, SEQ_LEN, N_FEATURES), dtype=np.float32)
    x[:, :, 0] = f[:, :, 0]
    x[:, :, 1:4] = f + 0.35 * rng.normal(size=(n_samples, SEQ_LEN, 3))
    x[:, :, 4:] = rng.normal(size=(n_samples, SEQ_LEN, N_FEATURES - 4))

    # 任务 1 依赖潜在因子的均值; 任务 2 依赖末段斜率 (相关但不等价)
    # 注意 f[:, -1] 是 (n,1), 必须显式取 [.., 0] 压成 (n,), 否则和 mean_f 做
    # 布尔运算时会广播成 (n,n)
    mean_f = f[:, :, 0].mean(axis=1)
    slope_f = f[:, -1, 0] - f[:, 0, 0]
    y1 = (mean_f > np.quantile(mean_f, 0.75)).astype(np.float32)
    y2 = ((slope_f > np.quantile(slope_f, 0.75)) & (mean_f > np.quantile(mean_f, 0.4))).astype(np.float32)
    assert y1.shape == (n_samples,) and y2.shape == (n_samples,), (y1.shape, y2.shape)
    return x, y1, y2


class MTLiquidStyle(nn.Module):
    """共享 CfC backbone + per-task 输出头, 带可学习任务不确定性 (Kendall 等)。"""

    def __init__(self, hidden: int = HIDDEN, n_tasks: int = N_TASKS):
        super().__init__()
        self.backbone = CfCNetwork(input_size=N_FEATURES, hidden_size=hidden, output_size=hidden)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_tasks)])
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, x):
        h = self.backbone(x)
        if isinstance(h, (list, tuple)):
            h = h[-1]
        if h.dim() == 3:
            h = h[:, -1]
        return [head(h).squeeze(-1) for head in self.heads]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_inference(model, device, batch_sizes, seq_len, repeats=50):
    model.eval()
    out = {}
    for bs in batch_sizes:
        x = torch.randn(bs, seq_len, N_FEATURES, device=device)
        with torch.no_grad():
            for _ in range(10):
                model(x)
        sync()
        lat = []
        with torch.no_grad():
            for _ in range(repeats):
                t0 = time.perf_counter()
                model(x)
                sync()
                lat.append((time.perf_counter() - t0) * 1000)
        out[bs] = {
            "latency_ms_mean": round(statistics.mean(lat), 4),
            "latency_ms_p50": round(statistics.median(lat), 4),
            "throughput_samples_per_s": round(bs / (statistics.mean(lat) / 1000), 1),
            "per_sample_ms": round(statistics.mean(lat) / bs, 4),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--n-test", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/workspace/LNN/analysis/jetson/mtliquid105k_cuda_probe.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_ok else "cpu")

    report = {
        "status": "ok_cuda" if cuda_ok else "ok_cpu",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "torch": torch.__version__,
            "cuda_available": cuda_ok,
            "cuda_runtime": torch.version.cuda if cuda_ok else None,
            "device_name": torch.cuda.get_device_name(0) if cuda_ok else "cpu",
            "python": sys.version.split()[0],
            "host": os.uname().nodename,
        },
        "model": {
            "name": "MTLiquid-style shared-CfC multitask",
            "hidden": HIDDEN,
            "n_features": N_FEATURES,
            "n_tasks": N_TASKS,
            "seq_len": SEQ_LEN,
            "data": "synthetic (seeded generator, no real patient data)",
        },
    }

    model = MTLiquidStyle().to(device)
    n_params = count_params(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = n_params * 4
    report["model"].update({
        "params": n_params,
        "params_trainable": n_train,
        "fp32_size_bytes": param_bytes,
        "fp32_size_mb": round(param_bytes / 1024 / 1024, 4),
        "reference_mtliquid_params": 105_000,
        "delta_vs_mtliquid_pct": round((n_params - 105_000) / 105_000 * 100, 2),
    })

    print(f"参数量 {n_params:,} (MTLiquid 报告 105K, 偏差 {report['model']['delta_vs_mtliquid_pct']:+.2f}%)")
    print(f"fp32 权重 {param_bytes/1024:.1f} KiB ({param_bytes/1024/1024:.4f} MB)")
    print(f"device={device} cuda={cuda_ok}")

    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        report["environment"]["gpu"] = {
            "name": props.name,
            "total_memory_mb": round(props.total_memory / 1024 / 1024, 1),
            "multi_processor_count": props.multi_processor_count,
            "compute_capability": f"{props.major}.{props.minor}",
        }

    # ---------------- 训练 ----------------
    xtr, y1, y2 = make_synthetic_multitask(args.n_train, seed=args.seed)
    xte, z1, z2 = make_synthetic_multitask(args.n_test, seed=args.seed + 999)
    xt = torch.from_numpy(xtr)
    yt = [torch.from_numpy(y1), torch.from_numpy(y2)]

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    if cuda_ok:
        torch.cuda.reset_peak_memory_stats()
        params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    epoch_times = []
    model.train()
    n_batches = xt.shape[0] // args.batch_size
    for ep in range(args.epochs):
        perm = torch.randperm(xt.shape[0])
        t0 = time.perf_counter()
        for b in range(n_batches):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            xb = xt[idx].to(device, non_blocking=True)
            # capped-cycling: 每 batch 同时喂两个任务 (MTLiquid 的做法)
            yb = [y[idx].to(device, non_blocking=True) for y in yt]
            outs = model(xb)
            loss = 0.0
            for k, (o, yk) in enumerate(zip(outs, yb)):
                ce = bce(o, yk).mean()
                prec = torch.exp(-model.log_vars[k])
                loss = loss + (prec * ce + model.log_vars[k])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sync()
        epoch_times.append(time.perf_counter() - t0)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:>2}/{args.epochs}  loss={loss.item():.4f}  {epoch_times[-1]:.3f}s")

    train_stats = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_train": args.n_train,
        "epoch_time_s_mean": round(statistics.mean(epoch_times), 4),
        "epoch_time_s_median": round(statistics.median(epoch_times), 4),
        "total_train_s": round(sum(epoch_times), 3),
    }
    if cuda_ok:
        peak = torch.cuda.max_memory_allocated()
        train_stats["peak_vram_allocated_mb"] = round(peak / 1024 / 1024, 3)
        train_stats["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1024 / 1024, 3)
        train_stats["vram_params_mb"] = round(params_bytes / 1024 / 1024, 3)
        train_stats["vram_overhead_mb"] = round(train_stats["peak_vram_reserved_mb"] - params_bytes / 1024 / 1024, 3)
    report["train"] = train_stats
    print(f"训练 {args.epochs} epoch 用时 {train_stats['total_train_s']:.2f}s")
    if cuda_ok:
        print(f"峰值显存 reserved={train_stats['peak_vram_reserved_mb']:.1f} MB "
              f"(权重仅 {train_stats['vram_params_mb']:.3f} MB, 开销 {train_stats['vram_overhead_mb']:.1f} MB)")

    # ---------------- 推理 ----------------
    xte_t = torch.from_numpy(xte).to(device)
    yte = [torch.from_numpy(z1).numpy(), torch.from_numpy(z2).numpy()]
    model.eval()
    with torch.no_grad():
        outs = model(xte_t)
        preds = [(torch.sigmoid(o) > 0.5).float().cpu().numpy() for o in outs]
    metrics = {}
    for k, (p, y) in enumerate(zip(preds, yte)):
        acc = float((p == y).mean())
        pos = y.sum()
        tp = float((p * y).sum())
        tpr = float(tp / pos) if pos else 0.0
        fp = float((p * (1 - y)).sum())
        tnr = float(((1 - p) * (1 - y)).sum() / (1 - y).sum()) if (1 - y).sum() else 0.0
        metrics[f"task{k}"] = {"acc": round(acc, 4), "recall": round(tpr, 4),
                              "specificity": round(tnr, 4), "pos_rate_true": round(float(pos / len(y)), 4)}
    report["eval"] = metrics
    for k, v in metrics.items():
        print(f"  {k}: acc={v['acc']:.4f} recall={v['recall']:.4f} spec={v['specificity']:.4f} (正例率 {v['pos_rate_true']:.2f})")

    report["inference"] = {"on": str(device)}
    report["inference"]["gpu"] = bench_inference(model, device, [1, 16, 64, 128], SEQ_LEN) if cuda_ok else None
    report["inference"]["cpu"] = bench_inference(model, torch.device("cpu"), [1, 16, 64, 128], SEQ_LEN)

    if cuda_ok:
        g = report["inference"]["gpu"]
        c = report["inference"]["cpu"]
        report["inference"]["speedup_gpu_over_cpu"] = {
            str(bs): round(c[bs]["per_sample_ms"] / g[bs]["per_sample_ms"], 2) for bs in g
        }
        print("\n=== 单样本延迟 (batch=1) ===")
        print(f"  GPU {g[1]['per_sample_ms']:.4f} ms   CPU {c[1]['per_sample_ms']:.4f} ms   "
              f"加速 {report['inference']['speedup_gpu_over_cpu']['1']}x")
        print("=== 吞吐 (batch=128) ===")
        print(f"  GPU {g[128]['throughput_samples_per_s']:.1f} 样本/秒   "
              f"CPU {c[128]['throughput_samples_per_s']:.1f} 样本/秒")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
