#!/usr/bin/env python3
"""Tox21-styled synthetic molecular smoke for GraphLNNPredictor.

PRD §8 task #6 follow-up A (queued from iter#5 GCN-CfC repo survey at
docs/reports/GCN-CfC_仓库结构化调研.md).  GCN-CfC's actual MoleculeNet
pipeline needs PyTorch Geometric + RDKit + TensorFlow, none of which are
installable in a reasonable budget on this Jetson Orin Nano (aarch64
wheels are thin on the ground).

Instead, this script exercises the same GNN+LNN coupling the project's
``lnn.core.graph.GraphLNNPredictor`` provides, on a Tox21-styled
synthetic dataset:

* Each "molecule" is a small random graph with 8-20 atoms and 1-4
  randomly-typed atom features.
* The binary "toxicity" label is a deterministic function of two
  graph-structural features (degree saturation + triangle density),
  so the model has to do real graph reasoning to score above chance.
* The model receives a single-snapshot dynamic graph (``time=1``);
  ``GraphLNNPredictor`` collapses to a GNN-encoder + 1-step CfC/LTC/GRU
  predictor, perfectly matching the GCN-CfC repo's "GNN-embedding →
  CfC classifier" pattern but **end-to-end PyTorch with shared
  gradient**.

We run all three available recurrent backbones on the same data and
emit one JSON + one MD report.  Output lives under
``analysis/molecular/`` (created on first run).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.graph import GraphLNNPredictor


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _random_graph_batch(
    rng: torch.Generator,
    num_samples: int,
    max_nodes: int,
    feature_size: int,
    edge_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate ``num_samples`` molecular-style random graphs + labels.

    Labels are assigned by comparing each graph's *triangle density* against
    the dataset median (so the +/− split is automatically ~50/50, even when
    edge_prob shifts the absolute statistics).  Triangle density isn't
    recoverable from atom-feature aggregation alone — the model needs the
    graph encoder's neighbour pooling to score above chance.
    """
    node_features = torch.zeros(num_samples, 1, max_nodes, feature_size)
    adjacency = torch.zeros(num_samples, 1, max_nodes, max_nodes)
    tri_density = torch.zeros(num_samples)
    atom_counts = torch.zeros(num_samples)

    for i in range(num_samples):
        n_atoms = int(torch.randint(8, max_nodes + 1, (1,), generator=rng).item())
        atom_counts[i] = float(n_atoms)
        atom_types = torch.randint(0, feature_size, (n_atoms,), generator=rng)
        for k in range(n_atoms):
            node_features[i, 0, k, atom_types[k]] = 1.0

        upper = torch.rand(n_atoms, n_atoms, generator=rng).triu(diagonal=1)
        edges = (upper < edge_prob).float()
        edges = edges + edges.t()
        adjacency[i, 0, :n_atoms, :n_atoms] = edges

        triangles = float((edges @ edges @ edges).diag().sum().item() / 6.0)
        denom = max(n_atoms * (n_atoms - 1) * (n_atoms - 2) / 6.0, 1.0)
        tri_density[i] = triangles / denom

    median = float(tri_density.median().item())
    labels = (tri_density > median).float()
    return node_features, adjacency, labels


def _train_eval(
    backbone: str,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    torch.manual_seed(args.seed)
    model = GraphLNNPredictor(
        node_feature_size=args.feature_size,
        graph_feature_size=args.graph_feature_size,
        hidden_size=args.hidden_size,
        output_size=1,
        recurrent_type=backbone,
    ).to(device)

    # PRD §9 #4 / iter#13: optional two-stage training that emulates GCN-CfC's
    # "GNN-encoder + frozen embeddings → CfC head" pipeline within our single-
    # stack PyTorch model.  Phase 1 pre-trains the encoder + a tiny linear
    # probe; Phase 2 freezes the encoder and trains only the recurrent head.
    # The end-to-end baseline is what we get with --frozen-encoder OFF.
    if getattr(args, "frozen_encoder", False):
        encoder = model.encoder
        encoder_parameters = sum(p.numel() for p in encoder.parameters())
        # Phase 1: train encoder only via a 1-layer linear probe to the label.
        probe = torch.nn.Linear(args.graph_feature_size, 1).to(device)
        pre_opt = torch.optim.AdamW(
            list(encoder.parameters()) + list(probe.parameters()), lr=args.lr
        )
        tn, ta, tl = (t.to(device) for t in train_data)
        pretrain_epochs = max(args.frozen_pretrain_epochs, 1)
        for _ in range(pretrain_epochs):
            idx = torch.randperm(tn.shape[0], device=device)
            for chunk in idx.split(args.batch_size):
                feat = encoder(tn[chunk], ta[chunk])
                # take the only (time=1) step
                pooled = feat[:, 0, :]
                logits = probe(pooled).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, tl[chunk])
                pre_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(probe.parameters()),
                    max_norm=1.0,
                )
                pre_opt.step()
        # Phase 2: freeze encoder, optimise only the recurrent + readout.
        for p in encoder.parameters():
            p.requires_grad_(False)
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(head_params, lr=args.lr)
        parameters = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in head_params)
        print(
            f"    [frozen] encoder={encoder_parameters:,} (frozen),"
            f" head trainable={trainable:,},"
            f" pretrain_epochs={pretrain_epochs}"
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        parameters = sum(p.numel() for p in model.parameters())

    tn, ta, tl = (t.to(device) for t in train_data)
    vn, va, vl = (t.to(device) for t in val_data)

    start = time.perf_counter()
    for _ in range(args.epochs):
        # mini-batch SGD by simple chunking.
        idx = torch.randperm(tn.shape[0], device=device)
        for chunk in idx.split(args.batch_size):
            batch = {"node_features": tn[chunk], "adjacency": ta[chunk]}
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, tl[chunk])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        # inference throughput: time pure forward over the val set in chunks.
        forwards = max(args.inference_repeats, 1)
        warm = {"node_features": vn[:1], "adjacency": va[:1]}
        for _ in range(2):
            _ = model(warm)
        t0 = time.perf_counter()
        total = 0
        for _ in range(forwards):
            for chunk in torch.arange(vn.shape[0], device=device).split(args.batch_size):
                _ = model({"node_features": vn[chunk], "adjacency": va[chunk]})
                total += chunk.numel()
        t_inf = time.perf_counter() - t0

        logits = model({"node_features": vn, "adjacency": va}).squeeze(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        accuracy = float((preds == vl).float().mean().item())
        # AUC via Mann–Whitney U / Wilcoxon estimator (no sklearn needed).
        positive = probs[vl == 1.0]
        negative = probs[vl == 0.0]
        if positive.numel() > 0 and negative.numel() > 0:
            comparison = (positive.unsqueeze(1) > negative.unsqueeze(0)).float()
            ties = (positive.unsqueeze(1) == negative.unsqueeze(0)).float() * 0.5
            auc = float((comparison + ties).mean().item())
        else:
            auc = float("nan")

    return {
        "backbone": backbone,
        "parameters": parameters,
        "val_accuracy": accuracy,
        "val_auc_roc": auc,
        "train_seconds": elapsed,
        "inference_seconds": t_inf,
        "inference_samples": total,
        "inference_samples_per_sec": total / max(t_inf, 1e-9),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-train", type=int, default=512)
    parser.add_argument("--num-val", type=int, default=128)
    parser.add_argument("--max-nodes", type=int, default=14)
    parser.add_argument("--feature-size", type=int, default=4)
    parser.add_argument("--edge-prob", type=float, default=0.35)
    parser.add_argument("--graph-feature-size", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbones", default="cfc,ltc,gru,liquid_tad")
    parser.add_argument(
        "--frozen-encoder", action="store_true",
        help=(
            "PRD §9 #4 / iter#13: two-stage training that emulates GCN-CfC. "
            "Phase 1 trains GraphSnapshotEncoder + a linear probe; "
            "Phase 2 freezes the encoder and trains only the recurrent head + readout."
        ),
    )
    parser.add_argument(
        "--frozen-pretrain-epochs", type=int, default=5,
        help="Epochs of phase-1 pretraining when --frozen-encoder is set.",
    )
    parser.add_argument("--output-dir", default="analysis/molecular")
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = torch.Generator().manual_seed(args.seed)

    print("=== Tox21-styled GraphLNN Molecular Smoke ===")
    print(f"Device: {device} | Train: {args.num_train} | Val: {args.num_val} | MaxNodes: {args.max_nodes}")

    train_data = _random_graph_batch(rng, args.num_train, args.max_nodes, args.feature_size, args.edge_prob)
    val_data = _random_graph_batch(rng, args.num_val, args.max_nodes, args.feature_size, args.edge_prob)
    pos_train = float(train_data[2].mean().item())
    pos_val = float(val_data[2].mean().item())
    print(f"Class balance — train +{pos_train*100:.1f}% / val +{pos_val*100:.1f}%")

    results = []
    for backbone in [b.strip() for b in args.backbones.split(",") if b.strip()]:
        print(f"\n--- backbone: {backbone}")
        record = _train_eval(backbone, train_data, val_data, args, device)
        print(
            f"    params={record['parameters']:,}  acc={record['val_accuracy']*100:.2f}%"
            f"  auc={record['val_auc_roc']:.4f}  train={record['train_seconds']:.2f}s"
            f"  inf={record['inference_samples_per_sec']:.0f} samples/s"
        )
        results.append(record)

    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_tox21_styled_graph_lnn.json"
    md_path = output_dir / f"{run_id}_tox21_styled_graph_lnn.md"

    payload = {
        "run_id": run_id,
        "experiment": "tox21_styled_graph_lnn",
        "generated_at": now.isoformat(),
        "config": vars(args),
        "class_balance": {"train_pos_ratio": pos_train, "val_pos_ratio": pos_val},
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
        },
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Tox21-styled GraphLNNPredictor Smoke — {run_id}",
        "",
        "## 任务",
        "- 模型: `lnn.core.graph.GraphLNNPredictor` (本仓端到端 GNN+LNN, PyTorch only)",
        "- 数据: 合成 Tox21-风格 — Erdős–Rényi 分子图 + (mean_degree>3 ⊻ triangle_ratio>0.05) 二分类",
        f"- 训练 / 验证: {args.num_train} / {args.num_val} | max_nodes={args.max_nodes} | edge_prob={args.edge_prob}",
        f"- 模型: graph_feat={args.graph_feature_size} hidden={args.hidden_size} | epoch={args.epochs} batch={args.batch_size} lr={args.lr} seed={args.seed}",
        f"- 类平衡: train +{pos_train*100:.1f}% / val +{pos_val*100:.1f}%",
        "",
        "## 结果",
        "| Backbone | 参数量 | Val acc | Val AUC | 训练秒 | 推理样本/秒 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{r['backbone']}` | {r['parameters']:,} | {r['val_accuracy']*100:.2f}% | "
            f"{r['val_auc_roc']:.4f} | {r['train_seconds']:.2f} | {r['inference_samples_per_sec']:.0f} |"
        )
    lines.extend([
        "",
        "## 解读",
        "- AUC ≥ 0.70 → 模型抓到了 degree/triangle 的结构信号;< 0.55 接近随机;",
        "- CfC / LTC vs GRU: 比 acc/AUC + 训练秒 + 推理吞吐三轴;",
        "- 比 `GCN-CfC` (Linlab2026) 的两阶段管线: 端到端单 stack,Jetson 部署友好。",
        "",
        f"JSON: `{json_path.relative_to(ROOT) if json_path.is_relative_to(ROOT) else json_path}`",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nJSON: {json_path}")
    print(f"MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
