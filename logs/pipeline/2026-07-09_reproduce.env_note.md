# 2026-07-09 reproduce env note (appended to logs/pipeline/2026-07-09_reproduce.log)

## ⚠️ 环境无 torch, 全部 torch-based 复现脚本失败 (不影响跳过)

| 已检测 python | torch |
|---|---|
| `/Users/hyx/miniconda3/bin/python3` (3.13) | ❌ no module |
| `/usr/local/bin/python3.13` (homebrew 3.13.3) | ❌ no module |
| `/usr/local/bin/python3.12` (homebrew 3.12.10) | ❌ no module |
| `/Users/hyx/workspace/LNN/.venv/bin/python3` (3.13.3) | ❌ no module |
| `/usr/local/bin/python3` | ❌ no module |

复现脚本 (全部依赖 torch):
- `scripts/replicate_paper_experiment.py`
- `scripts/replicate_temporal_dropout.py`
- `scripts/experiment_imitation_lnn.py`
- 等共 10+ 个

今日 (2026-07-09) digest 命中 2 篇复现候选
(2606.26849v1 → imitation; 2605.27467v1 → temporal_dropout),
全部因 `ModuleNotFoundError: No module named 'torch'` 退出 1, 单次 4 秒内完成。

## 后续动作建议
1. 在 Jeston / 带 GPU 的机器装 torch 后, 直接重跑这两个脚本
   (不会改 digest, 跳过 arxiv fetch 阶段即可):
   ```bash
   pip install torch numpy pandas scikit-learn
   python3 scripts/replicate_temporal_dropout.py \
     --output_dir analysis/replication/temporal_dropout
   python3 scripts/experiment_imitation_lnn.py \
     --output_dir analysis/replication/imitation
   ```
2. 或者在 Mac 上用 uv 临时装 torch (CPU 版即可):
   ```bash
   uv run --with torch --with numpy --with pandas \
     python3 scripts/replicate_temporal_dropout.py
   ```
