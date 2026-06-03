#!/usr/bin/env bash
# Jetson Orin Nano (BSP R36.4+ / CUDA 12.6) PyTorch CUDA 环境
#
# Source 本文件即可让 system python3.10 (装了 jetson-ai-lab cu126 wheel) 启用 CUDA:
#   $ . scripts/jetson_cuda_env.sh
#   $ python3.10 -c "import torch; print(torch.cuda.is_available())"
#   True
#
# 安装顺序(只需一次):
#   1) python3.10 -m pip install --force-reinstall --no-deps \
#        --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/ torch==2.10.0
#   2) mkdir -p ~/.local/opt && cd ~/.local/opt
#      curl -sSLO https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-aarch64/libcudss-linux-aarch64-0.8.0.10_cuda12-archive.tar.xz
#      tar -xJf libcudss-linux-aarch64-0.8.0.10_cuda12-archive.tar.xz
#
# 已知限制(2026-06-03):
# - torch 2.11.0 引入对 libcudss 的硬依赖,且 jetson-ai-lab 镜像未捆绑 cudss,
#   所以选 torch 2.10.0(2.11.0 也能跑,只是要先装 cudss)
# - Jetson Orin Nano Super 是统一显存(8 GB LPDDR5);CUDA 启动后能否 cudaMalloc
#   取决于当前系统空余 RAM。空载窗口跑 GPU 路径成功率最高。
# - 详细修复记录见 analysis/jetson/2026-06-03_loop_iteration2_cuda_fix_pareto.md

CUDSS_HOME="${CUDSS_HOME:-$HOME/.local/opt/libcudss-linux-aarch64-0.8.0.10_cuda12-archive}"

if [[ -d "${CUDSS_HOME}/lib" ]]; then
  export LD_LIBRARY_PATH="${CUDSS_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  echo "[warn] jetson_cuda_env.sh: libcudss not found at ${CUDSS_HOME}/lib" >&2
  echo "       Falling through without setting LD_LIBRARY_PATH." >&2
fi

# 可选: 强制选用 system python3.10(它是 jetson cu126 wheel 的目标解释器)
export JETSON_PYTHON="${JETSON_PYTHON:-/usr/bin/python3.10}"
