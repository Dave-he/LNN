---
title: LNN 每日研究追踪 - 2026-07-17
date: 2026-07-17
tags: [LNN, daily, automation, arxiv, github, huggingface]
---

# LNN 每日研究追踪 - 2026-07-17

> 自动生成：聚合 arXiv、GitHub 与 Hugging Face 的 LNN / LTC / CfC / NCP / LFM 相关更新，供人工筛选后进入深度研读。

## 摘要
- arXiv 候选论文：0 篇
- GitHub 候选仓库：0 个
- Hugging Face 候选模型：21 个
- 已下载 PDF：0 个

## 数据源状态
- `arXiv fetch failed: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1028)>`
- 若当天已有历史结果，脚本会保留上一轮成功获取的数据，避免 transient API 错误清空候选池。

## arXiv 候选论文
- 本次未发现通过关键词过滤的新候选论文。

## GitHub 候选仓库
- 本次未发现可记录的 GitHub 仓库。

## Hugging Face 候选模型
| 更新 | 模型 | 下载 | Likes | 任务 |
|---|---|---:|---:|---|
| 2026-07-16 | [reaperdoesntknow/LFM2.5-8B-A1B-Opus-Distil](https://huggingface.co/reaperdoesntknow/LFM2.5-8B-A1B-Opus-Distil) | 2400 | 5 | text-generation |
| 2026-07-16 | [Synaptics/liquidAI-LFM2p5-230M-LLM](https://huggingface.co/Synaptics/liquidAI-LFM2p5-230M-LLM) | 1093 | 0 | text-generation |
| 2026-07-16 | [hauser458original/lfm2.5-230m-code-math](https://huggingface.co/hauser458original/lfm2.5-230m-code-math) | 378 | 3 | text-generation |
| 2026-07-16 | [hauser458original/lfm2.5-230m-code-math-GGUF](https://huggingface.co/hauser458original/lfm2.5-230m-code-math-GGUF) | 189 | 2 | text-generation |
| 2026-07-16 | [dsaad68/LFM2.5-Embedding-350M-ONNX](https://huggingface.co/dsaad68/LFM2.5-Embedding-350M-ONNX) | 0 | 0 |  |
| 2026-07-16 | [trl-internal-testing/tiny-Lfm2ForCausalLM-2.5](https://huggingface.co/trl-internal-testing/tiny-Lfm2ForCausalLM-2.5) | 0 | 0 | text-generation |
| 2026-07-16 | [trl-internal-testing/tiny-Lfm2ForCausalLM](https://huggingface.co/trl-internal-testing/tiny-Lfm2ForCausalLM) | 0 | 0 | text-generation |
| 2026-07-16 | [simaai/LFM2.5-1.2B-Thinking-Autoround-Safetensors](https://huggingface.co/simaai/LFM2.5-1.2B-Thinking-Autoround-Safetensors) | 0 | 0 | text-generation |
| 2026-07-16 | [simaai/LFM2.5-1.2B-Instruct-Autoround-Safetensors](https://huggingface.co/simaai/LFM2.5-1.2B-Instruct-Autoround-Safetensors) | 0 | 0 | text-generation |
| 2026-07-16 | [simaai/LFM2-2.6B-Autoround-Safetensors](https://huggingface.co/simaai/LFM2-2.6B-Autoround-Safetensors) | 0 | 0 | text-generation |
| 2026-07-16 | [simaai/LFM2-1.2B-Autoround-Safetensors](https://huggingface.co/simaai/LFM2-1.2B-Autoround-Safetensors) | 0 | 0 | text-generation |
| 2026-07-16 | [hauser458original/lfm2.5-230m-code-math-v2-GGUF](https://huggingface.co/hauser458original/lfm2.5-230m-code-math-v2-GGUF) | 0 | 2 | text-generation |

## 建议动作
- 对标题和摘要同时命中 LNN/LTC/CfC/NCP 的论文，优先用 `skills/paper-analyzer` 生成独立研读报告。
- 对最近更新且 Star 较高的仓库，优先记录复现成本、依赖栈和 Jetson 部署可行性。
- 对 LFM2/LFM2.5 相关模型，优先筛选 350M、450M、1.2B 等边缘友好规格，进入 Jetson 量化/推理验证队列。

## 数据源
- arXiv API: https://export.arxiv.org/api/query
- GitHub Search API: https://docs.github.com/rest/search/search
- Hugging Face Models API: https://huggingface.co/docs/hub/api
