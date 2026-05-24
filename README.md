---
title: Liquid Neural Networks (LNN) Research & Projects
tags:
  - LNN
  - AI
  - project
date: 2026-05-24
---

# Liquid Neural Networks (LNN) Research & Projects

欢迎来到 **LNN (Liquid Neural Networks)** 研究与开源项目追踪仓库。本仓库旨在收集、整理并分析液态神经网络领域的最新论文、开源项目以及相关的实验代码。

## 📂 目录结构

```text
LNN/
├── AGENTS.md                   # 自动化 Agent 规划与工作流说明
├── README.md                   # 项目概述与指南 (本文档)
├── docs/                       # 文档目录（调研报告、论文总结、学习笔记等）
├── papers/                     # 论文归档与每日追踪
│   └── daily/                  # 每日/定期论文抓取记录
├── prompts/                    # 通用 AI Agents / Skills 提示词配置 (兼容多平台)
│   ├── paper-analyzer.md
│   └── paper-translator.md
├── projects/                   # 开源项目克隆、复现代码与实验项目
├── analysis/                   # 实验结果分析、数据或可视化相关
└── scripts/                    # 自动化脚本（论文抓取、数据处理等）
```

## 🎯 项目目标

1. **追踪前沿**：持续追踪液态神经网络（LNN）及相关领域（如连续时间循环神经网络、神经常微分方程）的最新学术进展。
2. **源码分析**：汇总、分析和复现主流的开源 LNN 框架和应用案例（如时间序列预测、自动驾驶决策等）。
3. **自动化研究**：构建基于 AI Agent 的自动化信息收集与分析工作流，提升科研效率。

## 🚀 快速开始

- 想要了解当前最新进展，请阅读：[[液态神经网络最新进展与开源项目调研]]
- 想要了解最新的论文总结，请阅读：[[Liquid_Neural_Networks_Latest_Papers_Summary|LNN 最新论文总结]]
- 关于本项目中自动化工具与工作流的规划，请参阅：[[AGENTS]]

## 📝 Obsidian 导入说明与使用规则

本项目完全兼容并推荐作为 **Obsidian Vault (知识库)** 导入，以获得最佳的双向链接阅读与网状知识管理体验。

### 📥 如何导入

1. 下载或 Clone 本项目到本地：`git clone https://github.com/Dave-he/LNN.git`
2. 启动 Obsidian，点击 **"Open folder as vault" (打开文件夹作为仓库)**。
3. 选择本地的 `LNN` 文件夹。
4. 导入完成！你可以在 Obsidian 中直接查看、编辑和浏览各个 LNN 文档之间的双链关联。

### ✍️ 写作与文档维护规则

为了保证项目在 GitHub 上的可读性，同时发挥 Obsidian 的最大优势，请在协作时遵循以下规则：

1. **双向链接语法**：文档之间的交叉引用请优先使用双向链接 `[[页面名称]]` 或 `[[页面名称|显示别名]]`。GitHub 目前已原生支持解析此类链接。
2. **文档命名**：
   - 优先使用有意义的英文或中文命名。
   - 避免使用系统中不允许的特殊符号。多个单词建议使用下划线 `_` 或中划线 `-` 连接。
3. **附件与图片**：
   - 插入图片或 PDF 附件时，推荐统一放置在对应文档同级目录的 `assets/` 文件夹下。
   - 建议在 Obsidian 设置中将 `Default location for new attachments` 设置为 `In subfolder under current folder`，并命名为 `assets`。
4. **元数据 (YAML Frontmatter)**：
   - 建议在每篇新建研究报告或笔记顶部添加 YAML frontmatter，至少包含 `title`, `tags`, `date` 等字段，便于 Obsidian 进行检索与属性管理。

## 🤖 通用 Agents / Skills (论文分析与研究)

本项目提取了专门用于**论文研读与分析**的 AI Agent 提示词配置（位于 `prompts/` 目录下）。这些 Prompt 是完全跨平台和模型无关的，**全面兼容 Claude, Gemini, Codex, Qwen, ChatGPT 等各类主流 AI 大模型工具**，也同样适用于 Cursor, Windsurf, Trae 等 AI 代码编辑器。

您可以直接将 `prompts/` 目录下的 `.md` 文件内容：
1. 粘贴发送给 Web 端的 **Claude / Gemini / Qwen**，让其扮演对应的专家角色；
2. 作为 System Prompt (系统提示词) 填入 API 调用代码中；
3. 配置到 AI IDE 的 `.cursorrules` / Custom Instructions 中进行全局挂载调用。

- **`prompts/paper-analyzer.md` (论文分析专家)**
  - **功能**：读取并分析学术论文，结构化提取论文的“核心问题”、“方法论”、“关键成果”及“局限性”。
  - **触发场景**：快速了解论文核心贡献，或帮您总结长篇文献时使用。
- **`prompts/paper-translator.md` (学术论文翻译专家)**
  - **功能**：用于学术英语与中文的精准翻译，严格保留学术术语（如 *Liquid Neural Networks*）、数学公式及排版。
  - **触发场景**：需要精准中英互译摘要、特定段落或通篇文献时使用。

## 🚀 后续计划
