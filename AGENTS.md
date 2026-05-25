---
title: 自动化 Agents 规划
tags:
  - Agent
  - workflow
  - automation
date: 2026-05-24
---

# 🤖 自动化 Agents 规划

为了更高效地追踪和研究液态神经网络（Liquid Neural Networks, LNN）领域的最新进展，本项目规划了一套基于 AI 的自动化 Agent 工作流。以下是项目中规划的 Agent 角色及功能设计：

## 1. 📄 论文追踪 Agent (Paper Tracker Agent)
- **目标**：自动监控 arXiv、Google Scholar、Hugging Face 等学术平台与信息源，获取 LNN 相关最新论文。
- **职责**：
  - 基于关键词（Liquid Neural Networks, CfC, LTC, Neural ODEs 等）进行定时检索。
  - 自动下载 PDF 并规范命名，归档到 `papers/daily/` 目录。
  - 提取论文基础信息（标题、作者、发表时间、摘要）并推送到追踪清单。

## 2. 📝 文档总结 Agent (Summarization Agent)
- **目标**：深入阅读论文，为**每篇论文生成独立的结构化研读报告**，并维护总知识库。
- **职责**：
  - 解析并阅读 PDF 论文内容。
  - **单篇报告生成规则 (Standard Operating Procedure)**：
    1. **文件命名**：在 `docs/reports/` 目录下生成独立文件，命名规范为 `论文文件名_研读报告.md`。
    2. **内容结构**：必须包含以下标准模块：
       - `元数据`：论文标题、作者、发表时间及标签。
       - `核心问题`：论文旨在解决的具体痛点（例如传统模型的局限）。
       - `方法论与核心思路`：创新点、模型架构设计，并说明上下文关系。
       - `核心公式提取`：提取论文中最关键的数学公式（使用 LaTeX 格式）。
       - `关键成果与贡献`：实验结果、性能指标提升。
       - `局限性与未来展望`：作者提及的不足或未来的研究方向。
  - **全局沉淀**：将单篇研读报告的精简版及链接，更新追加至全局索引文件 [[LNN_深度研读报告]] 中，保持知识库的系统性和迭代。

## 3. 💻 代码调研 Agent (Repo Analyzer Agent)
- **目标**：监控 GitHub 上的 LNN 开源实现与项目代码。
- **职责**：
  - 追踪高 Star 或高频更新的相关仓库。
  - 自动分析代码结构，评估其易用性、文档完整度以及复现难度。
  - 提供环境配置建议，并协助将有价值的仓库 clone 或记录至 `projects/` 目录中。

## 4. 📊 实验辅助 Agent (Experiment & Analysis Agent)
- **目标**：辅助运行基础复现实验及分析模型表现。
- **职责**：
  - 协助编写模型对比测试脚本，存放至 `scripts/` 目录。
  - 协助排查代码 Bug 与依赖报错。
  - 对比不同 LNN 模型变体（如 LTC vs CfC）的性能指标，将数据分析与可视化结果输出到 `analysis/` 目录。

---

## 🛠️ 工作流预期设计

1. **信息获取阶段**：由 **Paper Tracker Agent** 每日定期收集新出现的论文与代码库。
2. **深度理解阶段**：**Summarization Agent** 自动介入，针对有价值的文献生成速读摘要，帮助人类研究者快速决策是否精读。
3. **实践落地阶段**：**Repo Analyzer Agent** 协助配置环境和解析源码，结合 **Experiment Agent** 进行测试并记录 Benchmark。
4. **知识沉淀**：所有的产出自动汇总于 `docs/` 和 `analysis/` 文件夹中。可参考主页 [[README]] 了解完整项目结构。

## ✅ 当前已落地的自动化能力

- **每日追踪脚本**：`scripts/daily_lnn_research.py`，聚合 arXiv、GitHub 与 Hugging Face 的 LNN / LTC / CfC / NCP / LFM 相关更新。
- **本机定时任务**：`scripts/install_daily_lnn_timer.sh`，安装 user systemd timer，默认每天 06:30 运行 `scripts/run_daily_lnn_task.sh`。
- **GitHub Actions**：`.github/workflows/daily-lnn-research.yml`，每天自动生成研究摘要并推送回仓库。
- **Jetson 验证脚本**：`scripts/jetson_lnn_benchmark.py`，在 Jetson 上执行 LNN/CfC 风格模型与 GRU 的 quick benchmark，输出到 `analysis/jetson/`。
- **使用说明**：详见 [[每日自动化任务与Jetson验证]]。

---

## 🤖 通用 Agents / Skills (基于 Vercel Skills)

本项目使用 [Vercel Skills](https://github.com/vercel-labs/skills) 规范来管理专门用于**论文研读与分析**的 AI Agents。您可以将本项目 `skills/` 目录下的技能，通过软链接的方式一键导入到各种 AI 编码工具和终端中（如 Cursor, Windsurf, Trae, Claude Code 等），全面兼容 Claude, Gemini, Codex, Qwen, ChatGPT 等各类主流 AI 大模型工具，无需手动复制粘贴。

### 安装与使用
1. 确保您的环境中已安装 Node.js。
2. 运行以下命令，使用 `skills` CLI 将项目内的技能以软链接方式安装到您首选的 AI 助手中：

```bash
# 进入项目根目录
cd LNN

# 交互式添加本地技能 (软链接到您的 AI 工具)
npx skills add ./skills/paper-analyzer
npx skills add ./skills/paper-translator

# 此时，技能配置文件会软链接到工具的配置目录（如 .cursorrules 或 .trae/skills 等）
```

### 已实现的论文研究 Skills

- **`paper-analyzer` (论文分析专家)**
  - **功能**：读取并分析学术论文，结构化提取论文的“核心问题”、“方法论”、“关键成果”及“局限性”。（对应上述 Summarization Agent 职责）
  - **触发场景**：快速了解论文核心贡献，或帮您总结长篇文献时使用。
- **`paper-translator` (学术论文翻译专家)**
  - **功能**：用于学术英语与中文的精准翻译，严格保留学术术语（如 *Liquid Neural Networks*）、数学公式及排版。
  - **触发场景**：需要精准中英互译摘要、特定段落或通篇文献时使用。

您不仅可以在编辑器中调用，也可以将 `SKILL.md` 的内容发给 Web 端的 Claude, Gemini, Qwen 等大模型进行日常论文研读。
