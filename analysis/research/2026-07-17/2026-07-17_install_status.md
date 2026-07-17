# 本地 PDF 处理工具安装状态 — 2026-07-17

## ✅ 已安装:opendataloader-pdf 2.5.0
- 命令(Aliyun PyPI 镜像,绕过 github/HF SSL 阻断):
  ```
  .venv312/bin/pip install --index-url http://mirrors.aliyun.com/pypi/simple \
      --trusted-host mirrors.aliyun.com opendataloader-pdf
  ```
- 验证:
  ```
  .venv312/bin/python -c "from opendataloader_pdf import convert; print('OK')"
  ```
  → `opendataloader-pdf import OK` ✓
- 路径:`/Users/hyx/workspace/LNN/.venv312/lib/python3.12/site-packages/opendataloader_pdf/`
- 来源:`opendataloader-project/opendataloader-pdf` GitHub(v2.4.7 最新 release / PyPI 镜像 2.5.0)
- 用途:PDF → Markdown / JSON / HTML,保留版式,可用于 RAG / 论文研究 pipeline。

## ❌ 暂未安装:Unlimited-OCR(Baidu)
- 来源:`baidu/Unlimited-OCR` GitHub(3B 总参,570M 激活,端到端长文档 OCR,2026-06-22 开源)
- 试过的安装路径(全部被本地网络拦截):
  | 路径 | 结果 |
  |---|---|
  | `pip install unlimited-ocr` (PyPI) | `No matching distribution found` —— 不在 PyPI |
  | `pip install --index-url aliyun unlimited-ocr` | 同上 |
  | `git clone https://github.com/baidu/Unlimited-OCR.git` | SSL_ERROR_SYSCALL 拦截 |
  | `git clone https://gitcode.com/baidu/Unlimited-OCR.git` | 404 (Project not found) |
  | `git clone https://ai.gitcode.com/hf_mirrors/baidu/Unlimited-OCR.git` | redirect→home,mirror 不可达 |
  | `git clone https://code.aliyun.com/baidu/Unlimited-OCR.git` | SSL_ERROR_SYSCALL |
  | `modelscope snapshot_download 'baidu/Unlimited-OCR'` | 404 record not found |
  | `huggingface_hub.snapshot_download('baidu/Unlimited-OCR', endpoint='hf-mirror.com')` | SSL EOF |
  | `curl https://codeload.github.com/baidu/Unlimited-OCR/...` | SSL_ERROR_SYSCALL |

## 解决方案(下次有 VPN / 代理恢复后再装)
1. **优先 PyPI 法**:等待 Baidu 发布 unlimited-ocr PyPI wheel(预估 1-2 月内)。
2. **GitHub 法**:VPN 恢复后克隆 `https://github.com/baidu/Unlimited-OCR.git`,然后按 README 安装。
3. **HuggingFace 法**:VPN 恢复后 `huggingface-cli download baidu/Unlimited-OCR --local-dir tools/Unlimited-OCR/`。
4. **替代 OCR**(本机已有可用):
   - `pdfplumber` 已在 venv312 中 ✓ —— 用于纯文本/表格提取
   - `PyMuPDF`(fitz)—— 高保真 PDF 渲染 + 文本
   - `pytesseract` —— 系统 Tesseract OCR(若有)
   - **opendataloader-pdf** 本身也带 layout analysis,可以替代部分 OCR 功能

## 影响评估
- 当前 `papers/daily/*.pdf` → 文本提取有 pdfplumber/PyMuPDF 兜底,影响 ≤ 一项工作流 — 不阻塞本轮 round。
- `analysis/repo_watchlist` 已存成 JSON,可以文本搜索/索引,无需 OCR。
- 下一轮 round 报告里继续追踪此状态。
