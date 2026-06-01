#!/bin/bash

# LNN Demo - 快速启动 Xcode 项目
# =================================

echo "========================================"
echo "  LNN iPad 应用 - 快速启动"
echo "========================================"
echo ""

# 检查项目文件
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &amp;&amp; pwd )"
XCODE_PROJECT="$PROJECT_DIR/LNNDemo.xcodeproj"

if [ ! -d "$XCODE_PROJECT" ]; then
    echo "❌ 错误：找不到 Xcode 项目文件"
    echo "   检查路径: $XCODE_PROJECT"
    exit 1
fi

echo "✅ 找到项目文件"
echo "📁 项目路径: $PROJECT_DIR"
echo ""

# 检查 Xcode 是否安装
if ! command -v xcodebuild &amp;&gt; /dev/null; then
    echo "❌ 警告：未找到 Xcode 命令行工具"
    echo "   请先从 Mac App Store 安装 Xcode"
    echo ""
fi

# 打开项目
echo "🚀 正在打开 Xcode 项目..."
open "$XCODE_PROJECT"

echo "✅ 项目已打开！"
echo ""
echo "📖 下一步操作："
echo "1. 在 Xcode 中配置 Team 和 Bundle ID"
echo "2. 连接 iPad 并选择作为目标设备"
echo "3. 按 Cmd+R 运行应用"
echo ""
echo "📚 详细说明请查看：iPad真机部署和调试指南.md"
echo ""
