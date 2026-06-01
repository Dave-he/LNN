#!/bin/bash

# LNN 真机调试助手
# 这个脚本帮助检查常见配置问题并提供建议

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_DIR="/Users/hyx/workspace/LNN/ios/LNNDemo"
XCODE_PROJECT="$PROJECT_DIR/LNNDemo.xcodeproj"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   LNN 真机调试助手 🔧${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${YELLOW}[1/6]${NC} 检查项目文件..."
if [ -d "$XCODE_PROJECT" ]; then
    echo -e "${GREEN}✅${NC} 找到 Xcode 项目: $XCODE_PROJECT"
else
    echo -e "${RED}❌${NC} 找不到 Xcode 项目"
    exit 1
fi

echo ""
echo -e "${YELLOW}[2/6]${NC} 检查 Xcode 命令行工具..."
if command -v xcodebuild >/dev/null 2>&1; then
    XCODE_VERSION=$(xcodebuild -version 2>/dev/null | head -n1)
    echo -e "${GREEN}✅${NC} $XCODE_VERSION"
else
    echo -e "${RED}❌${NC} Xcode 命令行工具未安装或未配置"
fi

echo ""
echo -e "${YELLOW}[3/6]${NC} 检查源文件..."
cd "$PROJECT_DIR"
files_ok=0

check_file() {
    if [ -f "$1" ]; then
        ((files_ok++))
        echo -e "${GREEN}  ✓${NC} $1"
    else
        echo -e "${RED}  ✗${NC} $1 (缺失)"
    fi
}

check_file "LNNDemo/LNNDemoApp.swift"
check_file "LNNDemo/Info.plist"
check_file "LNNDemo/Models/CfCModel.swift"
check_file "LNNDemo/ViewModels/MainViewModel.swift"
check_file "LNNDemo/Views/ContentView.swift"
check_file "LNNDemo/Views/LineChartView.swift"

echo ""
echo -e "${GREEN}${files_ok}/6${NC} 源文件检查完成"

echo ""
echo -e "${YELLOW}[4/6]${NC} 检查 gitignore..."
if [ -f ".gitignore" ]; then
    echo -e "${GREEN}✅${NC} .gitignore 存在"
else
    echo -e "${YELLOW}⚠️${NC} .gitignore 缺失"
fi

echo ""
echo -e "${YELLOW}[5/6]${NC} 检查 Bundle ID 配置..."
PBXPROJ="$XCODE_PROJECT/project.pbxproj"
if [ -f "$PBXPROJ" ]; then
    if grep -q "com.example.LNNDemo" "$PBXPROJ"; then
        echo -e "${YELLOW}⚠️${NC} Bundle Identifier 仍为默认值"
        echo "   ${BLUE}建议${NC}：在 Xcode 中修改为唯一值"
    else
        echo -e "${GREEN}✅${NC} Bundle Identifier 配置可能已更新"
    fi
else
    echo -e "${RED}❌${NC} 无法读取项目配置"
fi

echo ""
echo -e "${YELLOW}[6/6]${NC} 完成检查！"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   下一步操作建议${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "1. 在 Xcode 中配置："
echo "   - Team: 选择你的 Apple ID"
echo "   - Bundle Identifier: 修改为唯一值"
echo ""
echo "2. 连接 iPad 并配置："
echo "   - 用 USB 线连接 iPad 到 Mac"
echo "   - 在 iPad 上点击「信任」"
echo "   - 在 iPad 上开启开发者模式"
echo ""
echo "3. 在 Xcode 中选择设备并运行："
echo "   - 选择你的 iPad"
echo "   - 按 Cmd + R 运行"
echo ""
echo "📖 详细指南："
echo "   - 网页版已在浏览器中打开"
echo "   - iPad真机部署和调试指南.md"
echo ""
echo -e "${GREEN}========================================${NC}"
echo ""
echo "正在打开 Xcode 项目..."
open "$XCODE_PROJECT"
echo "完成！祝你调试愉快！🚀"
