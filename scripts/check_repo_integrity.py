#!/usr/bin/env python3
"""仓库完整性门禁 —— 提交前拦住冲突标记和写坏的 JSON.

**为什么需要**: 2026-10-17 有 5 个被 git 跟踪的文件带着 merge 冲突标记进了
origin (`papers/daily/2026-10-17_lnn_research.json` 等), 一直没人发现, 直到
10-19 读 JSON 时报 `Expecting property name enclosed in double quotes`。
根因是那次 GitHub 403 触发的补救性重跑产生了降级产物, 随后一次带冲突的合并
被直接提交了 —— `push_with_rebase` 自己会 `rebase --abort`, 但绕开它的路径不会。

**为什么必须放在提交之前**: 这类损坏一旦进了 origin, 日常跑 digest 读 JSON
会直接抛异常, 而流水线本身是"照常产出"的, 所以它能一路潜伏好几天。

**为什么不用 `git diff --check`**: 那个只查**本次**改动里的空白错误, 已经被
提交进历史的冲突标记它一个都查不出来。这里查的是工作区的**全部被跟踪文件**。

退出码非 0 = 提交被叫停, 需要人工介入。
"""

from __future__ import annotations

import json
import pathlib
import re  # noqa: F401  (CONFLICT_* 正则需要)
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 冲突标记必须按**块**解析, 不能逐行独立匹配 —— markdown 里 `=======` 是合法
# 的分隔线 (setext 标题的下划线也长这样), 逐行匹配会把它误判成冲突。
# 真正的冲突块结构固定: <<<<<<< ... ======= ... >>>>>>> , 三者必须成组出现。
CONFLICT_START = re.compile(r"^<<<<<<< ")
CONFLICT_MIDDLE = re.compile(r"^=======\s*$")
CONFLICT_END = re.compile(r"^>>>>>>> ")

# 二进制文件不做文本扫描 (PDF/ONNX/tensorrt 权重等)。
BINARY_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".onnx", ".pt", ".pth",
    ".npy", ".npz", ".so", ".bin", ".zip", ".tar", ".gz", ".wav", ".engine",
}


def tracked_files() -> list[pathlib.Path]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    names = [n for n in out.stdout.decode("utf-8", "replace").split("\0") if n]
    return [ROOT / n for n in names if (ROOT / n).exists()]


def scan_conflict_markers(files: list[pathlib.Path]) -> list[str]:
    problems: list[str] = []
    for path in files:
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(ROOT)
        problems.extend(f"冲突标记: {rel}:{lineno}: {line[:60]}"
                        for lineno, line in find_conflict_blocks(text))
    return problems


def find_conflict_blocks(text: str) -> list[tuple[int, str]]:
    """返回真正成组的冲突块 (起始行号, 该行内容).

    只有 `<<<<<<<` 开头的行才进入冲突状态; 在遇到配对的 `=======` / `>>>>>>>`
    之前, 后面所有行都不算冲突。这样正文里的裸 `=======` 分隔线不会被误报,
    而半截冲突 (`<<<<<<<` 后面没有收尾) 仍然会被抓住。
    """
    hits: list[tuple[int, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not CONFLICT_START.match(line):
            i += 1
            continue
        hits.append((i + 1, line))
        saw_middle = False
        j = i + 1
        while j < len(lines):
            if CONFLICT_MIDDLE.match(lines[j]):
                saw_middle = True
            elif CONFLICT_END.match(lines[j]):
                break
            j += 1
        if j >= len(lines) or not saw_middle:
            # 标记没闭合 —— 冲突确实存在, 且多半是文件被截断了, 后面照常扫。
            hits.append((j + 1, "(冲突块未闭合)"))
            i = j + 1
        else:
            i = j + 1
    return hits


def scan_json(files: list[pathlib.Path]) -> list[str]:
    problems: list[str] = []
    for path in files:
        if path.suffix.lower() != ".json":
            continue
        rel = path.relative_to(ROOT)
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"JSON 解析失败: {rel}: {exc}")
        except (UnicodeDecodeError, OSError) as exc:
            problems.append(f"JSON 读取失败: {rel}: {exc}")
    return problems


def main() -> int:
    files = tracked_files()
    problems = scan_conflict_markers(files) + scan_json(files)
    if not problems:
        print(
            f"[OK] 仓库完整性: {len(files)} 个被跟踪文件, 无冲突标记, JSON 全部可解析"
        )
        return 0

    print(f"[FAIL] 仓库完整性检查未通过 ({len(problems)} 处):", file=sys.stderr)
    for problem in problems[:40]:
        print(f"  - {problem}", file=sys.stderr)
    if len(problems) > 40:
        print(f"  ... 另有 {len(problems) - 40} 处", file=sys.stderr)
    print(
        "\n这通常意味着一次带冲突的合并被提交进了仓库。修复方式:\n"
        "  1. 确认两侧内容, 手工保留正确的一侧 (别默认 ours 就对 —— 降级产物也可能在一侧)\n"
        "  2. 重新提交并推送\n"
        "  3. python3 scripts/check_repo_integrity.py 复验",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
