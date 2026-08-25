"""labelImg + PyQt5（sip 6）float 参数崩溃的源码补丁（Python 3.10 用）。

背景：PyQt5 >= 5.15.5 基于 sip 6 构建，QSpinBox/QScrollBar.setValue、
QPainter.drawLine/drawRect/drawText 等要求 int 参数，而 labelImg 1.8.6
的缩放/滚动/画框代码传 float，会在启动或标注时崩溃
（TypeError: ... argument 1 has unexpected type 'float'）。
Python 3.10 下所有 pip 版 PyQt5 wheel 均如此，降级 PyQt5 无效，只能改源码。

本脚本把安装目录里 labelImg 的 8 处 float 调用点包上 int()。
幂等：已打补丁时再次运行会报告 "already patched"。
重装/升级 labelImg 后重新运行本脚本即可。

用法：
    python scripts/patch_labelimg_pyqt5.py                # 自动定位 site-packages
    python scripts/patch_labelimg_pyqt5.py --base <site-packages 路径>   # 手动指定
"""
from __future__ import annotations

import argparse
import pathlib
import sys

try:
    import labelImg  # noqa: PLC0415

    DEFAULT_BASE = pathlib.Path(labelImg.__file__).resolve().parent.parent
except Exception:  # noqa: BLE001  (labelImg 未安装时给出可读报错)
    DEFAULT_BASE = None

PATCHES: list[tuple[str, str, str]] = [
    # labelImg/labelImg.py
    (
        "labelImg/labelImg.py",
        "bar.setValue(bar.value() + bar.singleStep() * units)",
        "bar.setValue(int(bar.value() + bar.singleStep() * units))",
    ),
    (
        "labelImg/labelImg.py",
        "self.zoom_widget.setValue(value)",
        "self.zoom_widget.setValue(int(value))",
    ),
    (
        "labelImg/labelImg.py",
        "h_bar.setValue(new_h_bar_value)",
        "h_bar.setValue(int(new_h_bar_value))",
    ),
    (
        "labelImg/labelImg.py",
        "v_bar.setValue(new_v_bar_value)",
        "v_bar.setValue(int(new_v_bar_value))",
    ),
    # libs/canvas.py
    (
        "libs/canvas.py",
        "p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)",
        "p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))",
    ),
    (
        "libs/canvas.py",
        "p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())",
        "p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), self.pixmap.height())",
    ),
    (
        "libs/canvas.py",
        "p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())",
        "p.drawLine(0, int(self.prev_point.y()), self.pixmap.width(), int(self.prev_point.y()))",
    ),
    # libs/shape.py
    (
        "libs/shape.py",
        "painter.drawText(min_x, min_y, self.label)",
        "painter.drawText(int(min_x), int(min_y), self.label)",
    ),
]


def _read_text(path: pathlib.Path) -> tuple[str, bool]:
    raw = path.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    return raw.decode("utf-8-sig" if has_bom else "utf-8"), has_bom


def _write_text(path: pathlib.Path, text: str, has_bom: bool) -> None:
    data = text.encode("utf-8")
    if has_bom:
        data = b"\xef\xbb\xbf" + data
    path.write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(DEFAULT_BASE or ""), help="site-packages 目录")
    args = parser.parse_args()

    if not args.base:
        print("未找到 labelImg 包，请先安装：python -m pip install labelImg；或 --base 指定 site-packages")
        return 1
    base = pathlib.Path(args.base)

    failed = 0
    patched = 0
    already = 0
    for rel, old, new in PATCHES:
        path = base / rel
        if not path.exists():
            print(f"[WARN] 文件不存在: {rel}")
            failed += 1
            continue
        text, has_bom = _read_text(path)
        if new in text and old not in text:
            print(f"[OK]   {rel}: already patched")
            already += 1
            continue
        count = text.count(old)
        if count != 1:
            print(f"[WARN] {rel}: 模式出现 {count} 次，跳过 -> {old[:55]!r}")
            failed += 1
            continue
        _write_text(path, text.replace(old, new), has_bom)
        print(f"[OK]   {rel}: patched ({old[:45]}...)")
        patched += 1

    print(f"patched={patched} already={already} failed={failed} / total={len(PATCHES)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
