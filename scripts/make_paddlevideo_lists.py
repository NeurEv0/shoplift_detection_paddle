#!/usr/bin/env python3
"""
Assemble PP-TSM FrameDataset data root and write train/valid/test lists.

Layout (as user confirmed):
  <data_root>/
    steal_1_cc015/    normal_train1_cc015/  ... (clip dirs, symlinked or copied)
    train.list   valid.list   test.list

Split (user-confirmed):
  train: steal_1 + normal_train1, normal_train2, normal_train3       (4 clips)
  valid: steal_2 + normal_valid1                                     (2 clips)
  test : steal_3 + normal_test1, normal_test2                        (3 clips)

List line format (PP-TSM FrameDataset): "<clip_path> <num_frames> <label>"
  label: 1 = steal, 0 = normal
Usage:
  python make_paddlevideo_lists.py \
      --steal-root <clips_steal> --normal-root <clips_normal> \
      --data-root <paddlevideo_data> [--copy]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


STEAL_CLIPS = {
    "steal_1_cc015": "train",
    "steal_2_d03": "valid",
    "steal_3_d03": "test",
}
NORMAL_CLIPS = {
    "normal_train1_cc015": "train",
    "normal_train2_cc015": "train",
    "normal_train3_d03": "train",
    "normal_valid1_d03": "valid",
    "normal_test1_d03": "test",
    "normal_test2_d03": "test",
}


def count_frames(clip_dir: Path) -> int:
    """Count img_*.jpg frames. os.listdir is symlink-safe (glob can miss symlinked dirs)."""
    try:
        names = os.listdir(clip_dir)
    except OSError:
        return 0
    return sum(1 for n in names if n.startswith("img_") and n.endswith(".jpg"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steal-root", required=True)
    p.add_argument("--normal-root", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--copy", action="store_true",
                   help="Copy clip dirs instead of symlinking (for non-POSIX fs).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    steal_root = Path(args.steal_root)
    normal_root = Path(args.normal_root)
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    entries = {}  # split -> list of (clip_name, n_frames, label)
    for name, split in {**STEAL_CLIPS, **NORMAL_CLIPS}.items():
        src = steal_root / name if name.startswith("steal") else normal_root / name
        if not src.is_dir():
            raise SystemExit(f"missing clip dir: {src}")
        dst = data_root / name
        if not (dst.exists() or dst.is_symlink()):
            if args.copy:
                shutil.copytree(src, dst)
                print(f"[copy] {name}")
            else:
                try:
                    # Use absolute target so the link resolves regardless of
                    # the link's own directory (relative targets break here).
                    dst.symlink_to(src.resolve(), target_is_directory=True)
                    print(f"[link] {name}")
                except OSError:
                    shutil.copytree(src, dst)
                    print(f"[copy-fallback] {name}")
        else:
            print(f"[exists] {name}")
        n = count_frames(dst)
        label = 1 if name.startswith("steal") else 0
        entries.setdefault(split, []).append((name, n, label))

    for split in ("train", "valid", "test"):
        lines = [f"{name} {n} {label}" for name, n, label in entries[split]]
        out = data_root / f"{split}.list"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[list] {split}.list ({len(lines)} lines):")
        for line in lines:
            print(f"    {line}")

    summary = {
        split: [{"clip": n, "frames": c, "label": l} for n, c, l in rows]
        for split, rows in entries.items()
    }
    (data_root / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] data root: {data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
