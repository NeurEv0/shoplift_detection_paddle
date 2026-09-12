#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trimmed-M4 (timeline scorer) part 1: cut sliding windows along FULL master
timelines so the ep24 open model can be scored per window.

All outputs are NEW files under --out-root (default steps/timeline_windows/):
  <out-root>/<clip>_wNN/img_00001..00016.jpg   window dirs (16 consecutive master frames)
  <out-root>/timeline_meta.json  wname -> {clip, start(1-based master index)}
  <out-root>/timeline.list       rows "<wname> 16 0"  (labels unused; probs only)

Nothing existing is touched.

Usage:
  python scripts/make_master_windows.py --dry-run
  python scripts/make_master_windows.py
"""
import argparse
import json
import os
import os.path as osp
import shutil

ALL_CLIPS = ["steal_1_cc015", "steal_2_d03", "steal_3_d03",
             "normal_train1_cc015", "normal_train2_cc015", "normal_train3_d03",
             "normal_valid1_d03", "normal_test1_d03", "normal_test2_d03"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master-root", default="outputs/wqh/paddlevideo/steps/master")
    ap.add_argument("--meta", default="outputs/wqh/paddlevideo/steps/master_meta.json")
    ap.add_argument("--out-root", default="outputs/wqh/paddlevideo/steps/timeline_windows")
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--clips", default=",".join(ALL_CLIPS),
                    help="comma list of master clips to score")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mm = json.load(open(args.meta, encoding="utf-8"))
    clips = [c for c in args.clips.split(",") if c]
    total = 0
    for c in clips:
        n = int(mm[c]["n_frames"])
        if n < args.win:
            print(f"[skip] {c}: n={n} < win={args.win}")
            continue
        nw = (n - args.win) // args.stride + 1
        total += nw
        print(f"  {c:<24} n_frames={n:<4} -> {nw} windows")
    print(f"[summary] clips={len(clips)} total windows={total} "
          f"(win={args.win} stride={args.stride})")
    if args.dry_run:
        print("[dry-run] nothing written")
        return

    os.makedirs(args.out_root, exist_ok=True)
    meta_out = []
    rows = []
    n_copied = 0
    for c in clips:
        n = int(mm[c]["n_frames"])
        if n < args.win:
            continue
        src_dir = osp.join(args.master_root, c)
        for k, s in enumerate(range(1, n - args.win + 2, args.stride), 1):
            wname = f"{c}_w{k:02d}"
            wdir = osp.join(args.out_root, wname)
            if osp.exists(wdir):
                shutil.rmtree(wdir)
            os.makedirs(wdir)
            for j in range(s, s + args.win):
                shutil.copy2(osp.join(src_dir, f"img_{j:05d}.jpg"),
                             osp.join(wdir, f"img_{j - s + 1:05d}.jpg"))
                n_copied += 1
            meta_out.append(dict(win=wname, clip=c, start=s))
            rows.append(f"{wname} {args.win} 0")

    with open(osp.join(args.out_root, "timeline_meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(win=args.win, stride=args.stride, samples=meta_out),
                  f, ensure_ascii=False, indent=1)
    with open(osp.join(args.out_root, "timeline.list"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    print(f"[done] wrote {len(rows)} window dirs ({n_copied} frame copies) -> {args.out_root}")


if __name__ == "__main__":
    main()
