#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2.5B: build the BINARY (open拆包装 vs not-open) window training set.

All outputs are NEW files under --out-root (default steps/open_data/):
  <out-root>/<pos|neg>_<segment>_wNN/img_00001..00016.jpg   window samples
  <out-root>/train.list     rows: <dir> 16 <label>   (1=open拆包装, 0=not-open)
  <out-root>/window_meta.json
  <out-root>/open_eval.list segment-level eval rows (steal_3 + normal_test2),
                              data_prefix = steps/data (existing clip dirs)

Nothing existing is touched: steps/data, window_data, step/4-class/binary
configs and outputs stay as they are.

Usage:
  python scripts/make_open_binary_data.py --dry-run          (preset 1)
  python scripts/make_open_binary_data.py --preset 2 --dry-run
  python scripts/make_open_binary_data.py                    (preset 1, real)
"""
import argparse
import json
import os
import os.path as osp
import shutil

POS_SEG = "steal_2_d03_s2"          # the only unwrap segment in train domain

# (segment, stride, max_windows) ; max_windows kept EVENLY SPACED over time
# EXCLUDED sources (hand actions not fully visible, per user):
#   steal_1_cc015 ALL segments, normal_test1_d03, normal_valid1_d03
PRESETS = {
    1: dict(
        neg=[
            ("steal_2_d03_s3", 2, 20),          # conceal(藏匿) 长段
            ("normal_train1_cc015_s2", 2, 14),  # reach(正常人伸手, 难例)
            ("steal_2_d03_s1", 2, 5),           # reach(偷盗伸手)
            ("normal_train3_d03_s1", 2, 5),     # idle 静止兜底
        ],
    ),
    2: dict(
        neg=[
            ("steal_2_d03_s3", 2, 29),
            ("normal_train1_cc015_s2", 2, 20),
            ("steal_2_d03_s1", 2, 9),
            ("normal_train3_d03_s1", 2, 17),
        ],
    ),
    # v2 retrain: preset1 negatives + FP-suppression hard negatives
    # (normal people actively using hands; normal_test2 kept OUT of train as eval probe)
    3: dict(
        neg=[
            ("steal_2_d03_s3", 2, 20),          # conceal 长段 (原有)
            ("normal_train1_cc015_s2", 2, 14),  # 正常人伸手 (原有)
            ("steal_2_d03_s1", 2, 5),           # 偷盗伸手 (原有)
            ("normal_train3_d03_s1", 2, 5),     # 静止 (原有)
            ("normal_test1_d03_s1", 2, 17),     # 新增: 吃面(手口持续动作 FP源)
            ("normal_train2_cc015_s1", 2, 17),  # 新增: idle 后段持续动作 FP源
            ("normal_train1_cc015_s1", 2, 8),   # 新增: idle 前段 FP源
            ("steal_1_cc015_s1", 2, 1),         # 新增: steal_1 装包(侧视 FP源)
            ("steal_1_cc015_s2", 2, 8),
            ("steal_1_cc015_s3", 2, 2),
            ("steal_1_cc015_s4", 2, 4),
        ],
    ),
}

EVAL_SEGS = [  # (segment, label)  -> steal_3 fully held out + one normal
    ("steal_3_d03_s2", 1),   # open 拆包装 -> 正
    ("steal_3_d03_s1", 0),   # reach
    ("steal_3_d03_s3", 0),   # conceal
    ("normal_test2_d03_s1", 0),
]


def even_spaced(seq, n):
    """keep n items of seq, evenly spaced (first & last kept)."""
    if n >= len(seq):
        return seq
    idx = sorted({round(i * (len(seq) - 1) / (n - 1)) for i in range(n)}) if n > 1 else [0]
    return [seq[i] for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta",
                    default="outputs/wqh/paddlevideo/steps/data/step_clips_meta.json")
    ap.add_argument("--src-root", default="outputs/wqh/paddlevideo/steps/data")
    ap.add_argument("--out-root", default="outputs/wqh/paddlevideo/steps/open_data")
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--pos-stride", type=int, default=2)
    ap.add_argument("--preset", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--pos-repeat", type=int, default=1,
                    help="duplicate positive rows N times in train.list (class balance)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = json.load(open(args.meta, encoding="utf-8"))
    if isinstance(meta, dict) and "clips" in meta:
        meta = meta["clips"]
    entries = list(meta.values()) if isinstance(meta, dict) else meta
    by_name = {e["name"]: e for e in entries}

    # build spec list: (name, label, stride, max_windows)
    spec = [(POS_SEG, 1, args.pos_stride, None)]
    for seg, stride, mx in PRESETS[args.preset]["neg"]:
        spec.append((seg, 0, stride, mx))

    counts = {0: 0, 1: 0}
    meta_out = []
    rows = []
    print(f"[preset {args.preset}] win={args.win}")
    for name, label, stride, mx in spec:
        e = by_name.get(name)
        if e is None:
            print(f"  !! missing segment {name} in meta"); continue
        n = int(e["n_frames"])
        if n < args.win:
            print(f"  [skip] {name}: n={n} < win={args.win}"); continue
        wins = [(s, s + args.win - 1) for s in range(1, n - args.win + 2, stride)]
        if mx is not None:
            wins = even_spaced(wins, mx)
        counts[label] += len(wins)
        for k, (a, b) in enumerate(wins, 1):
            wname = f"{'pos' if label else 'neg'}_{name}_w{k:02d}"
            rows.append((wname, args.win, label))
            meta_out.append(dict(win=wname, label=label, kind="pos" if label else "neg",
                                 src_dir=name, src_frames=[a, b], src_n=len(wins)))
        print(f"  {name:<28} label={label} n_frames={n:<4} -> {len(wins):<3} windows"
              + (f"  (kept {len(wins)}/{len(wins) if mx is None else 'all'})" if False else ""))

    ratio = counts[0] / counts[1] if counts[1] else float("inf")
    eff_pos = counts[1] * args.pos_repeat
    print(f"\n[summary] positives(open)={counts[1]} (x{args.pos_repeat} repeat -> {eff_pos} rows/epoch)"
          f"  negatives(not-open)={counts[0]}  "
          f"base ratio 1:{ratio:.2f}  eff ratio 1:{counts[0]/eff_pos:.2f}  "
          f"total dirs={sum(counts.values())} rows/epoch={eff_pos + counts[0]}")

    if args.dry_run:
        print("[dry-run] nothing written")
        return

    os.makedirs(args.out_root, exist_ok=True)
    n_copied = 0
    for wname, nf, label in rows:
        wdir = osp.join(args.out_root, wname)
        if osp.exists(wdir):
            shutil.rmtree(wdir)
        os.makedirs(wdir)
        ent = next(x for x in meta_out if x["win"] == wname)
        a, b = ent["src_frames"]
        for j in range(a, b + 1):
            src = osp.join(args.src_root, ent["src_dir"], f"img_{j:05d}.jpg")
            dst = osp.join(wdir, f"img_{j - a + 1:05d}.jpg")
            shutil.copy2(src, dst)
            n_copied += 1

    list_lines = []
    for wname, nf, label in rows:
        list_lines.append(f"{wname} {nf} {label}")
        if label == 1:
            for _ in range(args.pos_repeat - 1):
                list_lines.append(f"{wname} {nf} {label}")
    with open(osp.join(args.out_root, "train.list"), "w", encoding="utf-8") as f:
        f.write("\n".join(list_lines) + "\n")
    with open(osp.join(args.out_root, "window_meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(preset=args.preset, win=args.win, samples=meta_out),
                  f, ensure_ascii=False, indent=1)

    # segment-level eval list (labels: 1=open, 0=not-open) -> data_prefix steps/data
    eval_rows = []
    for name, label in EVAL_SEGS:
        e = by_name[name]
        eval_rows.append(f"{name} {int(e['n_frames'])} {label}")
    with open(osp.join(args.out_root, "open_eval.list"), "w", encoding="utf-8") as f:
        f.write("\n".join(eval_rows) + "\n")

    print(f"[done] wrote {len(rows)} sample dirs ({n_copied} frame copies) -> {args.out_root}")
    print("[list] open_eval.list:")
    for ln in eval_rows:
        print("   ", ln)


if __name__ == "__main__":
    main()
