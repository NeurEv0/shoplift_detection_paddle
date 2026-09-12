#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate PP-TSM open checkpoints directly on the open-binary dataset.

For each checkpoint report (threshold theta=0.6, matching production):
  - POS windows (steal_2 + steal_3, label=1):
        recall = #windows open_prob>=theta / total   (漏报 = 1-recall)
  - NEG windows (label=0, in-train negatives):
        false-positive rate = #windows open_prob>=theta / total   (误报)
  - held-out NEG clips (open_eval.list): eventize FP check
        a clip is a 误报 if it yields >=min_consecutive consecutive windows >=theta

Nothing is written; stdout only.  Read-only w.r.t. the dataset.

Usage (server, from project root):
  python scripts/eval_open_checkpoints.py \
      --weights outputs/wqh/paddlevideo/pptsm_open_v3_fight16/ppTSM_epoch_00026.pdparams \
                 outputs/wqh/paddlevideo/pptsm_open_v3_fight16/ppTSM_epoch_00120.pdparams
"""
from __future__ import annotations

import argparse
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from PIL import Image

THETA = 0.6
MIN_CONSECUTIVE = 3
WIN = 16
STRIDE = 2


def load_model(config, weights, paddlevideo_root):
    from shoplift.pptsm_open.infer import OpenModel
    return OpenModel.load(config, weights, device="gpu",
                          paddlevideo_root=paddlevideo_root)


def load_crops(dirpath, n):
    crops = []
    for i in range(1, n + 1):
        p = osp.join(dirpath, f"img_{i:05d}.jpg")
        crops.append(Image.open(p).convert("RGB"))
    return crops


def eval_window_dir(model, dirpath):
    return model.open_prob(load_crops(dirpath, WIN))


def clip_max_event(model, dirpath, n_frames):
    """Slide 16-frame windows over a clip; return (max_prob, is_fp_event)."""
    crops = load_crops(dirpath, n_frames)
    if len(crops) < WIN:
        return 0.0, False
    probs = []
    for start in range(0, len(crops) - WIN + 1, STRIDE):
        probs.append(model.open_prob(crops[start:start + WIN]))
    run = 0
    best_run = 0
    for p in probs:
        run = run + 1 if p >= THETA else 0
        best_run = max(best_run, run)
    return max(probs), best_run >= MIN_CONSECUTIVE


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="shoplift/configs/paddlevideo/pptsm_open_v3_fight16.yaml")
    ap.add_argument("--paddlevideo-root", default="third_party/PaddleVideo")
    ap.add_argument("--data-root", default="outputs/wqh/paddlevideo/steps/open_data_v3")
    ap.add_argument("--train-list", default="outputs/wqh/paddlevideo/steps/open_data_v3/train.list")
    ap.add_argument("--eval-list", default="outputs/wqh/paddlevideo/steps/open_data_v3/open_eval.list")
    ap.add_argument("--eval-clips-root", default="outputs/wqh/paddlevideo/steps/data")
    ap.add_argument("--weights", nargs="+", required=True)
    args = ap.parse_args()

    # parse train.list -> [(dir, label)]
    rows = []
    with open(args.train_list, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rows.append((parts[0], int(parts[2])))

    pos_dirs = sorted({d for d, lab in rows if lab == 1})
    neg_dirs = sorted({d for d, lab in rows if lab == 0})

    # parse open_eval.list -> [(clip, n_frames, label)]
    eval_rows = []
    with open(args.eval_list, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            eval_rows.append((parts[0], int(parts[1]), int(parts[2])))

    print(f"pos_windows={len(pos_dirs)} neg_windows={len(neg_dirs)} "
          f"heldout_clips={len(eval_rows)} theta={THETA} "
          f"min_consecutive={MIN_CONSECUTIVE}")

    header = f"{'checkpoint':<10} {'pos_recall':>10} {'pos<theta':>9} " \
             f"{'neg_fpr':>8} {'neg>=theta':>10} {'heldout_fp':>10}"
    print(header)
    print("-" * len(header))

    for weights in args.weights:
        model = load_model(args.config, weights, args.paddlevideo_root)

        # positives: recall (漏报)
        pos_hit = 0
        pos_below = []
        for d in pos_dirs:
            p = eval_window_dir(model, osp.join(args.data_root, d))
            if p >= THETA:
                pos_hit += 1
            else:
                pos_below.append((d, round(p, 3)))
        pos_recall = pos_hit / len(pos_dirs)

        # in-train negatives: false-positive rate (误报)
        neg_fp = []
        for d in neg_dirs:
            p = eval_window_dir(model, osp.join(args.data_root, d))
            if p >= THETA:
                neg_fp.append((d, round(p, 3)))
        neg_fpr = len(neg_fp) / len(neg_dirs)

        # held-out negatives: eventize FP check
        heldout_fp = []
        for clip, n_frames, _lab in eval_rows:
            maxp, is_fp = clip_max_event(
                model, osp.join(args.eval_clips_root, clip), n_frames)
            if is_fp:
                heldout_fp.append((clip, round(maxp, 3)))

        name = osp.basename(weights).replace("ppTSM_epoch_", "ep").replace(".pdparams", "")
        print(f"{name:<10} {pos_recall:>10.3f} {len(pos_below):>9} "
              f"{neg_fpr:>8.3f} {len(neg_fp):>10} {len(heldout_fp):>10}")

        # detail dumps (kept short)
        if pos_below:
            print(f"    [漏报 pos<theta] {len(pos_below)}: {pos_below[:6]}{'...' if len(pos_below)>6 else ''}")
        if neg_fp:
            print(f"    [误报 neg>=theta] {len(neg_fp)}: {neg_fp[:6]}{'...' if len(neg_fp)>6 else ''}")
        if heldout_fp:
            print(f"    [误报 heldout] {heldout_fp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
