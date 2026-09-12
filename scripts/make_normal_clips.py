#!/usr/bin/env python3
"""
Build normal (non-steal) clips from v3 track results.

Each normal person = 1 clip of N frames (default 48), uniformly sampled from the
union of their tracks' frames. Optional --window (start_s,end_s) restricts the
time window (used for super-long tracks like P4 and P517).

Output:
  <out_root>/
    normal_<key>/img_%05d.jpg
    normal_clips_meta.json

Usage:
  python make_normal_clips.py \
      --meta-cc015 <cc015 track_meta.json> --meta-d03 <d03 track_meta.json> \
      --video-cc015 <cc015 video> --video-d03 <d03 video> \
      --out-root <out_root> [--frames 48] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# key -> (video_key, tracks, window_s or None)
NORMALS = {
    "train1_cc015": ("cc015", ["30", "40", "56"], None),
    "train2_cc015": ("cc015", ["39", "42", "52"], None),
    "train3_d03": ("d03", ["4"], (0.0, 10.0)),
    "valid1_d03": ("d03", ["8", "28", "63", "270"], None),
    "test1_d03": ("d03", ["517"], (555.8, 566.0)),
    "test2_d03": ("d03", ["591", "595", "597"], None),
}


def load_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_tracks(meta: dict, pids: list[str], window_s) -> dict[int, dict]:
    """Union of frames across tracks within optional window. frame_id -> {box, score}."""
    merged: dict[int, dict] = {}
    lo = int(round(window_s[0] * 25.0)) if window_s else None
    hi = int(round(window_s[1] * 25.0)) if window_s else None
    for pid in pids:
        m = meta.get(pid)
        if m is None:
            print(f"[warn] track P{pid} not found, skipped", file=sys.stderr)
            continue
        for f, box, score in zip(m["frames"], m["boxes"], m["scores"]):
            if lo is not None and f < lo:
                continue
            if hi is not None and f > hi:
                continue
            cur = merged.get(f)
            if cur is None or score > cur["score"]:
                merged[f] = {"box": box, "score": score}
    return merged


def plan_sample(merged: dict[int, dict], target: int) -> list[int]:
    ids = sorted(merged.keys())
    if len(ids) <= target:
        return ids
    idx = np.linspace(0, len(ids) - 1, target).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [ids[i] for i in idx]


def write_clip(video: Path, clip_dir: Path, merged: dict[int, dict],
               sampled: list[int], fps: float) -> dict:
    import cv2

    clip_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    want = set(sampled)
    fid = 0
    n = 0
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fid in want:
            box = merged[fid]["box"]
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = box
            x0c, y0c = max(0, int(x0)), max(0, int(y0))
            x1c, y1c = min(w, int(x1)), min(h, int(y1))
            crop = frame[y0c:y1c, x0c:x1c]
            if crop.size > 0:
                out = clip_dir / f"img_{n + 1:05d}.jpg"
                cv2.imwrite(str(out), crop)
                n += 1
                rows.append({"index": n, "src_frame": fid,
                             "time_s": round(fid / fps, 2), "box": box})
        fid += 1
    cap.release()
    return {"clip_dir": str(clip_dir.name), "n_frames": n,
            "src_frames": sampled, "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-cc015", required=True)
    p.add_argument("--meta-d03", required=True)
    p.add_argument("--video-cc015", required=True)
    p.add_argument("--video-d03", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--frames", type=int, default=48)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    metas = {
        "cc015": load_meta(Path(args.meta_cc015)),
        "d03": load_meta(Path(args.meta_d03)),
    }
    videos = {
        "cc015": Path(args.video_cc015),
        "d03": Path(args.video_d03),
    }
    fps = 25.0
    out_root = Path(args.out_root)

    result = {}
    for key, (vkey, pids, window) in NORMALS.items():
        merged = merge_tracks(metas[vkey], pids, window)
        sampled = plan_sample(merged, args.frames)
        span = (sampled[0] / fps, sampled[-1] / fps) if sampled else (0, 0)
        print(f"[plan] normal_{key}: merged={len(merged)} sampled={len(sampled)} "
              f"span {span[0]:.1f}s..{span[1]:.1f}s window={window}")
        if args.dry_run:
            result[key] = {"merged": len(merged), "sampled": len(sampled),
                           "span_s": [round(span[0], 2), round(span[1], 2)],
                           "frames": sampled}
            continue
        clip_dir = out_root / f"normal_{key}"
        info = write_clip(videos[vkey], clip_dir, merged, sampled, fps)
        info["tracks"] = pids
        info["window_s"] = window
        result[key] = info
        print(f"[done] normal_{key}: wrote {info['n_frames']} frames -> {clip_dir}")

    if not args.dry_run:
        (out_root / "normal_clips_meta.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
