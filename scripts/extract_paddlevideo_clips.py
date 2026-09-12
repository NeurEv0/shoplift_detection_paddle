"""Extract PaddleVideo PP-TSM training clips from RAW source videos (whole-frame, no MOT).

Input : CSV with columns: person_id,video,start_sec,end_sec,label,split,note
        video: one of "cc015" | "D03" | "dcsass"
        label: "steal" | "normal"
        split: "train" | "test"
Output: FrameDataset dir + train.list / test.list (PaddleVideo format)
        每行: <clip_dir_rel> <label_id>   (0=normal, 1=steal)

Usage:
    python extract_paddlevideo_clips.py --csv clips.csv --out data_root [--frames 8] [--max-clips 3]
"""
import argparse
import csv
import os

import cv2
import numpy as np

VIDEOS = {
    "cc015": r"C:\User\WQH\code\shoplift_detection_paddle_server\shoplift_detection_paddle\datasets\test\test_videos\yulong_store\cc015bb3e050a8bc67ebbb2e6fd9951d.mp4",
    "D03": r"C:\User\WQH\code\shoplift_detection_paddle_server\shoplift_detection_paddle\datasets\test\test_videos\yulong_store\D03_堂食区_修复版.mpg",
    "dcsass": r"C:\User\WQH\code\shoplift_detection_paddle_server\shoplift_detection_paddle\datasets\test\test_videos\DCSASS\Shoplifting010_x264_merged.mp4",
}
LABEL_ID = {"normal": 0, "steal": 1}


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    return cap, fps


def read_frames_in_range(cap, fps, start_sec, end_sec):
    """Return list of whole frames within [start_sec, end_sec] via SEQUENTIAL read (mpg-safe)."""
    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))
    frames = []
    cur = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if cur >= start_frame and cur <= end_frame:
            frames.append(frame)
        elif cur > end_frame:
            break
        cur += 1
    return frames


def sample_frames(frames, n):
    """Uniformly sample n frames (whole-frame) from list; returns list of ndarray."""
    if len(frames) <= n:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="data root for FrameDataset")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-clips", type=int, default=3, help="max sliding clips per time window")
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    os.makedirs(args.out, exist_ok=True)
    train_lines, test_lines = [], []
    caps = {}  # video -> (cap, fps)
    clip_no = 0

    for r in rows:
        vid = r["video"].strip()
        label = r["label"].strip()
        split = r["split"].strip()
        pid = r["person_id"].strip()
        start, end = float(r["start_sec"]), float(r["end_sec"])
        if vid not in VIDEOS:
            print(f"!! unknown video {vid}, skip {pid}")
            continue
        if vid not in caps:
            caps[vid] = open_video(VIDEOS[vid])
        cap, fps = caps[vid]

        frames = read_frames_in_range(cap, fps, start, end)
        if len(frames) < args.frames:
            print(f"!! {pid}: only {len(frames)} frames in [{start},{end}]s, need {args.frames}, skip")
            continue

        # sliding windows: split window into up to max_clips overlapping chunks
        n_clips = min(args.max_clips, max(1, len(frames) // args.frames))
        for k in range(n_clips):
            # choose a sub-window then uniformly sample frames
            lo = int(round(k * (len(frames) - args.frames) / max(1, n_clips - 1)))
            sub = frames[lo: lo + args.frames] if len(frames) - lo >= args.frames else frames[-args.frames:]
            sampled = sample_frames(sub, args.frames)
            clip_dir = os.path.join(label, f"{pid}_{split}_{k:02d}")
            full = os.path.join(args.out, clip_dir)
            os.makedirs(full, exist_ok=True)
            for i, f in enumerate(sampled):
                cv2.imwrite(os.path.join(full, f"img_{i+1:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 90])
            lid = LABEL_ID[label]
            line = f"{clip_dir} {lid}"
            if split == "test":
                test_lines.append(line)
            else:
                train_lines.append(line)
            clip_no += 1

    for cap, _ in caps.values():
        cap.release()

    with open(os.path.join(args.out, "train.list"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(train_lines) + ("\n" if train_lines else ""))
    with open(os.path.join(args.out, "test.list"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(test_lines) + ("\n" if test_lines else ""))
    print(f"done: {clip_no} clips | train={len(train_lines)} test={len(test_lines)} -> {args.out}")


if __name__ == "__main__":
    main()
