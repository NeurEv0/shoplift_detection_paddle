"""Method B: extract ONE person's continuous frames via manual keyframe labeling + linear interpolation.

Workflow:
  1. Open raw video; every KEYFRAME_INTERVAL frames within [start_sec, end_sec], show the frame
     and let the user drag a box around the target person.
  2. Linear-interpolate bbox between keyframes for ALL frames in the window.
  3. Crop every frame by interpolated bbox (+padding) -> person continuous-frame clip.

Manual labeling is slow but fully controllable (no tracker drift). Use --every to control density.

Usage:
    python extract_person_clip_manual.py --video cc015 --person thief_A --label steal --split train \
        --start 12 --end 40 --out data_root [--every 25] [--pad 0.15]

Output: data_root/<label>/clip_<person>_<split>/img_00001.jpg ... + appends to train/test.list
"""
import argparse
import os

import cv2
import numpy as np

VIDEOS = {
    "cc015": r"C:\User\WQH\code\shoplift_detection_paddle_server\shoplift_detection_paddle\datasets\test\test_videos\yulong_store\cc015bb3e050a8bc67ebbb2e6fd9951d.mp4",
    "D03": r"C:\User\WQH\code\shoplift_detection_paddle_server\shoplift_detection_paddle\datasets\test\test_videos\yulong_store\D03_堂食区_修复版.mpg",
    "dcsass": r"C:\User\WQH\code\shoplift_detection_paddle_server\shoplift_detection_paddle\datasets\test\test_videos\DCSASS\Shoplifting010_x264_merged.mp4",
}
LABEL_ID = {"normal": 0, "steal": 1}


def read_range(path, start_frame, end_frame):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    cur = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if cur >= start_frame and cur <= end_frame:
            frames.append((cur, frame))
        elif cur > end_frame:
            break
        cur += 1
    cap.release()
    return frames, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, choices=list(VIDEOS))
    ap.add_argument("--person", required=True)
    ap.add_argument("--label", required=True, choices=["steal", "normal"])
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--start", required=True, type=float)
    ap.add_argument("--end", required=True, type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=25, help="label every N frames (~1s at 25fps)")
    ap.add_argument("--pad", type=float, default=0.15)
    args = ap.parse_args()

    fps = 25.0
    start_frame = int(round(args.start * fps))
    end_frame = int(round(args.end * fps))
    frames, fps = read_range(VIDEOS[args.video], start_frame, end_frame)
    if not frames:
        print("no frames read")
        return

    # collect keyframes
    key_idx = []
    key_boxes = []
    for i, (idx, frame) in enumerate(frames):
        if (i % args.every == 0) or i == len(frames) - 1:
            win = "label person (drag box, ENTER; press c to skip)"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            roi = cv2.selectROI(win, frame, showCrosshair=True)
            cv2.destroyAllWindows()
            if roi[2] > 0 and roi[3] > 0:
                key_idx.append(i)
                key_boxes.append([roi[0], roi[1], roi[2], roi[3]])
                print(f"  keyframe {i}/{len(frames)-1} (frame {idx}): {roi}")
            else:
                print(f"  keyframe {i}: skipped")

    if len(key_boxes) < 2:
        print("need >=2 labeled keyframes; abort")
        return

    # interpolate for all frame indices
    key_idx = np.array(key_idx)
    key_boxes = np.array(key_boxes, dtype=np.float64)
    all_idx = np.arange(len(frames))
    interp = np.zeros((len(frames), 4))
    for c in range(4):
        interp[:, c] = np.interp(all_idx, key_idx, key_boxes[:, c])

    clip_dir = f"clip_{args.person}_{args.split}"
    out_full = os.path.join(args.out, args.label, clip_dir)
    os.makedirs(out_full, exist_ok=True)
    n = 0
    for i, (idx, frame) in enumerate(frames):
        x, y, w, h = interp[i]
        hh, ww = frame.shape[:2]
        x1 = max(0, int(x - w * args.pad)); y1 = max(0, int(y - h * args.pad))
        x2 = min(ww - 1, int(x + w + w * args.pad)); y2 = min(hh - 1, int(y + h + h * args.pad))
        if x2 <= x1 or y2 <= y1:
            continue
        n += 1
        cv2.imwrite(os.path.join(out_full, f"img_{n:05d}.jpg"), frame[y1:y2, x1:x2],
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    lid = LABEL_ID[args.label]
    list_path = os.path.join(args.out, "test.list" if args.split == "test" else "train.list")
    with open(list_path, "a", encoding="utf-8") as fh:
        fh.write(f"{args.label}/{clip_dir} {lid}\n")
    print(f"ok {args.person}: {n} frames -> {out_full}")


if __name__ == "__main__":
    main()
