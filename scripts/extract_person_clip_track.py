"""Method A: extract ONE person's continuous frames from a RAW video via single-object tracking.

Workflow:
  1. Open raw video, seek to start_sec.
  2. Show one frame; user draws a bounding box around the target person (mouse).
  3. CSRT tracker follows the person frame-by-frame until end_sec.
  4. Crop each frame by the tracked bbox (+padding) -> person continuous-frame clip.
  5. Optionally write a debug video (tracking box overlay) for quality check.

Input can be given via CSV (person_id,video,start_sec,end_sec,label,split,note) or CLI args.

Usage (CSV, all rows processed; skip rows already present in output dir):
    python extract_person_clip_track.py --csv clips.csv --out data_root [--pad 0.15] [--debug]

Usage (single clip):
    python extract_person_clip_track.py --video cc015 --person thief_A --label steal --split train \
        --start 12 --end 40 --out data_root [--pad 0.15] [--debug]

Output: data_root/<label>/clip_<person>_<split>_<k>/img_00001.jpg ... + appends to train/test.list
"""
import argparse
import csv
import os

import cv2

VIDEOS = {
    "cc015": r"C:\User\WQH\code\shoplift_detection_paddle_server\shoplift_detection_paddle\datasets\test\test_videos\yulong_store\cc015bb3e050a8bc67ebbb2e6fd9951d.mp4",
    "D03": r"C:\User\WQH\code\shoplift_detection_paddle_server\shoplift_detection_paddle\datasets\test\test_videos\yulong_store\D03_堂食区_修复版.mpg",
    "dcsass": r"C:\User\WQH\code\shoplift_detection_paddle_server\shoplift_detection_paddle\datasets\test\test_videos\DCSASS\Shoplifting010_x264_merged.mp4",
}
LABEL_ID = {"normal": 0, "steal": 1}


def make_tracker():
    # CSRT is robust; fall back to KCF
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        return cv2.TrackerKCF_create()


def seek_to_second(cap, fps, target_sec):
    """Sequential-read up to target frame (mpg-safe: no POS_FRAMES seek on mpg)."""
    target = int(round(target_sec * fps))
    cur = 0
    frame = None
    while cur <= target:
        ok, frame = cap.read()
        if not ok:
            return None
        cur += 1
    return frame


def read_frames_in_range(cap, start_frame, end_frame):
    """Read frames [start_frame, end_frame] sequentially, return list of (idx, frame)."""
    out = []
    cur = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if cur >= start_frame and cur <= end_frame:
            out.append((cur, frame))
        elif cur > end_frame:
            break
        cur += 1
    return out


def crop_padded(frame, bbox, pad):
    h, w = frame.shape[:2]
    x, y, bw, bh = [int(v) for v in bbox]
    x1, y1 = max(0, int(x - bw * pad)), max(0, int(y - bh * pad))
    x2, y2 = min(w - 1, int(x + bw + bw * pad)), min(h - 1, int(y + bh + bh * pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def process_clip(args, person_id, video, start_sec, end_sec, label, split):
    cap = cv2.VideoCapture(VIDEOS[video])
    if not cap.isOpened():
        print(f"!! cannot open {video}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))

    init_frame = seek_to_second(cap, fps, start_sec)
    if init_frame is None:
        print(f"!! {person_id}: cannot seek to {start_sec}s")
        cap.release()
        return None

    # user selects the person on the first frame
    cv2.namedWindow("select person - drag box, press ENTER", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("select person - drag box, press ENTER", init_frame, showCrosshair=True)
    cv2.destroyAllWindows()
    if roi[2] <= 0 or roi[3] <= 0:
        print(f"!! {person_id}: no box selected, skip")
        cap.release()
        return None

    tracker = make_tracker()
    tracker.init(init_frame, roi)

    # read frames sequentially from start (we already consumed start frame)
    # rewind by reopening (simpler than tracking cap position)
    cap.release()
    cap = cv2.VideoCapture(VIDEOS[video])
    frames = read_frames_in_range(cap, start_frame, end_frame)
    cap.release()

    clip_dir = f"clip_{person_id}_{split}"
    out_full = os.path.join(args.out, label, clip_dir)
    os.makedirs(out_full, exist_ok=True)
    n = 0
    writer = None
    for idx, frame in frames:
        if idx == start_frame:
            ok, bbox = True, roi
        else:
            ok, bbox = tracker.update(frame)
        if not ok:
            continue
        crop = crop_padded(frame, bbox, args.pad)
        if crop is None:
            continue
        n += 1
        cv2.imwrite(os.path.join(out_full, f"img_{n:05d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if args.debug and writer is None:
            fps_out = fps
            writer = cv2.VideoWriter(os.path.join(out_full, "_tracking_debug.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (frame.shape[1], frame.shape[0]))
        if writer is not None:
            disp = frame.copy()
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(disp, person_id, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            writer.write(disp)
    if writer is not None:
        writer.release()
    if n == 0:
        print(f"!! {person_id}: no frames cropped")
        return None
    lid = LABEL_ID[label]
    line = f"{label}/{clip_dir} {lid}"
    list_path = os.path.join(args.out, "test.list" if split == "test" else "train.list")
    with open(list_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(f"ok {person_id}: {n} frames -> {os.path.join(args.out, label, clip_dir)}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV: person_id,video,start_sec,end_sec,label,split,note")
    ap.add_argument("--video", choices=list(VIDEOS))
    ap.add_argument("--person", help="person id")
    ap.add_argument("--label", choices=["steal", "normal"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=float, default=0.15, help="bbox padding ratio")
    ap.add_argument("--debug", action="store_true", help="write tracking debug video")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.csv:
        with open(args.csv, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if not (r["start_sec"] and r["end_sec"]):
                    continue
                process_clip(args, r["person_id"], r["video"], float(r["start_sec"]),
                             float(r["end_sec"]), r["label"], r["split"])
    else:
        need = [args.video, args.person, args.label, args.start, args.end]
        if any(v is None for v in need):
            ap.error("provide --csv OR all of --video/--person/--label/--start/--end")
        process_clip(args, args.person, args.video, args.start, args.end, args.label, args.split)


if __name__ == "__main__":
    main()
