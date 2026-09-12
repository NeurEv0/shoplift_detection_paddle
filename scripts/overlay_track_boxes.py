#!/usr/bin/env python3
"""
READ-ONLY visual checker: render ORIGINAL video frames over a src range and
overlay the boxes of given person tracks from track_meta, so you can see who
was tracked where (and who was NOT tracked) during a tracking hole.

Typical use: steal_2_d03's holes (src ~1939..2294). Render those seconds and
check whether the thief matches one of the drawn boxes (then just add that
person id to the merge list!) or appears WITHOUT any box (he was not tracked
there -> windowed detection needed, with better picking).

Usage (sequential mpg read, nothing is modified):
  python overlay_track_boxes.py \
      --video "$(ls datasets/test/test_videos/yulong_store/D03*.mpg)" \
      --track-meta outputs/wqh/paddlevideo/clips_iou/d03/track_meta.json \
      --tracks 4,69,79,80,81,86,91,93,95,108,112 \
      --src-start 1935 --src-end 2300 --step 10 \
      --out-dir outputs/wqh/paddlevideo/steps/debug/steal2_hole_overlay
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import cv2


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--track-meta", required=True)
    p.add_argument("--tracks", required=True, help="csv of person ids to draw")
    p.add_argument("--src-start", type=int, required=True)
    p.add_argument("--src-end", type=int, required=True)
    p.add_argument("--step", type=int, default=10, help="render every Nth src frame")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    tm = load_json(Path(args.track_meta))
    per: dict[str, dict[int, list[float]]] = {}
    for pid in args.tracks.split(","):
        pid = pid.strip()
        if not pid:
            continue
        m = tm.get(pid)
        if m is None:
            print(f"[warn] P{pid} not in track_meta, skipped")
            continue
        per[pid] = {int(f): [float(v) for v in b]
                    for f, b in zip(m["frames"], m["boxes"])}
    if not per:
        raise SystemExit("no tracks to draw")

    colors: dict[str, tuple[int, int, int]] = {}

    def color_for(pid: str):
        if pid not in colors:
            h = (int(pid) * 47 % 100) / 100.0
            rgb = colorsys.hsv_to_rgb(h, 0.8, 0.9)
            colors[pid] = tuple(int(c * 255) for c in rgb)
        return colors[pid]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    want = set(range(args.src_start, args.src_end + 1, args.step))
    n = 0
    fid = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fid > args.src_end:
                break
            if fid in want:
                for pid, d in per.items():
                    box = d.get(fid)
                    if box is None:
                        continue
                    x0, y0, x1, y1 = (int(round(v)) for v in box)
                    c = color_for(pid)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), c, 2)
                    label = f"P{pid} {x0},{y0} {x1},{y1}"
                    cv2.putText(frame, label, (x0, max(0, y0 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
                cv2.putText(frame, f"src {fid}  t={fid / src_fps:.2f}s",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 255), 2)
                p_out = out / f"f_{fid:05d}.jpg"
                cv2.imwrite(str(p_out), frame)
                n += 1
            fid += 1
    finally:
        cap.release()
    print(f"[done] {n} frames -> {out}  (step {args.step} @ ~{src_fps:.0f}fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
