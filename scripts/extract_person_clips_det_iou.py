#!/usr/bin/env python3
"""
Extract per-person continuous frame clips via detection + IOU association (v3).

WHY v3 (vs extract_person_clips_det_iou.py v2):
  v2 fixed the false-positive flood (person_mot returns 100 unfiltered candidates/frame)
  with confidence+size filtering, but still occasionally swaps IDs when two people
  cross or stand close: a detection box can IOU-match the WRONG track. v3 hardens the
  association with three simultaneous constraints:
    - IOU vs the track's median-position box (not just last box)
    - size ratio: detection size vs the track's median size must be within a band
    - center distance: detection center vs track's predicted center must be within a
      multiple of the track's median height
  A detection must satisfy ALL constraints to join a track; otherwise it starts a new
  track. This makes swaps much less likely when people cross.

Pipeline (run per video separately — never mix videos):
  1. Sequential video decode (handles mpg seek corruption).
  2. Person detection with person_mot model on EVERY processed frame at stride
     DET_STRIDE, filtered by confidence / min height / aspect ratio.
  3. Constrained greedy IOU association to active tracks (see above).
  4. Output: one clip directory per track under
       <out_root>/<video_key>/person-XXX/img_%05d.jpg
     plus track_meta.json (frames, per-frame bbox, scores) for human labeling.

Usage:
  python extract_person_clips_det_iou2.py \
      --video <path> --out-root <out_root> --video-key <key> \
      [--det-stride 3] [--det-thresh 0.6] [--min-person-h 150] \
      [--iou-thresh 0.3] [--size-ratio 0.6] [--center-ratio 1.0] \
      [--track-buffer 5] [--min-track-frames 8] [--max-frames N]

Output layout:
  <out_root>/<video_key>/
    person-XXX/img_%05d.jpg        # cropped person frames
    track_meta.json                # per-track metadata (labeling aid)
    det_stats.json                 # detection + association statistics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# PaddleDetection pure detector wrapper
# ---------------------------------------------------------------------------

def _prepare_paddledet_paths(paddledetection_root: Path) -> None:
    root = Path(paddledetection_root).resolve()
    paths = [
        root,
        root / "deploy",
        root / "deploy/pipeline",
        root / "deploy/python",
        root / "deploy/pptracking/python",
        root / "deploy/pptracking",
    ]
    for p in reversed(paths):
        text = str(p)
        if text not in sys.path:
            sys.path.insert(0, text)


class PureDetector:
    """Wraps PaddleDetection Detector (no global MOT tracker) for in-memory frames."""

    def __init__(self, model_dir: str, device: str, threshold: float):
        import paddle
        from det_infer import Detector

        paddle.enable_static()
        self._det = Detector(
            model_dir=model_dir,
            device=device.upper(),
            run_mode="paddle",
            batch_size=1,
            output_dir="/tmp/paddledet_out",
            threshold=threshold,
        )
        self._tmp_path = "/tmp/paddlevideo_det_frame.jpg"

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        import cv2

        cv2.imwrite(self._tmp_path, frame_bgr)
        try:
            inputs = self._det.preprocess([self._tmp_path])
            result = self._det.postprocess(inputs, self._det.predict())
        finally:
            try:
                Path(self._tmp_path).unlink()
            except OSError:
                pass
        boxes = result.get("boxes", np.zeros((0, 6), dtype=np.float32))
        out = []
        for row in boxes:
            cls_id, score, x0, y0, x1, y1 = [float(v) for v in row]
            out.append({"bbox": [x0, y0, x1, y1], "score": score})
        return out


# ---------------------------------------------------------------------------
# Video: sequential decode (safe for mpg)
# ---------------------------------------------------------------------------

def iter_video_frames(video_path: Path, max_frames: int | None, stride: int):
    """Yield (source_frame_id, bgr_frame). stride>1 skips frames in place."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_id >= max_frames:
                break
            if frame_id % stride == 0:
                yield frame_id, frame, fps, total
            frame_id += 1
    finally:
        cap.release()


def iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


# ---------------------------------------------------------------------------
# Track (pure detection box history, no tracker prediction)
# ---------------------------------------------------------------------------

class PersonTrack:
    __slots__ = (
        "person_id", "bbox", "frame_boxes", "scores",
        "missed", "born_frame", "last_frame", "active",
        "_hist", "median_w", "median_h",
    )

    def __init__(self, person_id: int, bbox, score: float, frame_id: int):
        self.person_id = person_id
        self.bbox = [float(v) for v in bbox]
        self.frame_boxes: dict[int, list[float]] = {frame_id: list(self.bbox)}
        self.scores: dict[int, float] = {frame_id: score}
        self.missed = 0
        self.born_frame = frame_id
        self.last_frame = frame_id
        self.active = True
        self._hist: list[list[float]] = [list(self.bbox)]
        self.median_w = self.bbox[2] - self.bbox[0]
        self.median_h = self.bbox[3] - self.bbox[1]

    def append(self, bbox, score: float, frame_id: int):
        self.bbox = [float(v) for v in bbox]
        self.frame_boxes[frame_id] = list(self.bbox)
        self.scores[frame_id] = score
        self.missed = 0
        self.last_frame = frame_id
        # keep a bounded history for robust median size / median-position estimates
        self._hist.append(list(self.bbox))
        if len(self._hist) > 40:
            self._hist.pop(0)
        ws = [b[2] - b[0] for b in self._hist]
        hs = [b[3] - b[1] for b in self._hist]
        self.median_w = float(np.median(ws))
        self.median_h = float(np.median(hs))

    def median_box(self):
        """Box anchored at the median size and the median center of the history."""
        cxs = [(b[0] + b[2]) / 2.0 for b in self._hist]
        cys = [(b[1] + b[3]) / 2.0 for b in self._hist]
        cx, cy = float(np.median(cxs)), float(np.median(cys))
        w, h = self.median_w, self.median_h
        return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="Input video path.")
    p.add_argument("--out-root", required=True, help="Output root directory.")
    p.add_argument("--video-key", required=True, help="Short key for this video (e.g. cc015).")
    p.add_argument("--paddledetection-root", default="src/PaddleDetection-release-2.9")
    p.add_argument("--model-dir", default="models/paddledetection/person_mot")
    p.add_argument("--device", default="gpu")
    p.add_argument("--det-stride", type=int, default=3,
                   help="Process every Nth source frame (frame stride).")
    p.add_argument("--det-thresh", type=float, default=0.6,
                   help="Detection confidence threshold. person_mot returns unfiltered "
                        "candidates; real people score ~0.9, false positives ~0.4-0.8.")
    p.add_argument("--min-person-h", type=float, default=150.0,
                   help="Ignore detections with height < min_person_h px.")
    p.add_argument("--iou-thresh", type=float, default=0.3,
                   help="IOU threshold vs track's median box to associate.")
    p.add_argument("--size-ratio", type=float, default=0.6,
                   help="Detection height must be within [1-size_ratio, 1+size_ratio] "
                        "of the track's median height (0.6 => 0.4x..1.6x).")
    p.add_argument("--center-ratio", type=float, default=1.0,
                   help="Detection center must lie within center_ratio * median_h "
                        "of the track's median center.")
    p.add_argument("--track-buffer", type=int, default=5,
                   help="Terminate a track after this many consecutive processed frames "
                        "without a matched detection.")
    p.add_argument("--min-track-frames", type=int, default=8,
                   help="Drop tracks shorter than this (frames).")
    p.add_argument("--max-frames", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    video_path = Path(args.video)
    out_root = Path(args.out_root)
    video_out = out_root / args.video_key
    video_out.mkdir(parents=True, exist_ok=True)

    _prepare_paddledet_paths(Path(args.paddledetection_root))
    print(f"[init] detector: {args.model_dir} device={args.device} "
          f"stride={args.det_stride} iou={args.iou_thresh} "
          f"size_ratio={args.size_ratio} center_ratio={args.center_ratio}", flush=True)
    detector = PureDetector(args.model_dir, args.device, args.det_thresh)

    tracks: list[PersonTrack] = []
    next_person_id = 1
    det_stats = {"detection_frames": 0, "boxes_total": 0, "frames_total": 0,
                 "tracks_created": 0, "tracks_terminated": 0,
                 "boxes_filtered": 0, "assoc_ok": 0, "assoc_rejected": 0}
    t0 = time.time()

    for frame_id, frame_bgr, fps, total in iter_video_frames(video_path, args.max_frames, args.det_stride):
        det_stats["frames_total"] += 1
        det_stats["detection_frames"] += 1

        all_dets = detector.detect(frame_bgr)
        det_stats["boxes_total"] += len(all_dets)
        dets = []
        for d in all_dets:
            x0, y0, x1, y1 = d["bbox"]
            h = y1 - y0
            w = x1 - x0
            if (h >= args.min_person_h and w >= 0.25 * h
                    and d["score"] >= args.det_thresh):
                dets.append(d)
            else:
                det_stats["boxes_filtered"] += 1
        dets.sort(key=lambda d: d["score"], reverse=True)

        matched_tracks = set()
        # greedy association with triple constraint
        for d in dets:
            best_t, best_score = None, 0.0
            for t in tracks:
                if not t.active or t.person_id in matched_tracks:
                    continue
                mbox = t.median_box()
                v = iou(d["bbox"], mbox)
                if v < args.iou_thresh:
                    continue
                # size constraint: detection height vs track median height
                d_h = d["bbox"][3] - d["bbox"][1]
                ratio = d_h / t.median_h if t.median_h > 1 else 1.0
                if not (1.0 - args.size_ratio <= ratio <= 1.0 + args.size_ratio):
                    continue
                # center constraint
                dcx, dcy = box_center(d["bbox"])
                mcx, mcy = box_center(mbox)
                dist = ((dcx - mcx) ** 2 + (dcy - mcy) ** 2) ** 0.5
                if dist > args.center_ratio * max(t.median_h, 10.0):
                    continue
                if v > best_score:
                    best_t, best_score = t, v
            if best_t is not None:
                best_t.append(d["bbox"], d["score"], frame_id)
                matched_tracks.add(best_t.person_id)
                det_stats["assoc_ok"] += 1
            else:
                t = PersonTrack(next_person_id, d["bbox"], d["score"], frame_id)
                tracks.append(t)
                next_person_id += 1
                det_stats["tracks_created"] += 1
                det_stats["assoc_rejected"] += 1

        for t in tracks:
            if not t.active or t.person_id in matched_tracks:
                continue
            t.missed += 1
            if t.missed >= args.track_buffer:
                t.active = False
                det_stats["tracks_terminated"] += 1

        if frame_id % 600 == 0:
            print(f"[progress] frame {frame_id}/{total} active="
                  f"{sum(1 for t in tracks if t.active)} created={det_stats['tracks_created']} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    # ---- finalize: write per-track clips using per-frame bboxes ----
    print("[finalize] writing clips...", flush=True)
    meta = {}
    kept = [t for t in tracks if len(t.frame_boxes) >= args.min_track_frames]
    for t in kept:
        pid_dir = video_out / f"person-{t.person_id:03d}"
        pid_dir.mkdir(parents=True, exist_ok=True)
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fid = 0
        n_written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            box = t.frame_boxes.get(fid)
            if box is not None:
                x0, y0, x1, y1 = box
                h, w = frame.shape[:2]
                x0c, y0c = max(0, int(x0)), max(0, int(y0))
                x1c, y1c = min(w, int(x1)), min(h, int(y1))
                crop = frame[y0c:y1c, x0c:x1c]
                if crop.size > 0:
                    out_path = pid_dir / f"img_{n_written + 1:05d}.jpg"
                    cv2.imwrite(str(out_path), crop)
                    n_written += 1
            fid += 1
        cap.release()
        frames_sorted = sorted(t.frame_boxes)
        meta[t.person_id] = {
            "person_id": t.person_id,
            "clip_dir": str(pid_dir.relative_to(video_out)),
            "n_frames": len(frames_sorted),
            "born_frame": t.born_frame,
            "last_frame": t.last_frame,
            "frames": frames_sorted,
            "boxes": [t.frame_boxes[f] for f in frames_sorted],
            "scores": [t.scores[f] for f in frames_sorted],
        }
        print(f"[clip] person-{t.person_id:03d}: n={len(frames_sorted)} "
              f"frames {frames_sorted[0]}..{frames_sorted[-1]}", flush=True)

    (video_out / "track_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (video_out / "det_stats.json").write_text(
        json.dumps(det_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] clips under {video_out} tracks_kept={len(meta)} "
          f"elapsed={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
