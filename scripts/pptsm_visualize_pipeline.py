#!/usr/bin/env python3
"""
PP-TSM universal visualization with training-consistent detection + single-object tracking.

Implements the SAME detection + constrained-IoU single-object tracking used to build the
training clips (scripts/extract_person_clips_det_iou.py v3):
  - pure PaddleDetection person detector (person_mot model, NO global MOT tracker)
  - greedy association with triple constraint (IOU + size ratio + center distance)
    against each track's median box -> avoids ID swaps when people cross
  - per-track 16-frame sliding window -> PP-TSM -> P(normal)/P(steal)

Static-graph detection and dynamic-graph PP-TSM CANNOT share one process, so the
pipeline is split into subprocesses:
    detect          : pass 1, detection + tracking -> detections jsonl (static graph)
    full-video      : pass 2, ppTSM classify + annotate ENTIRE video (dynamic graph)
    person-clips    : pass 2, ppTSM classify + one stitched video PER PERSON
    run-full / run-clips : one-shot orchestrators (detect + annotate)

Two output modes:

1) full-video : annotate the ENTIRE video - every person's box + ID + steal/normal
                confidence on EVERY frame -> annotated mp4 + per-frame jsonl.

2) person-clips : for EVERY person track, crop the person region frame by frame,
                annotate box + confidence, stitch into ONE video per person
                -> out_dir/person-XXX.mp4 + person-XXX.jsonl.

Examples:
  # one-shot: entire video
  python pptsm_visualize_pipeline.py run-full \
      --video d03.mpg --config ppts_steal.yaml --weights ppTSM_epoch_00120.pdparams \
      --out full_annotated.mp4 --det-stride 3

  # one-shot: per-person clips
  python pptsm_visualize_pipeline.py run-clips \
      --video d03.mpg --config ppts_steal.yaml --weights ppTSM_epoch_00120.pdparams \
      --out-dir person_clips --det-stride 3

  # step by step (same results, inspect intermediates)
  python pptsm_visualize_pipeline.py detect --video d03.mpg --out det.jsonl --det-stride 3
  python pptsm_visualize_pipeline.py full-video --video d03.mpg --detections det.jsonl \
      --config ppts_steal.yaml --weights ppTSM_epoch_00120.pdparams --out full_annotated.mp4

Notes:
  - num_seg is read from the PP-TSM config sampler; must match backbone (16).
  - Detection uses the person_mot model directory as a PURE detector via
    det_infer.Detector (same as the training clip extractor).
  - --det-stride 3 means detection runs every 3rd frame; tracking carries boxes on
    intermediate frames so annotation is smooth at full framerate.
  - All paths below can be overridden via CLI (see _add_det_args).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/e-ai2/ZHITAI_2tb/shoplift_detection_paddle")
DEFAULT_MOT_MODEL = PROJECT_ROOT / "models/paddledetection/person_mot"
DEFAULT_PPDET_ROOT = PROJECT_ROOT / "src/PaddleDetection-release-2.9"
CLASS_NAMES = ["normal", "steal"]


def _prepare_paddledet_paths(root: Path) -> None:
    paths = [
        root, root / "deploy", root / "deploy/pipeline", root / "deploy/python",
        root / "deploy/pptracking/python", root / "deploy/pptracking",
    ]
    for p in reversed(paths):
        text = str(p)
        if text not in sys.path:
            sys.path.insert(0, text)


# ---------------------------------------------------------------------------
# Pass 1: detection + constrained-IoU tracking (training-consistent)
# ---------------------------------------------------------------------------

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


class PureDetector:
    """PaddleDetection Detector (no global MOT tracker), in-memory frames via temp file."""

    def __init__(self, model_dir: str, device: str, threshold: float):
        import paddle
        from det_infer import Detector
        paddle.enable_static()
        self._tmp_path = tempfile.mktemp(suffix=".jpg")
        self._det = Detector(model_dir=model_dir, device=device, threshold=threshold)

    def detect(self, frame_bgr):
        import cv2
        cv2.imwrite(self._tmp_path, frame_bgr)
        try:
            inputs = self._det.preprocess([self._tmp_path])
            result = self._det.postprocess(inputs, self._det.predict())
        finally:
            try:
                os.unlink(self._tmp_path)
            except OSError:
                pass
        boxes = result.get("boxes", np.zeros((0, 6), dtype=np.float32))
        out = []
        for row in boxes:
            cls_id, score, x0, y0, x1, y1 = [float(v) for v in row]
            out.append({"bbox": [x0, y0, x1, y1], "score": score})
        return out


class PersonTrack:
    __slots__ = ("person_id", "bbox", "frame_boxes", "scores", "missed",
                 "born_frame", "last_frame", "active", "_hist",
                 "median_w", "median_h")

    def __init__(self, person_id, bbox, score, frame_id):
        self.person_id = person_id
        self.bbox = [float(v) for v in bbox]
        self.frame_boxes = {frame_id: list(self.bbox)}
        self.scores = {frame_id: score}
        self.missed = 0
        self.born_frame = frame_id
        self.last_frame = frame_id
        self.active = True
        self._hist = [list(self.bbox)]
        self.median_w = self.bbox[2] - self.bbox[0]
        self.median_h = self.bbox[3] - self.bbox[1]

    def append(self, bbox, score, frame_id):
        self.bbox = [float(v) for v in bbox]
        self.frame_boxes[frame_id] = list(self.bbox)
        self.scores[frame_id] = score
        self.missed = 0
        self.last_frame = frame_id
        self._hist.append(list(self.bbox))
        if len(self._hist) > 40:
            self._hist.pop(0)
        ws = [b[2] - b[0] for b in self._hist]
        hs = [b[3] - b[1] for b in self._hist]
        self.median_w = float(np.median(ws))
        self.median_h = float(np.median(hs))

    def median_box(self):
        cxs = [(b[0] + b[2]) / 2.0 for b in self._hist]
        cys = [(b[1] + b[3]) / 2.0 for b in self._hist]
        cx, cy = float(np.median(cxs)), float(np.median(cys))
        w, h = self.median_w, self.median_h
        return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def cmd_detect(args) -> int:
    """Pass 1 (static graph): detection + constrained-IoU tracking -> jsonl."""
    _prepare_paddledet_paths(Path(args.paddledetection_root))
    detector = PureDetector(str(args.model_dir), args.device, args.det_thresh)

    import cv2
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracks: list[PersonTrack] = []
    next_person_id = 1
    stats = {"frames_total": 0, "boxes_total": 0, "tracks_created": 0,
             "tracks_terminated": 0, "assoc_ok": 0}

    frame_id = 0
    out_f = open(args.out, "w", encoding="utf-8")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if ts_ms < args.start_ms:
                frame_id += 1
                continue
            if args.end_ms is not None and ts_ms > args.end_ms:
                break
            if args.max_frames is not None and frame_id >= args.max_frames:
                break

            if frame_id % args.det_stride == 0:
                stats["frames_total"] += 1
                all_dets = detector.detect(frame)
                stats["boxes_total"] += len(all_dets)
                dets = []
                for d in all_dets:
                    x0, y0, x1, y1 = d["bbox"]
                    h, w = y1 - y0, x1 - x0
                    if h >= args.min_person_h and w >= 0.25 * h and d["score"] >= args.det_thresh:
                        dets.append(d)
                dets.sort(key=lambda d: d["score"], reverse=True)

                matched = set()
                for d in dets:
                    best_t, best_v = None, 0.0
                    for t in tracks:
                        if not t.active or t.person_id in matched:
                            continue
                        mbox = t.median_box()
                        v = iou(d["bbox"], mbox)
                        if v < args.iou_thresh:
                            continue
                        d_h = d["bbox"][3] - d["bbox"][1]
                        ratio = d_h / t.median_h if t.median_h > 1 else 1.0
                        if not (1.0 - args.size_ratio <= ratio <= 1.0 + args.size_ratio):
                            continue
                        dcx, dcy = box_center(d["bbox"])
                        mcx, mcy = box_center(mbox)
                        dist = ((dcx - mcx) ** 2 + (dcy - mcy) ** 2) ** 0.5
                        if dist > args.center_ratio * max(t.median_h, 10.0):
                            continue
                        if v > best_v:
                            best_t, best_v = t, v
                    if best_t is not None:
                        best_t.append(d["bbox"], d["score"], frame_id)
                        matched.add(best_t.person_id)
                        stats["assoc_ok"] += 1
                    else:
                        tracks.append(PersonTrack(next_person_id, d["bbox"], d["score"], frame_id))
                        next_person_id += 1
                        stats["tracks_created"] += 1

                for t in tracks:
                    if not t.active or t.person_id in matched:
                        continue
                    t.missed += 1
                    if t.missed >= args.track_buffer:
                        t.active = False
                        stats["tracks_terminated"] += 1

            # write per-frame persons (carry last box on non-stride frames)
            persons = []
            for t in tracks:
                if not t.active:
                    continue
                if frame_id in t.frame_boxes:
                    b = t.frame_boxes[frame_id]
                else:
                    b = t.bbox
                persons.append({"track_id": t.person_id,
                                "box": [float(v) for v in b],
                                "score": t.scores.get(frame_id, t.scores.get(t.last_frame, 0.0))})
            out_f.write(json.dumps({"frame": frame_id, "timestamp_ms": ts_ms,
                                    "persons": persons}, ensure_ascii=False) + "\n")

            if frame_id % 600 == 0:
                print(f"[det] frame {frame_id}/{total} active="
                      f"{sum(1 for t in tracks if t.active)} "
                      f"created={stats['tracks_created']}", flush=True)
            frame_id += 1
    finally:
        cap.release()
        out_f.close()

    stats["tracks_total"] = len(tracks)
    print(f"[det] done: {stats}")
    return 0


# ---------------------------------------------------------------------------
# Pass 2: ppTSM classification + annotation (dynamic graph)
# ---------------------------------------------------------------------------

class TrackBuffer:
    def __init__(self, num_seg):
        self.num_seg = num_seg
        self.frames = []  # PIL RGB crops (most recent last)

    def push(self, crop):
        self.frames.append(crop)
        if len(self.frames) > self.num_seg:
            self.frames.pop(0)

    def window(self):
        if len(self.frames) >= self.num_seg:
            return self.frames[-self.num_seg:]
        pad = [self.frames[0]] * (self.num_seg - len(self.frames)) if self.frames else []
        return pad + self.frames


def _load_pptsm(config: str, weights: str):
    sys.path.insert(0, str(PROJECT_ROOT / "third_party/PaddleVideo"))
    import paddle
    from paddlevideo.utils import get_config
    from paddlevideo.modeling.builder import build_model
    from paddlevideo.loader.builder import build_pipeline

    paddle.set_device("gpu")
    cfg = get_config(config, show=False)
    num_seg = cfg.PIPELINE.test.sample.num_seg
    transform = build_pipeline({"transform": cfg.PIPELINE.test.transform})
    model = build_model(cfg.MODEL)
    state = paddle.load(weights)
    model.set_state_dict(state)
    model.eval()
    return model, transform, num_seg


def classify_window(model, transform, crops):
    """crops: list of PIL RGB images (num_seg). Returns (p_normal, p_steal, verdict)."""
    import paddle
    res = {"imgs": crops}
    res = transform(res)
    arr = np.stack(res["imgs"], axis=0)
    x = paddle.to_tensor(arr[None, ...])
    with paddle.no_grad():
        scores = model((x,), mode="test")
    prob = paddle.nn.functional.softmax(scores[0], axis=-1).numpy()
    p_normal, p_steal = float(prob[0]), float(prob[1])
    verdict = CLASS_NAMES[int(prob.argmax())]
    return p_normal, p_steal, verdict


def _load_dets(det_jsonl: str) -> dict:
    dets = {}
    with open(det_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            dets[d["frame"]] = d
    return dets


def _open_video(video: str):
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, vfps, W, H


def cmd_full_video(args) -> int:
    """Pass 2: annotate the entire video - every person box + ID + confidence, every frame."""
    import cv2
    from PIL import Image

    print("[ann] loading ppTSM ...", flush=True)
    model, transform, num_seg = _load_pptsm(args.config, args.weights)
    dets = _load_dets(args.detections)
    print(f"[ann] detections: {len(dets)} frames", flush=True)

    cap, vfps, W, H = _open_video(args.video)
    out_fps = args.fps or vfps
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))

    frames_det = sorted(dets.keys())
    if not frames_det:
        print("[ann] no detections, nothing to annotate")
        cap.release()
        writer.release()
        return 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames_det[0])
    frame_id = frames_det[0]
    max_det = frames_det[-1]

    buffers: dict[int, TrackBuffer] = {}
    results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id > max_det:
            break
        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        det = dets.get(frame_id)
        persons = det["persons"] if det else []

        active = []
        for p in persons:
            x0, y0, x1, y1 = [float(v) for v in p["box"]]
            tid = p["track_id"]
            xi0, yi0 = max(0, int(x0)), max(0, int(y0))
            xi1, yi1 = min(W, int(x1)), min(H, int(y1))
            if xi1 <= xi0 or yi1 <= yi0:
                continue
            crop_bgr = frame[yi0:yi1, xi0:xi1]
            crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            buf = buffers.setdefault(tid, TrackBuffer(num_seg))
            buf.push(crop)
            pn, ps, vd = classify_window(model, transform, buf.window())
            buf.pn, buf.ps, buf.vd = pn, ps, vd
            active.append((tid, p, pn, ps, vd))

        for tid, p, pn, ps, vd in active:
            x0, y0, x1, y1 = [int(v) for v in p["box"]]
            color = (0, 0, 255) if vd == "steal" else (0, 255, 0)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 3)
            label = f"ID{tid} steal {ps:.2f} normal {pn:.2f}"
            cv2.putText(frame, label, (x0, max(24, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.rectangle(frame, (0, 0), (W, 34), (0, 0, 0), -1)
        info = f"frame {frame_id} t={ts_ms/1000:.2f}s persons={len(active)}"
        cv2.putText(frame, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)

        results.append({"frame": frame_id, "timestamp_ms": ts_ms,
                        "persons": [{"track_id": tid, "box": [round(v, 1) for v in p["box"]],
                                     "p_normal": round(pn, 4), "p_steal": round(ps, 4),
                                     "verdict": vd} for tid, p, pn, ps, vd in active]})
        if frame_id % 600 == 0:
            print(f"[ann] frame {frame_id} persons={len(active)}", flush=True)
        frame_id += 1

    cap.release()
    writer.release()
    print(f"[ann] wrote {args.out} ({frame_id} frames)")

    jsonl = args.out_jsonl or (os.path.splitext(args.out)[0] + ".jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ann] wrote {jsonl}")
    return 0


def cmd_person_clips(args) -> int:
    """Pass 2: one stitched, annotated video PER PERSON track."""
    import cv2
    from PIL import Image

    print("[clip] loading ppTSM ...", flush=True)
    model, transform, num_seg = _load_pptsm(args.config, args.weights)
    dets = _load_dets(args.detections)
    print(f"[clip] detections: {len(dets)} frames", flush=True)

    # collect per-track frame->box (jsonl carries every frame)
    track_frames: dict[int, dict[int, list[float]]] = {}
    for d in dets.values():
        fid = d["frame"]
        for p in d["persons"]:
            track_frames.setdefault(p["track_id"], {})[fid] = [float(v) for v in p["box"]]

    keep = {tid: fb for tid, fb in track_frames.items() if len(fb) >= args.min_track_frames}
    print(f"[clip] tracks={len(track_frames)} kept(>= {args.min_track_frames} frames)={len(keep)}",
          flush=True)
    if not keep:
        print("[clip] nothing to do")
        return 0

    cap, vfps, W, H = _open_video(args.video)
    out_fps = args.fps or vfps

    # ---- single pass: dump per-track crops to temp jpg dirs (memory-safe) ----
    tmp_root = Path(tempfile.mkdtemp(prefix="pptsm_clips_"))
    crop_dirs: dict[int, Path] = {}
    for tid in keep:
        d = tmp_root / f"t{tid:04d}"
        d.mkdir(parents=True, exist_ok=True)
        crop_dirs[tid] = d

    needed = set()
    for fb in keep.values():
        needed.update(fb.keys())
    min_f = min(needed)
    max_f = max(needed)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min_f)
    frame_id = min_f
    n_crops = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id > max_f:
            break
        for tid, fb in keep.items():
            box = fb.get(frame_id)
            if box is None:
                continue
            x0, y0, x1, y1 = [float(v) for v in box]
            xi0, yi0 = max(0, int(x0)), max(0, int(y0))
            xi1, yi1 = min(W, int(x1)), min(H, int(y1))
            if xi1 <= xi0 or yi1 <= yi0:
                continue
            crop_bgr = frame[yi0:yi1, xi0:xi1]
            cv2.imwrite(str(crop_dirs[tid] / f"f{frame_id:06d}.jpg"), crop_bgr)
            n_crops += 1
        frame_id += 1
        if frame_id % 2000 == 0:
            print(f"[clip] scan frame {frame_id} crops={n_crops}", flush=True)
    cap.release()
    print(f"[clip] scanned, {n_crops} crops dumped", flush=True)

    # ---- per track: sliding-window classify + stitch annotated video ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tid, fb in keep.items():
        fids = sorted(fb.keys())
        buf = TrackBuffer(num_seg)
        frames_out = []   # (box, PIL crop, pn, ps, vd)
        json_rows = []
        for fid in fids:
            jpg = crop_dirs[tid] / f"f{fid:06d}.jpg"
            if not jpg.exists():
                continue
            crop = Image.open(jpg).convert("RGB")
            buf.push(crop)
            pn, ps, vd = classify_window(model, transform, buf.window())
            frames_out.append((fb[fid], crop, pn, ps, vd))
            json_rows.append({"frame": fid, "box": [round(v, 1) for v in fb[fid]],
                              "p_normal": round(pn, 4), "p_steal": round(ps, 4), "verdict": vd})
        if not frames_out:
            continue
        Hc = max(int(r[0][3] - r[0][1]) for r in frames_out)
        Wc = max(int(r[0][2] - r[0][0]) for r in frames_out)
        Hc, Wc = max(Hc, 64), max(Wc, 64)
        writer = cv2.VideoWriter(str(out_dir / f"person-{tid:03d}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (Wc, Hc))
        for box, crop, pn, ps, vd in frames_out:
            cw, ch = crop.size
            scale = min(Wc / cw, Hc / ch)
            nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
            rgb_arr = np.array(crop)
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            resized = cv2.resize(bgr_arr, (nw, nh))
            canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
            ox, oy = (Wc - nw) // 2, (Hc - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = resized
            color = (0, 0, 255) if vd == "steal" else (0, 255, 0)
            cv2.rectangle(canvas, (ox, oy), (ox + nw, oy + nh), color, 3)
            label = f"ID{tid} steal {ps:.2f} normal {pn:.2f}"
            cv2.putText(canvas, label, (8, max(24, oy + 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            writer.write(canvas)
        writer.release()
        with open(out_dir / f"person-{tid:03d}.jsonl", "w", encoding="utf-8") as f:
            for r in json_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[clip] person-{tid:03d}: {len(frames_out)} frames -> "
              f"{out_dir}/person-{tid:03d}.mp4", flush=True)

    import shutil
    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"[done] person clips in {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# One-shot orchestrators (subprocess isolation: static vs dynamic graph)
# ---------------------------------------------------------------------------

def _run_stage(sub_args: list[str]) -> int:
    """Run this script in a subprocess with the given argv (after 'python script')."""
    cmd = [sys.executable, os.path.abspath(__file__)] + sub_args
    print(f"[run] {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def cmd_run_full(args) -> int:
    tmp_det = tempfile.mktemp(suffix=".jsonl")
    det_args = _det_args_to_list(args) + ["--out", tmp_det]
    rc = _run_stage(["detect"] + det_args)
    if rc != 0:
        print(f"[run] detect failed rc={rc}")
        return rc
    ann_args = ["--video", args.video, "--detections", tmp_det,
                "--config", args.config, "--weights", args.weights,
                "--out", args.out]
    if args.out_jsonl:
        ann_args += ["--out-jsonl", args.out_jsonl]
    if args.fps:
        ann_args += ["--fps", str(args.fps)]
    rc = _run_stage(["full-video"] + ann_args)
    try:
        os.unlink(tmp_det)
    except OSError:
        pass
    return rc


def cmd_run_clips(args) -> int:
    tmp_det = tempfile.mktemp(suffix=".jsonl")
    det_args = _det_args_to_list(args) + ["--out", tmp_det]
    rc = _run_stage(["detect"] + det_args)
    if rc != 0:
        print(f"[run] detect failed rc={rc}")
        return rc
    ann_args = ["--video", args.video, "--detections", tmp_det,
                "--config", args.config, "--weights", args.weights,
                "--out-dir", args.out_dir, "--min-track-frames", str(args.min_track_frames)]
    if args.fps:
        ann_args += ["--fps", str(args.fps)]
    rc = _run_stage(["person-clips"] + ann_args)
    try:
        os.unlink(tmp_det)
    except OSError:
        pass
    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_det_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MOT_MODEL,
                   help="person_mot model dir (used as pure detector)")
    p.add_argument("--paddledetection-root", type=Path, default=DEFAULT_PPDET_ROOT)
    p.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    p.add_argument("--det-stride", type=int, default=3,
                   help="run detection every N frames (tracking carries boxes between)")
    p.add_argument("--det-thresh", type=float, default=0.6)
    p.add_argument("--min-person-h", type=float, default=150.0)
    p.add_argument("--iou-thresh", type=float, default=0.3)
    p.add_argument("--size-ratio", type=float, default=0.6)
    p.add_argument("--center-ratio", type=float, default=1.0)
    p.add_argument("--track-buffer", type=int, default=5)
    p.add_argument("--min-track-frames", type=int, default=8,
                   help="drop tracks shorter than this (person-clips only)")
    p.add_argument("--start-ms", type=int, default=0)
    p.add_argument("--end-ms", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)


def _det_args_to_list(args) -> list[str]:
    out = ["--video", args.video, "--model-dir", str(args.model_dir),
           "--paddledetection-root", str(args.paddledetection_root),
           "--device", args.device, "--det-stride", str(args.det_stride),
           "--det-thresh", str(args.det_thresh), "--min-person-h", str(args.min_person_h),
           "--iou-thresh", str(args.iou_thresh), "--size-ratio", str(args.size_ratio),
           "--center-ratio", str(args.center_ratio), "--track-buffer", str(args.track_buffer),
           "--start-ms", str(args.start_ms)]
    if args.end_ms is not None:
        out += ["--end-ms", str(args.end_ms)]
    if args.max_frames is not None:
        out += ["--max-frames", str(args.max_frames)]
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # detect
    p = sub.add_parser("detect", help="Pass 1: detection + tracking -> detections jsonl")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    _add_det_args(p)
    p.set_defaults(func=cmd_detect)

    # full-video
    p = sub.add_parser("full-video", help="Pass 2: annotate entire video from detections")
    p.add_argument("--video", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--out-jsonl", default=None)
    p.add_argument("--fps", type=int, default=None)
    p.set_defaults(func=cmd_full_video)

    # person-clips
    p = sub.add_parser("person-clips", help="Pass 2: one stitched video PER PERSON")
    p.add_argument("--video", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--min-track-frames", type=int, default=8)
    p.set_defaults(func=cmd_person_clips)

    # one-shot
    p = sub.add_parser("run-full", help="One-shot: detect + annotate entire video")
    p.add_argument("--video", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--out-jsonl", default=None)
    p.add_argument("--fps", type=int, default=None)
    _add_det_args(p)
    p.set_defaults(func=cmd_run_full)

    p = sub.add_parser("run-clips", help="One-shot: detect + per-person clips")
    p.add_argument("--video", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fps", type=int, default=None)
    _add_det_args(p)
    p.set_defaults(func=cmd_run_clips)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
