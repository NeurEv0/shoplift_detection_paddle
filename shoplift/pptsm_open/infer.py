"""PP-TSM open-binary production inference.

Loads the v2 ep26 open model (ResNetTweaksTSM-50, num_seg=16, 2 classes:
``0=not-open`` / ``1=open 拆包装``) and classifies 16-frame sliding windows over a
person-crop timeline, producing per-window open probabilities.

This module imports Paddle / PaddleVideo lazily (inside functions), so the pure
fusion/eventize layer stays unit-testable without those dependencies. Running
the actual inference requires the PaddleVideo source tree and the open config +
weights (both live on the training server).

Frame-cadence red line (docs/pptsm_open_binary_report.md §9.4): the production
person timeline MUST be resampled to the same cadence used at training (16 frames
≈ the same number of seconds as the master timeline). The cadence is controlled
by ``timeline_fps`` (see ``DEFAULT_TIMELINE_FPS``): the training-domain open
segment ``steal_2_d03`` has master frames every 8 source frames at 25 fps, i.e.
``25 / 8 = 3.125 fps`` (0.32 s per frame, 16-frame window ≈ 5.12 s).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from shoplift.pptsm_open.types import OpenWindow

# Training-domain master cadence: steal_2_d03 master frames are every 8 source
# frames at 25 fps => 3.125 fps (verified against .dsh/tmp/master_meta.json).
DEFAULT_TIMELINE_FPS = 3.125


@dataclass(frozen=True)
class TimelineFrame:
    """One resampled frame of a person's crop timeline."""

    frame_id: int
    timestamp_ms: int
    # PIL RGB Image crop, matching PaddleVideo Sampler
    # (``Image.open(...).convert('RGB')``). The test transform
    # (Scale→CenterCrop→Image2Array→ImageNet Normalization) expects PIL Images
    # and ImageNet RGB mean/std, exactly as training-time decode produced.
    image: Any = None

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")


class OpenModel:
    """Thin wrapper around a loaded PP-TSM open model + its test transform."""

    def __init__(self, model: Any, transform: Any, num_seg: int) -> None:
        self.model = model
        self.transform = transform
        self.num_seg = num_seg

    @classmethod
    def load(
        cls,
        config_path: str | Path,
        weights_path: str | Path,
        *,
        device: str = "gpu",
        paddlevideo_root: str | Path | None = None,
    ) -> "OpenModel":
        """Load the open model from its training yaml + a ``.pdparams`` checkpoint."""
        import paddle

        if paddlevideo_root is not None:
            root = str(Path(paddlevideo_root).resolve())
            if root not in __import__("sys").path:
                __import__("sys").path.insert(0, root)

        from paddlevideo.modeling.builder import build_model
        from paddlevideo.loader.builder import build_pipeline
        from paddlevideo.utils import get_config

        paddle.set_device(device)
        cfg = get_config(str(config_path), show=False)
        num_seg = int(cfg.PIPELINE.test.sample.num_seg)
        transform = build_pipeline({"transform": cfg.PIPELINE.test.transform})
        model = build_model(cfg.MODEL)
        model.set_state_dict(paddle.load(str(weights_path)))
        model.eval()
        return cls(model, transform, num_seg)

    def open_prob(self, crops: Sequence[Any]) -> float:
        """Return P(open) for ``num_seg`` crops (one window).

        Crops may be PIL RGB Images (the timeline's native form) or BGR numpy
        arrays (converted defensively to match the PaddleVideo Sampler).
        """
        import numpy as np
        import paddle
        from PIL import Image

        if len(crops) != self.num_seg:
            raise ValueError(
                f"expected {self.num_seg} crops, got {len(crops)}"
            )
        pil_crops = []
        for crop in crops:
            if isinstance(crop, np.ndarray):
                crop = Image.fromarray(np.ascontiguousarray(crop[..., ::-1]))
            pil_crops.append(crop)
        transformed = self.transform({"imgs": pil_crops})
        imgs = np.asarray(transformed["imgs"], dtype="float32")
        tensor = paddle.to_tensor(imgs[None, ...])  # (1, num_seg, C, H, W)
        with paddle.no_grad():
            output = self.model((tensor,), mode="test")
        logits = _logits_from_output(output)
        prob = paddle.nn.functional.softmax(paddle.to_tensor(logits), axis=-1).numpy()
        # class 1 = open 拆包装
        return float(prob[1])


def build_person_timeline(
    video_path: str | Path,
    track: Sequence[tuple[int, list[float]]],
    *,
    timeline_fps: float,
    source_fps: float | None = None,
    padding: float = 0.15,
) -> list[TimelineFrame]:
    """Resample one person track into a crop timeline at ``timeline_fps``.

    ``track`` is an ordered list of ``(source_frame_id, bbox)`` anchors (as
    produced by the pipeline MOT). Boxes are linearly interpolated between
    anchors; frames are read from the source video (BGR via OpenCV), cropped to
    the person region (plus ``padding``), and stored as PIL RGB Images so the
    PaddleVideo test transform sees exactly what training-time decode produced.
    """
    import cv2
    import numpy as np
    from PIL import Image

    if not track:
        return []
    anchors = [(int(frame_id), [float(v) for v in bbox]) for frame_id, bbox in track]
    anchors.sort(key=lambda item: item[0])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = source_fps or cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    fps = float(fps)
    step = fps / timeline_fps

    first = anchors[0][0]
    last = anchors[-1][0]
    # target source frames, uniformly spaced at ~timeline_fps
    target_frames: set[int] = set()
    position = float(first)
    while position <= last:
        target_frames.add(int(round(position)))
        position += step

    def bbox_at(frame_id: int) -> list[float] | None:
        if frame_id <= anchors[0][0]:
            return anchors[0][1]
        if frame_id >= anchors[-1][0]:
            return anchors[-1][1]
        for (fa, ba), (fb, bb) in zip(anchors, anchors[1:]):
            if fa <= frame_id <= fb:
                if fb == fa:
                    return list(ba)
                t = (frame_id - fa) / (fb - fa)
                return [ba[i] + t * (bb[i] - ba[i]) for i in range(4)]
        return anchors[-1][1]

    frames: list[TimelineFrame] = []
    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id > last:
                break
            if frame_id in target_frames:
                box = bbox_at(frame_id)
                if box is None:
                    frame_id += 1
                    continue
                crop = _crop_padded(frame, box, padding)
                if crop is not None:
                    rgb = np.ascontiguousarray(crop[..., ::-1])  # BGR -> RGB
                    frames.append(
                        TimelineFrame(
                            frame_id=frame_id,
                            timestamp_ms=int(round(frame_id * 1000.0 / fps)),
                            image=Image.fromarray(rgb),  # PIL RGB, matches Sampler
                        )
                    )
            frame_id += 1
    finally:
        cap.release()

    return frames


def _crop_padded(frame: Any, box: list[float], padding: float) -> Any | None:
    import numpy as np

    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return None
    cx0 = max(0, int(x0 - w * padding))
    cy0 = max(0, int(y0 - h * padding))
    cx1 = min(width - 1, int(x1 + w * padding))
    cy1 = min(height - 1, int(y1 + h * padding))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return np.ascontiguousarray(frame[cy0:cy1, cx0:cx1])


def infer_open_windows(
    model: OpenModel,
    timeline: Sequence[TimelineFrame],
    *,
    camera_id: str,
    person_track_id: str,
    win: int = 16,
    stride: int = 2,
) -> list[OpenWindow]:
    """Slide ``win``-frame windows (stride ``stride``) and classify each.

    Returns one ``OpenWindow`` per window, in timeline order.
    """
    if not camera_id:
        raise ValueError("camera_id must be a non-empty string")
    if not person_track_id:
        raise ValueError("person_track_id must be a non-empty string")
    if win <= 0 or stride <= 0:
        raise ValueError("win and stride must be positive")

    ordered = list(timeline)
    if len(ordered) < win:
        return []

    windows: list[OpenWindow] = []
    for start in range(0, len(ordered) - win + 1, stride):
        chunk = ordered[start : start + win]
        crops = [frame.image for frame in chunk]
        if any(image is None for image in crops):
            continue
        open_prob = model.open_prob(crops)
        anchor = chunk[win // 2]
        windows.append(
            OpenWindow(
                start_timestamp_ms=chunk[0].timestamp_ms,
                end_timestamp_ms=chunk[-1].timestamp_ms,
                open_prob=open_prob,
                frame_id=anchor.frame_id,
            )
        )
    return windows


def _logits_from_output(output: Any) -> Any:
    """Normalize a PaddleVideo ``mode="test"`` output to a flat 2-class logits vector.

    Return shapes vary across PaddleVideo versions/frameworks: ``(1, num_classes)``
    tensor/array, a list/tuple wrapping it, or (rarely) a metrics dict. Best-effort
    extraction to ``[P(not-open), P(open)]`` logits; confirm the exact shape on the
    server's PaddleVideo version on first run.
    """
    if isinstance(output, dict):
        for key in ("output", "logits", "scores", "preds"):
            if key in output:
                output = output[key]
                break
    if isinstance(output, (list, tuple)):
        output = output[0] if output else output
    if hasattr(output, "numpy"):
        output = output.numpy()
    import numpy as np

    array = np.asarray(output).reshape(-1)
    if array.shape[0] < 2:
        raise ValueError(f"unexpected model output shape: {array.shape}")
    return array[:2]


__all__ = [
    "OpenModel",
    "TimelineFrame",
    "build_person_timeline",
    "infer_open_windows",
]
