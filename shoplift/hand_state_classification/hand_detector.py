from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


ImageInput = Union[str, Path, np.ndarray]


@dataclass(frozen=True)
class HandDetection:
    """Single 2D hand detection in full-image xyxy coordinates."""

    bbox: np.ndarray
    score: float
    class_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", np.asarray(self.bbox, dtype=np.float32))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "class_id", int(self.class_id))

    @property
    def is_right(self) -> int:
        return int(self.class_id == 1)

    @property
    def side(self) -> str:
        return "right" if self.is_right else "left"

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.astype(float).tolist(),
            "score": float(self.score),
            "class_id": int(self.class_id),
            "side": self.side,
            "is_right": self.is_right,
        }


class WiLoRHandDetector:
    """Reusable wrapper around WiLoR's YOLO hand detector.

    The pretrained detector uses two classes:
    class 0 = left hand, class 1 = right hand. The `boxes_and_sides`
    helper returns exactly the arrays expected by `ViTDetDataset`.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "./pretrained_models/detector.pt",
        device: Optional[Union[str, object]] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "WiLoRHandDetector requires ultralytics. Install it with "
                "`pip install ultralytics==8.1.34` or `pip install -r requirements.txt`."
            ) from exc

        self.checkpoint_path = Path(checkpoint_path)
        self.model = YOLO(str(self.checkpoint_path))
        if device is not None:
            self.to(device)

    def to(self, device: Union[str, object]) -> "WiLoRHandDetector":
        self.model = self.model.to(device)
        return self

    def detect(
        self,
        image: ImageInput,
        conf: float = 0.3,
        iou: float = 0.5,
        verbose: bool = False,
        **predict_kwargs,
    ) -> List[HandDetection]:
        """Run hand detection on an image path or numpy image.

        For numpy inputs, keep the same convention used by Ultralytics and
        the original WiLoR demos: OpenCV BGR images are accepted directly.
        """

        result = self.model(
            image,
            conf=conf,
            iou=iou,
            verbose=verbose,
            **predict_kwargs,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float32)
        class_ids = boxes.cls.detach().cpu().numpy().astype(np.int64)

        return [
            HandDetection(bbox=bbox, score=float(score), class_id=int(class_id))
            for bbox, score, class_id in zip(xyxy, scores, class_ids)
        ]

    def __call__(self, image: ImageInput, *args, **kwargs) -> List[HandDetection]:
        return self.detect(image, *args, **kwargs)

    @staticmethod
    def boxes_and_sides(
        detections: Sequence[HandDetection],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(boxes, is_right)` arrays compatible with WiLoR crops."""

        if len(detections) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        boxes = np.stack([det.bbox for det in detections]).astype(np.float32)
        is_right = np.asarray([det.is_right for det in detections], dtype=np.float32)
        return boxes, is_right

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: Sequence[HandDetection],
        thickness: int = 3,
    ) -> np.ndarray:
        """Draw detections on a copy of an RGB or BGR image."""

        import cv2

        output = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int).tolist()
            color = (53, 165, 154) if det.side == "left" else (255, 199, 60)
            label = f"{det.side[0].upper()} - {det.score:.3f}"
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(output, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
            cv2.putText(
                output,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
        return output
