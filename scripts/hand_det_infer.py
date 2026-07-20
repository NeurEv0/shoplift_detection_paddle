from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shoplift.hand_state_classification.hand_detector import WiLoRHandDetector  # noqa: E402


def infer_single_sample(
    input_path: Path,
    output_path: Path,
    model: WiLoRHandDetector,
    conf: float,
) -> None:
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Failed to read image: {input_path}")
    detections = model.detect(image, conf=conf)
    vis = model.draw_detections(image, detections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)

    print([det.to_dict() for det in detections])
    print(f"saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WiLoR hand detection on images.")
    parser.add_argument(
        "--input",
        default="datasets/test_props/origins",
        help="Image file or image folder. Defaults to datasets/test_props/origins.",
    )
    parser.add_argument(
        "--output",
        default="outputs/inference_visualization/hand_det_output_origins",
        help="Output file or folder.",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/_downloads/hand_detector.pt",
        help="Path to hand_detector.pt.",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model = WiLoRHandDetector(args.checkpoint)

    if input_path.is_file():
        if output_path.suffix == "":
            output_path = output_path / input_path.name
        infer_single_sample(input_path, output_path, model, args.conf)
    else:
        input_images = sorted(input_path.glob("*.jpg"))
        for image_path in input_images:
            infer_single_sample(
                image_path,
                output_path / image_path.name,
                model,
                args.conf,
            )
