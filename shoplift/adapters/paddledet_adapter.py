"""PaddleDetection output adapters.

The adapter is intentionally model-free: it accepts PaddleDetection/Pipeline
postprocess outputs as Python dictionaries, lists, tuples, or NumPy-like arrays
and converts them into the internal dataclasses used by the shoplift modules.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from shoplift.core.types import BBox, BodyPose, DetectionBox, FrameMeta, HandRegion, Point, Tracklet
from shoplift.vision.body_pose import BodyPoseBuilder
from shoplift.vision.pose_hand import HandRegionExtractor, PersonPose


DEFAULT_CLASS_ID_TO_CATEGORY = {0: "person"}
SHOPLIFT_CLASS_ID_TO_CATEGORY = {
    0: "item",
    1: "product",
    2: "bag",
    3: "backpack",
    4: "handbag",
    5: "basket",
    6: "cart",
    7: "stroller",
    8: "helmet",
    9: "clothing_region",
    10: "pocket_region",
}
DEFAULT_CATEGORY_ALIASES = {
    "product": "item",
    "backpack": "bag",
    "handbag": "bag",
    "cart": "basket",
    "pocket_region": "clothing_region",
}

@dataclass(frozen=True)
class PaddleDetectionEnvironment:
    root: Path = Path("src/PaddleDetection-release-2.9")

    def resolve_root(self, project_root: Path | None = None) -> Path:
        root = self.root
        if root.is_absolute():
            return root
        return (project_root or Path.cwd()) / root

    def package_path(self, project_root: Path | None = None) -> Path:
        return self.resolve_root(project_root) / "ppdet"

    def is_available(self, project_root: Path | None = None) -> bool:
        package_path = self.package_path(project_root)
        return package_path.exists() and (package_path / "__init__.py").exists()


@dataclass(frozen=True)
class PaddleDetectionFrameResult:
    """Internal per-frame view of PaddleDetection outputs."""

    frame: FrameMeta
    detections: tuple[DetectionBox, ...] = field(default_factory=tuple)
    person_tracks: tuple[Tracklet, ...] = field(default_factory=tuple)
    body_poses: tuple[BodyPose, ...] = field(default_factory=tuple)
    hand_regions: tuple[HandRegion, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame.to_dict(),
            "detections": [detection.to_dict() for detection in self.detections],
            "person_tracks": [tracklet.to_dict() for tracklet in self.person_tracks],
            "body_poses": [body_pose.to_dict() for body_pose in self.body_poses],
            "hand_regions": [hand_region.to_dict() for hand_region in self.hand_regions],
            "metadata": self.metadata,
        }


def ensure_ppdet_path(root: str | Path = "src/PaddleDetection-release-2.9") -> Path:
    resolved = Path(root).resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    return resolved


def _to_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _as_sequence(value: Any) -> list[Any]:
    value = _to_python(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_row(value: Any) -> bool:
    value = _to_python(value)
    if isinstance(value, Mapping):
        return True
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return False
    return all(not isinstance(_to_python(item), (list, tuple, dict)) for item in value)


def _rows_from_result(result: Any, key: str = "boxes") -> list[Any]:
    result = _to_python(result)
    if result is None:
        return []
    if isinstance(result, Mapping):
        if key in result:
            return _as_sequence(result[key])
        if "output" in result:
            return _as_sequence(result["output"])
        if _is_row(result):
            return [result]
        return []
    if _is_row(result):
        return [result]
    return _as_sequence(result)


def _first_present(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _clip_bbox(bbox: BBox, frame: FrameMeta | None) -> BBox:
    x1, y1, x2, y2 = bbox
    if frame is None:
        return (x1, y1, x2, y2)
    return (
        max(0.0, min(float(frame.width), x1)),
        max(0.0, min(float(frame.height), y1)),
        max(0.0, min(float(frame.width), x2)),
        max(0.0, min(float(frame.height), y2)),
    )


def _bbox_from_values(values: Sequence[Any], bbox_format: str = "xyxy") -> BBox:
    if len(values) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    v0, v1, v2, v3 = (float(value) for value in values)
    if bbox_format in {"xywh", "tlwh"}:
        return (v0, v1, v0 + v2, v1 + v3)
    if bbox_format != "xyxy":
        raise ValueError(f"unsupported bbox_format: {bbox_format}")
    return (v0, v1, v2, v3)


def _safe_score(value: Any, default: float = 1.0) -> float:
    if value is None:
        return default
    score = float(value)
    return max(0.0, min(1.0, score))


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(_to_python(value)))
    except (TypeError, ValueError):
        return None


def _format_track_id(category: str, raw_track_id: Any) -> str:
    text = str(_to_python(raw_track_id))
    if text.startswith(f"{category}-"):
        return text
    if text.endswith(".0"):
        text = text[:-2]
    return f"{category}-{text}"


def _metadata_without_keys(mapping: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: _to_python(value) for key, value in mapping.items() if key not in keys}


@dataclass(frozen=True)
class PaddleDetectionAdapter:
    """Convert PaddleDetection postprocess outputs into shoplift dataclasses."""

    class_id_to_category: Mapping[int, str] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_ID_TO_CATEGORY)
    )
    category_aliases: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_CATEGORY_ALIASES))
    min_detection_score: float = 0.0
    min_keypoint_score: float = 0.2
    hand_min_size_px: float = 18.0
    hand_scale_from_forearm: float = 0.55

    def category_for(self, class_id: Any = None, label: Any = None) -> str:
        if label is not None and str(label) != "":
            category = str(_to_python(label)).strip().lower()
        elif class_id is not None:
            parsed_class_id = _to_int_or_none(class_id)
            category = (
                self.class_id_to_category.get(parsed_class_id)
                if parsed_class_id is not None
                else None
            )
            if category is None:
                category = (
                    f"class_{parsed_class_id}"
                    if parsed_class_id is not None
                    else str(_to_python(class_id)).strip().lower()
                )
        else:
            category = "object"
        return self.category_aliases.get(category, category)

    def convert_detection_result(
        self,
        result: Any,
        frame: FrameMeta,
        *,
        bbox_format: str = "xyxy",
        source: str = "det",
    ) -> tuple[DetectionBox, ...]:
        """Convert PPDet detector rows.

        Supported rows:
        - `[class_id, score, x1, y1, x2, y2]`
        - dict rows with `bbox`/`box`, `score`, and class/label fields
        """

        detections: list[DetectionBox] = []
        for index, row in enumerate(_rows_from_result(result, "boxes")):
            parsed = self._parse_detection_row(row, bbox_format=bbox_format)
            if parsed is None or parsed["score"] < self.min_detection_score:
                continue
            detections.append(
                DetectionBox(
                    box_id=f"{source}-{frame.frame_id}-{index}",
                    frame_id=frame.frame_id,
                    timestamp_ms=frame.timestamp_ms,
                    category=parsed["category"],
                    bbox=_clip_bbox(parsed["bbox"], frame),
                    score=parsed["score"],
                    track_id=parsed.get("track_id"),
                    attributes=parsed["attributes"],
                )
            )
        return tuple(detections)

    def convert_mot_result(
        self,
        result: Any,
        frame: FrameMeta,
        *,
        source: str = "mot",
    ) -> tuple[Tracklet, ...]:
        """Convert PP-Human/MOT outputs.

        Supported inputs:
        - Pipeline `mot_res`: `{"boxes": [[track_id, class_id, score, x1, y1, x2, y2], ...]}`
        - SDE output: `{"online_tlwhs": ..., "online_scores": ..., "online_ids": ...}`
        - SDE tuple/list: `[online_tlwhs, online_scores, online_ids]`
        """

        rows = self._mot_rows(result)
        tracklets: list[Tracklet] = []
        for index, row in enumerate(rows):
            parsed = self._parse_mot_row(row)
            if parsed is None or parsed["score"] < self.min_detection_score:
                continue
            category = parsed["category"]
            track_id = _format_track_id(category, parsed["raw_track_id"])
            detection = DetectionBox(
                box_id=f"{source}-{frame.frame_id}-{track_id}",
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                category=category,
                bbox=_clip_bbox(parsed["bbox"], frame),
                score=parsed["score"],
                track_id=track_id,
                attributes={
                    **parsed["attributes"],
                    "source_index": index,
                    "raw_track_id": _to_python(parsed["raw_track_id"]),
                },
            )
            tracklets.append(
                Tracklet(
                    track_id=track_id,
                    category=category,
                    boxes=(detection,),
                    timestamps_ms=(frame.timestamp_ms,),
                    metadata={"source": source},
                )
            )
        return tuple(tracklets)

    def convert_keypoint_result(
        self,
        result: Any,
        frame: FrameMeta,
        person_tracks: Sequence[Tracklet] | None = None,
    ) -> tuple[HandRegion, ...]:
        """Convert PP-Human keypoints into left/right hand ROIs."""

        keypoints, scores = self._split_keypoint_result(result)
        if not keypoints:
            return ()

        person_tracks = tuple(person_tracks or ())
        person_poses: list[PersonPose] = []
        for person_index, person_keypoints in enumerate(keypoints):
            person_scores = scores[person_index] if person_index < len(scores) else []
            person_track = person_tracks[person_index] if person_index < len(person_tracks) else None
            person_track_id = person_track.track_id if person_track is not None else f"person-{person_index}"
            person_bbox = person_track.boxes[-1].bbox if person_track and person_track.boxes else None
            person_poses.append(
                PersonPose(
                    person_track_id=person_track_id,
                    keypoints=tuple(person_keypoints),
                    scores=tuple(person_scores),
                    person_bbox=person_bbox,
                    metadata={"person_index": person_index},
                )
            )
        extractor = HandRegionExtractor(
            min_keypoint_score=self.min_keypoint_score,
            hand_min_size_px=self.hand_min_size_px,
            hand_scale_from_forearm=self.hand_scale_from_forearm,
            source="ppdet_keypoint",
        )
        return extractor.extract(frame, person_poses)

    def convert_body_pose_result(
        self,
        result: Any,
        frame: FrameMeta,
        person_tracks: Sequence[Tracklet] | None = None,
    ) -> tuple[BodyPose, ...]:
        """Convert PP-Human keypoints into full-body pose evidence."""

        keypoints, scores = self._split_keypoint_result(result)
        if not keypoints:
            return ()
        builder = BodyPoseBuilder(
            min_keypoint_score=self.min_keypoint_score,
            source="ppdet_keypoint",
        )
        return builder.build(frame, keypoints, scores, person_tracks)

    def convert_frame_result(
        self,
        frame: FrameMeta,
        *,
        det_result: Any | None = None,
        mot_result: Any | None = None,
        keypoint_result: Any | None = None,
    ) -> PaddleDetectionFrameResult:
        detections = self.convert_detection_result(det_result, frame) if det_result is not None else ()
        person_tracks = self.convert_mot_result(mot_result, frame) if mot_result is not None else ()
        hand_regions = (
            self.convert_keypoint_result(keypoint_result, frame, person_tracks)
            if keypoint_result is not None
            else ()
        )
        body_poses = (
            self.convert_body_pose_result(keypoint_result, frame, person_tracks)
            if keypoint_result is not None
            else ()
        )
        return PaddleDetectionFrameResult(
            frame=frame,
            detections=detections,
            person_tracks=person_tracks,
            body_poses=body_poses,
            hand_regions=hand_regions,
            metadata={
                "det_count": len(detections),
                "track_count": len(person_tracks),
                "body_pose_count": len(body_poses),
                "hand_region_count": len(hand_regions),
            },
        )

    def convert_pphuman_result(self, result: Mapping[str, Any], frame: FrameMeta) -> PaddleDetectionFrameResult:
        """Convert a Pipeline `Result.res_dict`-like dictionary."""

        det_result = result.get("det")
        mot_result = result.get("mot")
        keypoint_result = result.get("kpt") or result.get("keypoint")
        return self.convert_frame_result(
            frame,
            det_result=det_result,
            mot_result=mot_result,
            keypoint_result=keypoint_result,
        )

    def _parse_detection_row(self, row: Any, *, bbox_format: str) -> dict[str, Any] | None:
        row = _to_python(row)
        if isinstance(row, Mapping):
            bbox = _first_present(row, ("bbox", "box", "rect"))
            if bbox is None:
                coords = _first_present(row, ("xyxy", "tlbr"))
                bbox = coords if coords is not None else [
                    row.get("x1", row.get("xmin")),
                    row.get("y1", row.get("ymin")),
                    row.get("x2", row.get("xmax")),
                    row.get("y2", row.get("ymax")),
                ]
            row_bbox_format = str(row.get("bbox_format", bbox_format)).lower()
            class_id = _first_present(row, ("class_id", "category_id", "label_id", "cls_id", "class"))
            label = _first_present(row, ("category", "label", "class_name", "name"))
            score = _first_present(row, ("score", "confidence", "conf"))
            category = self.category_for(class_id, label)
            ignored = {
                "bbox",
                "box",
                "rect",
                "xyxy",
                "tlbr",
                "bbox_format",
                "class_id",
                "category_id",
                "label_id",
                "cls_id",
                "class",
                "category",
                "label",
                "class_name",
                "name",
                "score",
                "confidence",
                "conf",
                "track_id",
            }
            track_id = row.get("track_id")
            return {
                "category": category,
                "bbox": _bbox_from_values(_as_sequence(bbox), row_bbox_format),
                "score": _safe_score(score),
                "track_id": str(track_id) if track_id is not None else None,
                "attributes": {
                    **_metadata_without_keys(row, ignored),
                    "source_class_id": _to_python(class_id),
                    "source_category": _to_python(label),
                },
            }

        values = _as_sequence(row)
        if len(values) < 6:
            return None
        class_id, score = values[0], values[1]
        category = self.category_for(class_id)
        return {
            "category": category,
            "bbox": _bbox_from_values(values[2:6], bbox_format),
            "score": _safe_score(score),
            "track_id": None,
            "attributes": {
                "source_class_id": _to_python(class_id),
                "source_category": self.class_id_to_category.get(_to_int_or_none(class_id)),
            },
        }

    def _parse_mot_row(self, row: Any) -> dict[str, Any] | None:
        row = _to_python(row)
        if isinstance(row, Mapping):
            raw_track_id = _first_present(row, ("track_id", "id", "mot_id"))
            bbox = _first_present(row, ("bbox", "box", "rect", "tlwh", "xywh"))
            if raw_track_id is None or bbox is None:
                return None
            class_id = _first_present(row, ("class_id", "category_id", "label_id", "cls_id", "class"))
            label = _first_present(row, ("category", "label", "class_name", "name"))
            score = _first_present(row, ("score", "confidence", "conf"))
            bbox_format = str(row.get("bbox_format", "xyxy")).lower()
            if "tlwh" in row or "xywh" in row:
                bbox_format = "xywh"
            category = self.category_for(class_id, label)
            ignored = {
                "track_id",
                "id",
                "mot_id",
                "bbox",
                "box",
                "rect",
                "tlwh",
                "xywh",
                "bbox_format",
                "class_id",
                "category_id",
                "label_id",
                "cls_id",
                "class",
                "category",
                "label",
                "class_name",
                "name",
                "score",
                "confidence",
                "conf",
            }
            return {
                "raw_track_id": raw_track_id,
                "category": category,
                "bbox": _bbox_from_values(_as_sequence(bbox), bbox_format),
                "score": _safe_score(score),
                "attributes": {
                    **_metadata_without_keys(row, ignored),
                    "source_class_id": _to_python(class_id),
                    "source_category": _to_python(label),
                },
            }

        values = _as_sequence(row)
        if len(values) < 7:
            return None
        raw_track_id, class_id, score = values[0], values[1], values[2]
        return {
            "raw_track_id": raw_track_id,
            "category": self.category_for(class_id),
            "bbox": _bbox_from_values(values[3:7], "xyxy"),
            "score": _safe_score(score),
            "attributes": {
                "source_class_id": _to_python(class_id),
                "source_category": self.class_id_to_category.get(_to_int_or_none(class_id)),
            },
        }

    def _mot_rows(self, result: Any) -> list[Any]:
        result = _to_python(result)
        if result is None:
            return []
        if isinstance(result, Mapping) and "boxes" in result:
            return _as_sequence(result["boxes"])
        if isinstance(result, Mapping) and {"online_tlwhs", "online_scores", "online_ids"} <= set(result):
            return self._mot_rows_from_online(
                result["online_tlwhs"],
                result["online_scores"],
                result["online_ids"],
            )
        if isinstance(result, (list, tuple)) and len(result) == 3:
            return self._mot_rows_from_online(result[0], result[1], result[2])
        return _rows_from_result(result, "boxes")

    def _mot_rows_from_online(self, tlwhs: Any, scores: Any, ids: Any) -> list[list[Any]]:
        rows: list[list[Any]] = []
        tlwhs = _to_python(tlwhs)
        scores = _to_python(scores)
        ids = _to_python(ids)

        if isinstance(tlwhs, Mapping):
            class_ids = sorted(tlwhs.keys())
            for class_id in class_ids:
                rows.extend(
                    self._mot_rows_from_online_class(
                        class_id,
                        _as_sequence(tlwhs.get(class_id)),
                        _as_sequence(scores.get(class_id) if isinstance(scores, Mapping) else []),
                        _as_sequence(ids.get(class_id) if isinstance(ids, Mapping) else []),
                    )
                )
            return rows

        return self._mot_rows_from_online_class(0, _as_sequence(tlwhs), _as_sequence(scores), _as_sequence(ids))

    def _mot_rows_from_online_class(
        self,
        class_id: Any,
        tlwhs: Sequence[Any],
        scores: Sequence[Any],
        ids: Sequence[Any],
    ) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for index, tlwh in enumerate(tlwhs):
            if index >= len(ids):
                continue
            x, y, w, h = (float(value) for value in _as_sequence(tlwh)[:4])
            score = scores[index] if index < len(scores) else 1.0
            rows.append([ids[index], class_id, score, x, y, x + w, y + h])
        return rows

    def _split_keypoint_result(self, result: Any) -> tuple[list[list[Point]], list[list[float]]]:
        result = _to_python(result)
        if result is None:
            return [], []

        if isinstance(result, Mapping):
            if "keypoint" in result:
                keypoint_value = _to_python(result["keypoint"])
                if (
                    isinstance(keypoint_value, (list, tuple))
                    and len(keypoint_value) == 2
                    and self._looks_like_keypoint_score_array(keypoint_value[1])
                ):
                    raw_keypoints, raw_scores = keypoint_value[0], keypoint_value[1]
                else:
                    raw_keypoints = keypoint_value
                    raw_scores = result.get("score") or result.get("scores")
            else:
                raw_keypoints = result.get("keypoints") or result.get("kpts")
                raw_scores = result.get("score") or result.get("scores")
        else:
            raw_keypoints = result
            raw_scores = None

        keypoints = self._normalize_keypoints(raw_keypoints)
        scores = self._normalize_keypoint_scores(raw_scores, raw_keypoints, len(keypoints))
        return keypoints, scores

    def _looks_like_single_person_keypoints(self, value: Any) -> bool:
        value = _to_python(value)
        if not isinstance(value, (list, tuple)) or not value:
            return False
        first = _to_python(value[0])
        return isinstance(first, (list, tuple)) and first and _is_number(_to_python(first[0]))

    def _looks_like_keypoint_score_array(self, value: Any) -> bool:
        value = _to_python(value)
        sequence = _as_sequence(value)
        if not sequence:
            return False
        if all(_is_number(_to_python(item)) for item in sequence):
            return True
        for item in sequence:
            item_sequence = _as_sequence(item)
            if not item_sequence or len(item_sequence) in {2, 3}:
                return False
            if not all(_is_number(_to_python(score)) for score in item_sequence):
                return False
        return True

    def _normalize_keypoints(self, raw_keypoints: Any) -> list[list[Point]]:
        raw_keypoints = _to_python(raw_keypoints)
        if raw_keypoints is None:
            return []
        people = _as_sequence(raw_keypoints)
        if people and self._looks_like_single_person_keypoints(people):
            people = [people]

        normalized_people: list[list[Point]] = []
        for person in people:
            points: list[Point] = []
            for point in _as_sequence(person):
                values = _as_sequence(point)
                if len(values) < 2:
                    continue
                points.append((float(values[0]), float(values[1])))
            normalized_people.append(points)
        return normalized_people

    def _normalize_keypoint_scores(
        self,
        raw_scores: Any,
        raw_keypoints: Any,
        person_count: int,
    ) -> list[list[float]]:
        raw_scores = _to_python(raw_scores)
        if raw_scores is None:
            keypoint_people = _as_sequence(_to_python(raw_keypoints))
            if keypoint_people and self._looks_like_single_person_keypoints(keypoint_people):
                keypoint_people = [keypoint_people]
            derived: list[list[float]] = []
            for person in keypoint_people:
                person_scores: list[float] = []
                for point in _as_sequence(person):
                    values = _as_sequence(point)
                    person_scores.append(_safe_score(values[2]) if len(values) >= 3 else 1.0)
                derived.append(person_scores)
            return derived

        score_people = _as_sequence(raw_scores)
        if score_people and all(_is_number(_to_python(item)) for item in score_people):
            score_people = [score_people]
        normalized = [[_safe_score(score) for score in _as_sequence(person)] for person in score_people]
        while len(normalized) < person_count:
            normalized.append([])
        return normalized
