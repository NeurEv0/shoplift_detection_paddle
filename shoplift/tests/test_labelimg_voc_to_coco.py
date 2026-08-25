"""Tests for scripts/labelimg_voc_to_coco.py."""

from __future__ import annotations

import json

import pytest

from scripts.labelimg_voc_to_coco import (
    build_coco_split,
    load_annotations,
    main,
    materialize_coco,
    parse_voc_annotation,
    read_label_list,
)


LABELS = ("bag", "backpack", "basket")


def _write_image(path, *, width: int = 640, height: int = 480) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A real image is not required: the converter reads dimensions from the XML.
    path.write_bytes(b"fake-jpeg-bytes")


def _write_xml(path, *, filename: str, width: int, height: int, objects: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<annotation>",
        f"  <filename>{filename}</filename>",
        "  <size>",
        f"    <width>{width}</width>",
        f"    <height>{height}</height>",
        "    <depth>3</depth>",
        "  </size>",
    ]
    for obj in objects:
        name = obj["name"]
        difficult = obj.get("difficult", 0)
        truncated = obj.get("truncated", 0)
        x1, y1, x2, y2 = obj["bbox"]
        parts += [
            "  <object>",
            f"    <name>{name}</name>",
            f"    <difficult>{difficult}</difficult>",
            f"    <truncated>{truncated}</truncated>",
            "    <bndbox>",
            f"      <xmin>{x1}</xmin>",
            f"      <ymin>{y1}</ymin>",
            f"      <xmax>{x2}</xmax>",
            f"      <ymax>{y2}</ymax>",
            "    </bndbox>",
            "  </object>",
        ]
    parts.append("</annotation>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _build_workdir(tmp_path):
    xml_dir = tmp_path / "xml"
    image_dir = tmp_path / "images"
    label_list = tmp_path / "label_list.txt"
    label_list.write_text("\n".join(LABELS) + "\n", encoding="utf-8")

    _write_image(image_dir / "frame_0001.jpg")
    _write_xml(
        xml_dir / "frame_0001.xml",
        filename="frame_0001.jpg",
        width=640,
        height=480,
        objects=[
            {"name": "bag", "bbox": (10, 20, 100, 150)},
            {"name": "backpack", "bbox": (200, 50, 300, 260), "difficult": 1},
        ],
    )

    _write_image(image_dir / "frame_0002.jpg")
    _write_xml(
        xml_dir / "frame_0002.xml",
        filename="frame_0002.jpg",
        width=640,
        height=480,
        objects=[{"name": "basket", "bbox": (400, 300, 600, 450)}],
    )

    _write_image(image_dir / "frame_0003.jpg")
    _write_xml(
        xml_dir / "frame_0003.xml",
        filename="frame_0003.jpg",
        width=640,
        height=480,
        objects=[],
    )
    return xml_dir, image_dir, label_list


def test_read_label_list(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("bag\nbackpack\n# comment\nbasket\n", encoding="utf-8")
    assert read_label_list(path) == ("bag", "backpack", "basket")


def test_parse_voc_annotation_filters_difficult(tmp_path):
    xml_dir, image_dir, _ = _build_workdir(tmp_path)
    annotation = parse_voc_annotation(
        xml_dir / "frame_0001.xml",
        image_dir=image_dir,
        labels=LABELS,
        keep_difficult=False,
    )
    assert annotation.width == 640
    assert annotation.height == 480
    assert [obj.name for obj in annotation.objects] == ["bag"]

    annotation_all = parse_voc_annotation(
        xml_dir / "frame_0001.xml",
        image_dir=image_dir,
        labels=LABELS,
        keep_difficult=True,
    )
    assert [obj.name for obj in annotation_all.objects] == ["bag", "backpack"]
    assert annotation_all.objects[1].difficult is True


def test_parse_voc_annotation_unknown_label_raises(tmp_path):
    xml_dir, image_dir, _ = _build_workdir(tmp_path)
    # The referenced image must exist: image-resolution happens before the label check.
    _write_image(image_dir / "bad.jpg")
    bad_xml = xml_dir / "bad.xml"
    _write_xml(
        bad_xml,
        filename="bad.jpg",
        width=100,
        height=100,
        objects=[{"name": "flying_saucer", "bbox": (0, 0, 50, 50)}],
    )
    with pytest.raises(ValueError, match="not in label list"):
        parse_voc_annotation(bad_xml, image_dir=image_dir, labels=LABELS, keep_difficult=False)


def test_load_annotations_missing_image_skip(tmp_path):
    xml_dir, image_dir, _ = _build_workdir(tmp_path)
    orphan = xml_dir / "orphan.xml"
    _write_xml(
        orphan,
        filename="missing.jpg",
        width=100,
        height=100,
        objects=[{"name": "bag", "bbox": (0, 0, 50, 50)}],
    )
    records, skipped, difficult = load_annotations(
        xml_dir,
        image_dir=image_dir,
        labels=LABELS,
        keep_difficult=False,
        missing_image_policy="skip",
    )
    assert skipped == 1
    assert len(records) == 3
    assert difficult == 1  # backpack box from frame_0001


def test_materialize_coco_end_to_end(tmp_path):
    xml_dir, image_dir, label_list = _build_workdir(tmp_path)
    records, skipped, difficult = load_annotations(
        xml_dir,
        image_dir=image_dir,
        labels=LABELS,
        keep_difficult=False,
        missing_image_policy="error",
    )
    output = tmp_path / "coco"
    summary = materialize_coco(
        records,
        output,
        labels=LABELS,
        val_ratio=0.34,
        test_ratio=0.0,
        seed=2026,
        include_empty=True,
        overwrite=True,
        skipped_missing=skipped,
        difficult_box_count=difficult,
    )
    assert summary.xml_count == 3
    assert summary.selected_record_count == 3  # empty frame kept via include_empty
    assert summary.difficult_box_count == 1
    assert summary.split_image_counts["train"] + summary.split_image_counts["val"] == 3

    train_json = json.loads((output / "annotations" / "instances_train.json").read_text(encoding="utf-8"))
    val_json = json.loads((output / "annotations" / "instances_val.json").read_text(encoding="utf-8"))
    assert [category["name"] for category in train_json["categories"]] == list(LABELS)
    assert [category["id"] for category in train_json["categories"]] == [0, 1, 2]

    all_annotations = train_json["annotations"] + val_json["annotations"]
    assert len(all_annotations) == 2  # bag + basket; difficult backpack filtered
    by_name = {}
    for annotation in all_annotations:
        label = LABELS[annotation["category_id"]]
        by_name[label] = annotation
        x, y, width, height = annotation["bbox"]
        assert width > 0 and height > 0
        assert annotation["ignore"] == 0
    assert set(by_name) == {"bag", "basket"}
    assert by_name["bag"]["bbox"] == [10.0, 20.0, 90.0, 130.0]  # xyxy -> xywh


def test_main_cli(tmp_path):
    xml_dir, image_dir, label_list = _build_workdir(tmp_path)
    output = tmp_path / "cli_coco"
    exit_code = main(
        [
            "--xml-dir",
            str(xml_dir),
            "--image-dir",
            str(image_dir),
            "--label-list",
            str(label_list),
            "--output",
            str(output),
            "--val-ratio",
            "0.34",
            "--include-empty",
            "--overwrite",
        ]
    )
    assert exit_code == 0
    summary_path = output / "annotations" / "export_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["selected_record_count"] == 3
    assert summary["difficult_box_count"] == 1
    assert summary["category_counts"] == {"bag": 1, "basket": 1}
    # Images were copied into split folders.
    image_count = sum(
        len(list((output / "images" / split).glob("*.jpg")))
        for split in ("train", "val", "test")
    )
    assert image_count == 3


def test_build_coco_split_bbox_clipping(tmp_path):
    xml_dir, image_dir, _ = _build_workdir(tmp_path)
    annotation = parse_voc_annotation(
        xml_dir / "frame_0001.xml",
        image_dir=image_dir,
        labels=LABELS,
        keep_difficult=True,
    )
    coco = build_coco_split(
        [annotation],
        split="train",
        output_image_root=tmp_path / "images_out",
        labels=LABELS,
        overwrite=True,
    )
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 2
    assert (tmp_path / "images_out" / "train" / "frame_0001.jpg").exists()
