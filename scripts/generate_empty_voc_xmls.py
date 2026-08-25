"""为"无容器"图片批量生成空 PascalVOC XML（负样本兜底工具）。

背景：labelImg 只对"改过（dirty）"的图自动保存；空图（无任何容器）
没画框 = 没改动，翻页时不会生成 XML。而负样本（无容器帧）是训练必需
（规范 §6.2：占总图 10%~20%，配合转换脚本 --include-empty 进入 COCO）。
本脚本扫描图片目录，找出没有对应 XML 的图片，确认后批量生成"空 XML"
（只含 <size>、不含 <object>），供转换脚本识别为负样本。

用法：
    # 只列出缺 XML 的图片（不写任何文件）
    python scripts/generate_empty_voc_xmls.py --image-dir <图片目录> --xml-dir <XML目录>

    # 确认这些图确实无容器后，批量生成空 XML
    python scripts/generate_empty_voc_xmls.py --image-dir <图片目录> --xml-dir <XML目录> --generate

注意：--generate 会把"所有没有 XML 的图片"都当成负样本；若其中混有
"忘了标注"的图，请先回去补框，再执行生成。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import xml.etree.ElementTree as ET

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _image_size(path: pathlib.Path) -> tuple[int, int]:
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("需要 Pillow：python -m pip install Pillow") from exc
    with Image.open(path) as im:
        return im.size  # (width, height)


def _empty_xml(image_path: pathlib.Path, xml_dir: pathlib.Path) -> pathlib.Path:
    width, height = _image_size(image_path)
    folder = image_path.parent.name or "."
    annotation = ET.Element("annotation")
    folder_el = ET.SubElement(annotation, "folder")
    folder_el.text = folder
    filename_el = ET.SubElement(annotation, "filename")
    filename_el.text = image_path.name
    path_el = ET.SubElement(annotation, "path")
    path_el.text = str(image_path)
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    size = ET.SubElement(annotation, "size")
    w = ET.SubElement(size, "width")
    w.text = str(width)
    h = ET.SubElement(size, "height")
    h.text = str(height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"
    ET.indent(annotation)
    out = xml_dir / (image_path.stem + ".xml")
    ET.ElementTree(annotation).write(out, encoding="utf-8", xml_declaration=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=pathlib.Path, required=True, help="图片目录")
    parser.add_argument("--xml-dir", type=pathlib.Path, required=True, help="XML 输出目录")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="批量生成空 XML（默认只列出缺 XML 的图片，不写文件）",
    )
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        raise SystemExit(f"图片目录不存在：{args.image_dir}")

    images = sorted(
        p
        for p in args.image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    missing = [p for p in images if not (args.xml_dir / (p.stem + ".xml")).exists()]

    print(f"图片 {len(images)} 张；已有 XML {len(images) - len(missing)}；缺 XML（候选负样本）{len(missing)} 张")
    for p in missing:
        print("  -", p.name)

    if not args.generate:
        print("\n以上图片没有对应 XML。确认这些帧确实无容器后，加 --generate 生成空 XML；")
        print("若有忘了标注的图，请先在 labelImg 里补框。")
        return 0

    if not missing:
        print("没有需要生成的 XML。")
        return 0

    args.xml_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for image in missing:
        _empty_xml(image, args.xml_dir)
        created += 1
    print(f"\n已生成 {created} 个空 XML 到 {args.xml_dir}")
    print("转换时请带 --include-empty，负样本才会进入 COCO。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
