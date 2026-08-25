# -*- coding: utf-8 -*-
"""从原始 VOC XML 直接生成 container_det COCO 标注并 8:1:1 划分。

用法（在项目根目录下执行）:
    python scripts/split_container_det_881_xml.py

行为:
1. 解析 datasets_annotation/container_det/yulong_store_stride90/ 下所有 VOC XML
2. 类别表从 instances_train.json.bak_881 继承（保证 9 类 id 顺序与已训练模型一致）
3. 按 8:1:1 随机划分（seed=42 可复现），写出 instances_train/val/test.json
4. 把图像文件移动到 images/train | images/val | images/test
"""
import json
import os
import random
import shutil
import glob
import xml.etree.ElementTree as ET
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_DIR = os.path.join(ROOT, 'datasets_annotation', 'container_det',
                       'yulong_store_stride90')
ANN = os.path.join(ROOT, 'datasets', 'container_det', 'annotations')
IMG = os.path.join(ROOT, 'datasets', 'container_det', 'images')
SEED = 42


def main():
    # 类别表：从旧 json 继承（9 类，id 顺序与已训练模型一致）
    cats_src = os.path.join(ANN, 'instances_train.json.bak_881')
    if not os.path.exists(cats_src):
        cats_src = os.path.join(ANN, 'instances_train.json')
    cats = json.load(open(cats_src, encoding='utf-8'))['categories']
    name2id = {c['name']: c['id'] for c in cats}
    print('categories ({}): {}'.format(len(cats),
                                       [c['name'] for c in cats]))

    images = []
    annotations = []
    ann_id = 0
    ignored_names = set()
    for img_id, xml_path in enumerate(sorted(glob.glob(os.path.join(XML_DIR, '*.xml')))):
        root = ET.parse(xml_path).getroot()
        fname = os.path.basename(root.findtext('filename')
                                 or root.findtext('path') or '')
        w = int(root.findtext('size/width') or 0)
        h = int(root.findtext('size/height') or 0)
        images.append({
            'id': img_id,
            'file_name': fname,
            'width': w,
            'height': h
        })
        for obj in root.findall('object'):
            name = obj.findtext('name')
            if name not in name2id:
                ignored_names.add(name)
                continue
            b = obj.find('bndbox')
            x1 = float(b.findtext('xmin'))
            y1 = float(b.findtext('ymin'))
            x2 = float(b.findtext('xmax'))
            y2 = float(b.findtext('ymax'))
            bw = max(x2 - x1, 0.)
            bh = max(y2 - y1, 0.)
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': name2id[name],
                'bbox': [x1, y1, bw, bh],
                'area': bw * bh,
                'iscrowd': 0,
            })
            ann_id += 1

    print('XML parsed: {} images, {} annotations, ignored obj names: {}'.
          format(len(images), len(annotations), sorted(ignored_names)))

    dups = {k: v for k, v in Counter(im['file_name']
                                     for im in images).items() if v > 1}
    if dups:
        print('WARN duplicate filenames:', dups)

    # 8:1:1 划分
    random.seed(SEED)
    ids = list(range(len(images)))
    random.shuffle(ids)
    n = len(ids)
    n_tr = int(n * 0.8)
    n_va = int(n * 0.1)
    split_ids = {
        'train': set(ids[:n_tr]),
        'val': set(ids[n_tr:n_tr + n_va]),
        'test': set(ids[n_tr + n_va:]),
    }
    print('total: {} -> train {} | val {} | test {}'.format(
        n, len(split_ids['train']), len(split_ids['val']),
        len(split_ids['test'])))

    for split in ('train', 'val', 'test'):
        sid = split_ids[split]
        out_imgs = [im for im in images if im['id'] in sid]
        out_anns = [a for a in annotations if a['image_id'] in sid]
        out = {
            'info': {},
            'licenses': [],
            'images': out_imgs,
            'annotations': out_anns,
            'categories': cats,
        }
        with open(os.path.join(ANN, 'instances_{}.json'.format(split)),
                  'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)

        # 移动图像文件（在 images/ 全目录下搜索）
        dst_dir = os.path.join(IMG, split)
        os.makedirs(dst_dir, exist_ok=True)
        empty = missing = 0
        for im in out_imgs:
            fname = im['file_name']
            src = None
            for root, _, files in os.walk(IMG):
                if fname in files:
                    src = os.path.join(root, fname)
                    break
            if src is None:
                missing += 1
                print('[warn] missing image file:', fname)
                continue
            dst = os.path.join(dst_dir, fname)
            if src != dst:
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
            if not any(a['image_id'] == im['id'] for a in out_anns):
                empty += 1
        print('{}: {} images ({} empty, {} missing-file), {} annotations'.
              format(split, len(out_imgs), empty, missing, len(out_anns)))

    print('done.')


if __name__ == '__main__':
    main()
