# Person Attribute Dataset Layout

This directory is a placeholder for training data and is not expected to be
committed with real images.

Recommended layout:

```text
datasets/person_attribute/
├── images/
│   ├── train/
│   └── val/
├── train.csv
└── val.csv
```

CSV columns:

```csv
image_path,left_hand_state,left_hand_visibility,right_hand_state,right_hand_visibility,body_orientation,occlusion_level
train/person_000001.jpg,holding_product,clear,empty,clear,side,light
```

Allowed labels:

| Field | Labels |
|---|---|
| `left_hand_state` / `right_hand_state` | `empty`, `holding_object`, `holding_product`, `uncertain` |
| `left_hand_visibility` / `right_hand_visibility` | `clear`, `partial_occluded`, `not_judgable` |
| `body_orientation` | `front`, `side`, `back`, `unknown` |
| `occlusion_level` | `none`, `light`, `heavy` |

Images should be person crops aligned with the tracked person bbox. The training
loader resizes each crop to `192x256` and applies ImageNet normalization.

To bootstrap labeling from videos, use:

```powershell
python scripts/prepare_person_attribute_dataset.py --input path/to/videos --output datasets/person_attribute/working --frame-jsonl outputs/.../frame_results.jsonl
```

The script writes crop images under `images/<split>/` and a CSV template with
`label_status=unreviewed` plus conservative placeholder labels for manual review.
