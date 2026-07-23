# Container Detection Dataset

This folder is the project dataset root for the PaddleDetection RT-DETR container baseline.

## Layout

```text
datasets/container_det/
  annotations/
    instances_train.json
    instances_val.json
    instances_test.json
  images/
    train/
    val/
    test/
  label_list.txt
  container_labeling_spec.md
  label_studio_config.xml
  groundingdino_work/
    predictions.jsonl
    label_studio_tasks.json
    label_studio_config.xml
    summary.json
```

The annotation format is COCO detection. Category ids must be contiguous and start from 0 in PaddleDetection's internal loader order. Keep the order in `label_list.txt`:

| id | class |
|---:|---|
| 0 | `bag` |
| 1 | `backpack` |
| 2 | `handbag` |
| 3 | `suitcase` |
| 4 | `basket` |
| 5 | `cart` |
| 6 | `plastic_bag` |
| 7 | `stroller` |
| 8 | `helmet` |

## GroundingDINO Semi-Automated Labeling

The project pipeline is:

1. Extract CCTV frames into `datasets/container_det/<source>/images/full` and keep the frame manifest CSV.
2. Use GroundingDINO to create high-recall candidate boxes.
3. Review and correct the boxes in Label Studio.
4. Export the reviewed tasks to PaddleDetection COCO JSON.

Install GroundingDINO separately from its official repository and keep the
config/checkpoint paths local, for example:

```powershell
pip install -e path\to\GroundingDINO
```

Create prelabels from the existing Yulong frame manifest:

```powershell
python scripts/container_det_groundingdino_pipeline.py prelabel `
  --input datasets/container_det/yulong_store_stride90/full.csv `
  --output datasets/container_det/groundingdino_work `
  --config models/pretrained/groundingdino/GroundingDINO_SwinT_OGC.py `
  --weights models/pretrained/groundingdino/groundingdino_swint_ogc.pth `
  --device cuda `
  --box-threshold 0.28 `
  --text-threshold 0.22 `
  --save-visualizations `
  --overwrite
```

The command writes:

- `groundingdino_work/predictions.jsonl`: raw candidates for auditing or fallback export.
- `groundingdino_work/label_studio_tasks.json`: import this into Label Studio.
- `groundingdino_work/label_studio_config.xml`: project labeling interface.
- `groundingdino_work/visualizations/`: optional quick-look overlays.

For Label Studio local files, enable local serving and point the document root at
this repository:

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=(Get-Location).Path
label-studio
```

Create a Label Studio object-detection project, paste `label_studio_config.xml`
as the labeling interface, import `label_studio_tasks.json`, then review every
box against `container_labeling_spec.md`. Keep ambiguous supermarket shopping
bags as `bag` unless checkout context proves they are bulk `plastic_bag`.

After review, export Label Studio tasks as JSON and convert them to the final
COCO layout:

```powershell
python scripts/container_det_groundingdino_pipeline.py export-coco `
  --input outputs/container_det_label_studio_export.json `
  --output datasets/container_det `
  --val-ratio 0.2 `
  --test-ratio 0.0 `
  --overwrite
```

For a fast smoke test before human review, export trusted prelabels directly:

```powershell
python scripts/container_det_groundingdino_pipeline.py export-coco `
  --input datasets/container_det/groundingdino_work/predictions.jsonl `
  --input-format predictions-jsonl `
  --output datasets/container_det/smoke_coco `
  --drop-empty `
  --min-score 0.35 `
  --overwrite
```

Final training data should live at the top-level COCO layout shown above:
`images/train`, `images/val`, and `annotations/instances_train.json`,
`annotations/instances_val.json`.

## Baseline

Training config:

```powershell
cd src/PaddleDetection-release-2.9
python tools/train.py -c ../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det.yml --eval
```

The config loads COCO-trained RT-DETR-R50 weights and fine-tunes on this dataset:

```text
shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det.yml
shoplift/configs/paddledetection/container_det/container_det_detection.yml
```

Export the best checkpoint for project inference:

```powershell
cd src/PaddleDetection-release-2.9
python tools/export_model.py -c ../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det.yml -o weights=output/rtdetr_r50vd_6x_container_det/best_model --output_dir=../../models/shoplift/item_container
```

Then point `backend.item_container.model_dir` at the exported model directory and set:

```yaml
backend:
  item_container:
    enabled: true
    model_dir: ./models/shoplift/item_container/rtdetr_r50vd_6x_container_det
    threshold: 0.35
    batch_size: 1
    class_id_to_category:
      0: bag
      1: backpack
      2: handbag
      3: suitcase
      4: basket
      5: cart
      6: plastic_bag
      7: stroller
      8: helmet
```
