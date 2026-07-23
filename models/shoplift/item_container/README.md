# Item/Container Detection Model

This directory is reserved for exported PaddleDetection inference models used by `backend.item_container`.

Recommended baseline export:

```powershell
cd src/PaddleDetection-release-2.9
python tools/export_model.py -c ../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det.yml -o weights=output/rtdetr_r50vd_6x_container_det/best_model --output_dir=../../models/shoplift/item_container
```

Expected exported model directory:

```text
models/shoplift/item_container/rtdetr_r50vd_6x_container_det/
  infer_cfg.yml
  model.pdiparams
  model.pdiparams.info
  model.pdmodel
```

When the baseline is exported with the nine container-only classes, configure the backend with:

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
