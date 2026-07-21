# Person Attribute Pretrained Weights

Put local pretrained `.pdparams` files here. Binary weight files are intentionally
not committed to the repository.

Recommended options:

| File | Intended source | Notes |
|---|---|---|
| `pphgnetv2_s_pretrained.pdparams` | PaddleDetection/PaddleClas PPHGNetV2-S classification pretrained weights | Recommended default backbone for accuracy. |
| `pphgnetv2_n_pretrained.pdparams` | PaddleDetection/PaddleClas PPHGNetV2-N classification pretrained weights | Faster, lower capacity. |
| `lcnet_1_0_pretrained.pdparams` | PaddleDetection/PaddleClas PP-LCNet x1.0 pretrained weights | Lightweight baseline. |
| `PPLCNet_x1_0_person_attribute_945_infer/` | Existing PP-Human person-attribute inference model | Useful as a reference model, not directly compatible with the new six-head labels without fine-tuning/conversion. |

Training config path:

```yaml
backbone:
  pretrained: ./models/pretrained/person_attribute/pphgnetv2_s_pretrained.pdparams
```

The loader uses non-strict parameter matching, so backbone parameters can be
reused while the six shoplift-specific classification heads are initialized
from scratch.

