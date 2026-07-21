# Exported Person Attribute Model

This directory is reserved for exported inference models. Binary exports are not
committed.

Expected output after export:

```text
models/shoplift/person_attribute/inference/
├── inference.pdmodel
├── inference.pdiparams
└── infer_cfg.yml
```

The PP-Human backend can load this directory through:

```yaml
backend:
  person_attribute:
    enabled: true
    model_dir: ./models/shoplift/person_attribute/inference
```

