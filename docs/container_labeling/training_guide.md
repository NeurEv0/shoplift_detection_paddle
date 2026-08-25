# 容器检测训练指南（RT-DETR-R50VD 基线）

标注完成后，从标注产物到训练基线的完整流程。适用于第一批数据（278 张 / 1371 框）。

## 0. 当前数据状况（第一批，yulong_store_stride90）

- 图片 278 张，XML 全覆盖；负样本（空图）21 张，占比 7.6%
- 各类框数：plastic_bag 509 / cart 451 / handbag 365 / basket 24 / backpack 12 / bag 4 / stroller 3 / **suitcase 0 / helmet 0**
- 决策：**先训练一轮基线看结果**（suitcase/helmet 零样本，预期 AP≈0；后续补标迭代）
- 转换已验证通过（train 222 图 / 1115 框；val 56 图 / 253 框）；COCO 数据最终放 `datasets/container_det/`（文件内路径已统一为相对项目根的相对路径）

## 1. 数据落地到 datasets/container_det（在项目根目录执行，路径均为相对项目根的相对路径）

`datasets/` 为服务器挂载目录，`--output` 指到它即写入服务器：

```powershell
python scripts\labelimg_voc_to_coco.py `
  --xml-dir "datasets_annotation/container_det/yulong_store_stride90/annotations" `
  --image-dir "datasets_annotation/container_det/yulong_store_stride90/images/full" `
  --label-list "datasets\container_det\label_list.txt" `
  --output "datasets\container_det" --val-ratio 0.2 --include-empty
```

生成（与 `container_det_detection.yml` 的 dataset_dir 对应）：

```text
datasets/container_det/
  images/train/  images/val/          # 图片（test 划分当前为空，配置未用）
  annotations/instances_train.json    # 训练标注
  annotations/instances_val.json      # 验证标注
  annotations/export_summary.json     # 核对报告
```

转完先看 `export_summary.json`，与 §0 的数字一致再进训练。

## 2. 服务器环境确认（SSH 到 ubuntu@10.200.10.10）

```bash
nvidia-smi                                          # 确认 GPU 与显存（决定 batch_size）
conda run -n shoplift-paddle python -c "import paddle; print(paddle.__version__)"
ls /home/ubuntu/data_1t/shoplift_detection_paddle/src/PaddleDetection-release-2.9/tools/train.py
```

> ⚠️ 配置的 `_BASE_` 引用了 `../../../../src/PaddleDetection-release-2.9/configs/...`（相对配置文件解析），
> 服务器上必须有**同结构**的 PaddleDetection 2.9 checkout，否则训练直接报错。

## 3. 训练（服务器执行）

> ⚠️ **必须在 `src/PaddleDetection-release-2.9` 目录下运行**：
> 配置里 `dataset_dir: ../../datasets/container_det` 是相对运行目录（CWD）解析的，
> 在仓库根目录或别处跑会找不到数据。

```bash
cd /home/ubuntu/data_1t/shoplift_detection_paddle/src/PaddleDetection-release-2.9
conda activate shoplift-paddle
mkdir -p ../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det
python tools/train.py -c ../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det.yml \
  --eval --use_vdl=True \
  --vdl_log_dir=../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/vdl \
  2>&1 | tee ../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/train.log
```

要点：

- 预训练权重自动从 bcebos 下载（rtdetr_r50vd_6x_coco.pdparams），服务器需能访问外网
- 配置：36 epoch、batch_size 4（16G 显存方案 A；显存不足改 2 + warmup 200）、warmup 100、lr 5e-5、`snapshot_epoch: 1`（每 epoch 存一次）
- TrainDataset 已开 `allow_empty: true`（负样本参与训练）；EvalDataset 原本就有
- 训练产物：`../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/`（相对运行目录上两级 = 项目总目录 `outputs/wqh/...`；含每 epoch 权重、`best_model`/`model_final`、`train_result.json`，产物不落 git）
- 可视化监测：另开一个 SSH 窗口执行 `visualdl --logdir ../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/vdl --port 8040`，浏览器打开 http://10.200.10.10:8040

## 4. 评估（服务器执行）

```bash
cd /home/ubuntu/data_1t/shoplift_detection_paddle/src/PaddleDetection-release-2.9
python tools/eval.py -c ../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det.yml \
  -o weights=../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/best_model.pdparams
```

看每类 AP 与 mAP：plastic_bag / cart / handbag 应有明显检出；suitcase / helmet 预期 ≈0；bag / backpack / stroller / basket 偏低属预期。

## 5. 导出与部署

```bash
python tools/export_model.py -c ../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det.yml \
  -o weights=../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/best_model.pdparams
```

导出产物给 `pipeline.container_rtdetr_baseline.yml`（model_dir: ./models/shoplift/item_container/rtdetr_r50vd_6x_container_det）使用。

## 6. 数据迭代（基线之后再补）

1. 从其他序列/帧段采样包含 suitcase / helmet / backpack / bag / stroller / basket 的帧
2. 目标：每类 ≥100 框（规范 §6.2：理想 ≥300）；负样本保持 10%~20%
3. 标注（本地或服务器）→ 转换（`--include-empty`）→ 在基线权重上继续训练/微调
