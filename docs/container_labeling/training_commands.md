# 容器检测训练/评估/测试命令速查

> 适用于 RT-DETR-R50VD + 9 类容器检测（bag, backpack, handbag, suitcase, basket, cart, plastic_bag, stroller, helmet）

## 0. 环境与路径

- 服务器项目根：`/home/ubuntu/data_1t/shoplift_detection_paddle`
- **所有命令在 `src/PaddleDetection-release-2.9` 目录下执行**（config 里的相对路径基于此）
- conda 环境：`shoplift-paddle`（`conda activate shoplift-paddle`）
- 输出目录：`outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/`（已被 gitignore）
- 数据集：`datasets/container_det/`，8:1:1 划分（train 222 图/1083 框、val 27/141、test 29/147）；XML 源在 `datasets_annotation/container_det/yulong_store_stride90/annotation/`
- 配置：`shoplift/configs/paddledetection/container_det/container_det_detection.yml`（数据集）+ `rtdetr_r50vd_6x_container_det.yml`（模型/训练）
- 模型：RT-DETR-R50VD（42M 参数，136 GFLOPs），COCO 预训练 + 9 类微调

## 1. 数据划分（重新划分时）

```bash
cd /home/ubuntu/data_1t/shoplift_detection_paddle
python scripts/split_container_det_881_xml.py
```

从 XML 直接生成 `instances_train/val/test.json`（8:1:1，随机划分）。

## 2. 训练

**从头训练**（新数据 / 清空输出后）：

```bash
cd /home/ubuntu/data_1t/shoplift_detection_paddle/src/PaddleDetection-release-2.9
python tools/train.py -c ../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det.yml \
  --eval \
  -o epoch=60 weights= RTDETRTransformer.num_denoising=0 \
  2>&1 | tee ../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/train.log
```

**续训**（从第 N 轮继续）：

```bash
cd /home/ubuntu/data_1t/shoplift_detection_paddle/src/PaddleDetection-release-2.9
python tools/train.py -c ../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det.yml \
  --eval \
  --resume ../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/{N} \
  -o epoch={更大的值} RTDETRTransformer.num_denoising=0 \
  2>&1 | tee -a ../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/train.log
```

**必带参数（血泪坑）**：

| 参数 | 原因 |
|---|---|
| `-o epoch=60`（**单数**，config 键是 `epoch`） | 写 `epochs`（复数）无效 → 静默退出 |
| `weights=`（置空） | config 里 weights 指向旧 model_final，会误加载 |
| `RTDETRTransformer.num_denoising=0` | paddle 3.x 下 denoising 分支报 13.7EB 内存 bug，必须关 |
| `--resume .../{N}` | 前缀不带后缀；从 `.pdopt` 的 last_epoch 恢复；末轮用 `model_final` |

**检查点命名**：每轮存 `{N}.pdparams/.pdopt/.pdema/.pdstates`，最后一轮额外存 `model_final.*`；`best_model.*` = val mAP 最高轮（当前训练为 epoch 496，mAP 0.689）。

## 3. 评估

**完整评估**（mAP/AP50/AP75，val_loss 写入 TensorBoard）：

```bash
cd /home/ubuntu/data_1t/shoplift_detection_paddle/src/PaddleDetection-release-2.9
python tools/eval.py -c ../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det.yml \
  -o weights=../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/best_model.pdparams
```

（或训练时 `--eval` 每轮自动评估）

**逐类 AP**（eval 后会生成 `bbox.json`，用 pycocotools 算，几秒钟）：

```bash
cd /home/ubuntu/data_1t/shoplift_detection_paddle/src/PaddleDetection-release-2.9
python - <<'EOF'
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt_path = '../../datasets/container_det/annotations/instances_val.json'
dt_path = 'bbox.json'

coco_gt = COCO(gt_path)
coco_dt = coco_gt.loadRes(dt_path)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()

p = coco_eval.eval['precision']          # [iou(10), recall(101), K, area(4), maxDets(3)]
cat_ids = coco_eval.params.catIds
names = {c['id']: c['name'] for c in json.load(open(gt_path))['categories']}

def mean_ignoring_nan(x):
    x = x[np.isfinite(x)]
    return x.mean() if x.size else float('nan')

print('%-14s %8s %8s %8s' % ('class', 'mAP', 'AP50', 'AP75'))
for k, cid in enumerate(cat_ids):
    a = p[:, :, k, 0, -1]
    print('%-14s %8.4f %8.4f %8.4f' % (names.get(cid, cid),
          mean_ignoring_nan(a), mean_ignoring_nan(a[0]), mean_ignoring_nan(a[5])))
EOF
```

## 4. 测试 / 可视化

```bash
cd /home/ubuntu/data_1t/shoplift_detection_paddle/src/PaddleDetection-release-2.9
python tools/infer.py \
  -c ../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det.yml \
  -o weights=../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/best_model.pdparams \
  --infer_dir=../../datasets/container_det/images/test \
  --output_dir=../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/infer_test_best \
  --draw_threshold=0.3
```

- `--draw_threshold`：0.3 诊断用（暴露边缘检测/误检），0.5 展示用（只看高把握框）
- 换 `model_final.pdparams` 可对比最终轮

## 5. TensorBoard

```bash
nohup tensorboard \
  --logdir /home/ubuntu/data_1t/shoplift_detection_paddle/outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/tb \
  --port 6006 --host 0.0.0.0 \
  > /home/ubuntu/data_1t/shoplift_detection_paddle/outputs/wqh/container_det/rtdetr_r50vd_6x_container_det/tensorboard.log 2>&1 &
```

浏览器：`http://10.200.10.10:6006`。曲线：`train/loss*`、`eval/val_loss`、`eval/bbox-mAP/AP50/AP75`（每 epoch 一个点）。

## 6. 调优注意事项（重要背景）

1. **数据泄露（已证实）**：两个视频切帧随机划分 → val/test 100% 在 train 有近重复帧（像素 MAE<0.15）。**val 的 loss/mAP 均失真**，只能横向对比，不能当真实泛化。跨视频划分评估干净但 mAP 明显低（组长已否决）。
2. **类别极度不均衡**（train 框数）：plastic_bag 403 / cart 359 / handbag 284 / basket 21 / backpack 12 / stroller 3 / **bag 1** / **suitcase 0** / **helmet 0**。当前 mAP 0.689 = 5 个有 GT 类的平均（bag 0.006 拖低），backpack/suitcase/stroller/helmet 在 val 无 GT 无法评估。**提升重点 = 补稀有类标注**（helmet/suitcase/bag/stroller/backpack）+ 明确 bag/handbag 标注边界。
3. **已 patch 的源码**（本地 + 服务器已同步）：`callbacks.py`（TensorBoardWriter + 每轮检查点 + VdlWriter 全局 step）、`trainer.py`（eval_with_loss 计算 val_loss + TensorBoardWriter 注册）、`utils.py`（deformable_attention PIR 兼容）、`detr_head.py`/`rtdetr_transformerv3.py`（RT-DETRv3 o2m patch）。
4. **训练配置要点**：base_lr 5e-5、milestones [24,32]（30 轮后 lr=1e-6，后期提升极慢）、warmup 100 步、EMA、log_iter 50、snapshot_epoch 1；`eval_with_loss: True` + `use_tensorboard: True` 已写进 config。
5. **磁盘**：每轮检查点约 700MB，长训注意 `df -h`。
