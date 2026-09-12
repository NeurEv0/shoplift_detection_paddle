# 2026-09-09 v2exp + classw102 回合记录(治 cc015 贴篮背包幻影)

完整背景/验收见 `docs/stage_b_模型版本与最终状态记录.md` §8。本文记录本回合可复现命令。

## 1. 新增标注(cc015_window_v2exp)
- 帧抽取:cc015 f375-1440 stride15 共 72 帧,放 `datasets_annotation/container_det/cc015_window_v2exp/images/full/`,
  清单 `full.csv`(image_path 带 full/ 前缀,frame_id/timestamp_ms 正确)。
- 标注:本地 labelImg(PascalVOC),类别文件 = 9 类 label_list.txt 顺序;
  XML 存 `annotations/`(文件名=图片名)。
- 规则:f375-585 标真背包;f585-1440 不标背包(入篮被遮挡),只标可见容器;空帧存空 XML。

## 2. 合并进 v2exp(只进 train)
```bash
# .tmp/merge_v2exp.py(已执行): 解析 72 XML -> 追加进 datasets/container_det_v2exp/annotations/instances_train.json
# 60 帧新增(12 帧与旧 train 重复跳过); val/test 不动。
```

## 3. 训练
```bash
cd /home/e-ai2/ZHITAI_2tb/shoplift_detection_paddle/src/PaddleDetection-release-2.9
mkdir -p ../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det_classw102
PYTHONPATH=. nohup /home/e-ai2/ZHITAI_2tb/conda_envs/shoplift-paddle/bin/python tools/train.py \
  -c ../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det_classw102.yml \
  --eval > ../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det_classw102/train.log 2>&1 &
```
- 权重 [1,2,1.2,1,1,1,1,1,1];TrainDataset -> v2exp;allow_empty:false(旧 json 有 19 张 D03 零标注图,true 会崩)。
- best @ep99, val AP 0.699(同 val 26 图,classw101 0.709)。

## 4. 类别 AP 门控(同 val,--classwise)
```bash
cd /home/e-ai2/ZHITAI_2tb/shoplift_detection_paddle/src/PaddleDetection-release-2.9
P=/home/e-ai2/ZHITAI_2tb/conda_envs/shoplift-paddle/bin/python
$P tools/eval.py -c ../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det_classw102.yml \
  -o weights=../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det_classw102/best_model.pdparams --classwise
$P tools/eval.py -c ../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det_classw101.yml \
  -o weights=../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det_classw101/best_model.pdparams --classwise
```
- eval.py 只接受 .pdparams/.pdopt/.pdmodel,不接受 .pdema。
- handbag 0.785->0.796(门控>2pt 未触发 PASS);mAP 0.709->0.696。

## 5. 导出
```bash
cd /home/e-ai2/ZHITAI_2tb/shoplift_detection_paddle/src/PaddleDetection-release-2.9
PYTHONPATH=. /home/e-ai2/ZHITAI_2tb/conda_envs/shoplift-paddle/bin/python tools/export_model.py \
  -c ../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det_classw102.yml \
  -o weights=../../outputs/wqh/container_det/rtdetr_r50vd_6x_container_det_classw102/best_model.pdema \
  -o output_dir=../../models/shoplift/item_container/rtdetr_r50vd_6x_container_det_classw102
```

## 6. 场景回归(classw102 vs classw101 基线)
```bash
cd /home/e-ai2/ZHITAI_2tb/shoplift_detection_paddle
SHOPLIFT_USER=wqh nohup /home/e-ai2/ZHITAI_2tb/conda_envs/shoplift-paddle/bin/python -m shoplift.cli.video_infer_visualize \
  --config shoplift/configs/pipeline.cc015_v7b_classw102_test.yml \
  --input ./datasets/test/test_videos/yulong_store/cc015bb3e050a8bc67ebbb2e6fd9951d.mp4 &
# d03 同理用 pipeline.d03_v7b_m80_classw102_test.yml + D03_堂食区_修复版.mpg
# 对比: python3 /tmp/compare_cw101_vs_cw102.py cc015|d03 (events diff + 窗口 source_category 统计)
```
- 结果:cc015 贴篮 backpack 4862->791(-84%),幻影窗口 37-51s 743->1;事件 64->66
  (+person-71 HIGH concealment 0.891@23040ms = 规则自重叠假阳性,非真事件,见 stage_b §8.4.1;
  +person-339 LOW 正常豁免);
  D03 57 条事件逐条不变。
