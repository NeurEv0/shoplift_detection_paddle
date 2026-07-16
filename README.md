# Shoplift Detection Paddle

本项目是在 PaddleDetection 能力底座之上建设的商超可疑拿取/藏匿行为分析开发区。项目目标不是自动认定盗窃，而是从门店视频中提取可解释的视觉证据，输出可供全栈告警系统和人工复核系统消费的结构化风险事件。

## 项目定位

开发范围为算法与规则侧的六个核心能力：

1. 人员检测/跟踪门控
2. 人体关键点/手部区域分析
3. 商品与容器检测
4. 手-商品-容器轨迹关联
5. 可疑动作识别
6. 风险评分与规则校验

## 开发环境

推荐使用独立 Conda 环境：

```powershell
conda env create -f environment.yml
conda activate shoplift-paddle
```

如果只安装运行时依赖：

```powershell
pip install -r requirements.txt
pip install -e src/PaddleDetection-release-2.9
```

GPU 推理环境需要按本机 CUDA 版本单独安装匹配的 `paddlepaddle-gpu`，再安装其余依赖。模型权重、样本视频、输出和缓存目录都通过配置文件引用，不提交到仓库。

## 环境自检

复制并调整示例配置后执行自检：

```powershell
Copy-Item shoplift/configs/env.example.yml shoplift/configs/env.local.yml
python scripts/check_env.py --config shoplift/configs/env.local.yml
```

自检覆盖 Python 版本、`paddle`、GPU 可见性、`cv2`、`ppdet` 路径和关键配置文件。CPU 基础环境可以直接运行不依赖模型权重的单元测试：

```powershell
python -m unittest discover -s shoplift/tests
```

## 离线 CLI

可以先用 dry-run 验证 CLI 参数和配置入口：

```powershell
python -m shoplift.cli.offline_analyze --config shoplift/configs/pipeline.example.yml --input data/samples/demo.mp4 --output outputs/shoplift --dry-run
```

离线分析支持本地视频或帧目录，输出逐帧 JSONL、空事件文件和调试可视化：

```powershell
python -m shoplift.cli.offline_analyze --config shoplift/configs/pipeline.example.yml --input data/samples/demo.mp4 --output outputs/shoplift
python -m shoplift.cli.offline_analyze --config shoplift/configs/pipeline.example.yml --input data/samples/frames --output outputs/shoplift
```

当前 P0 CLI 使用 model-free 后端跑通输入、门控、手部 ROI/商品容器结果承载、JSONL 写出和 debug 可视化。真实 PaddleDetection 推理结果可通过后续后端接入同一逐帧结构。

真实 PaddleDetection/PP-Human 后端已接入为可选 backend。先在 `shoplift/configs/pipeline.example.yml` 的 `backend` 段配置本地导出的 PP-Human MOT、关键点和可选商品/容器检测模型目录，再执行：

```powershell
python -m shoplift.cli.offline_analyze --config shoplift/configs/pipeline.example.yml --backend paddledet_pphuman --input data/samples/demo.mp4 --output outputs/shoplift_pphuman
```

如果只想检查参数和路径入口，不加载 Paddle 或模型：

```powershell
python -m shoplift.cli.offline_analyze --config shoplift/configs/pipeline.example.yml --backend paddledet_pphuman --input data/samples/demo.mp4 --dry-run
```

`events.json` 现在由 P1 事件引擎生成；没有商品/容器检测权重时，后端只会输出人员跟踪和可选手部 ROI，不会伪造藏匿事件。

## DCSASS 初版评测

本地 `datasets/DCSASS_Shoplifting` 可通过 clip-level 评测入口批量验证：

```powershell
python -m shoplift.eval.dcsass_eval --config shoplift/configs/pipeline.example.yml --dataset-root datasets/DCSASS_Shoplifting --output outputs/dcsass_eval --backend paddledet_pphuman --no-debug
```

建议先跑一个带可视化的小批量 smoke eval，输出集中保留在清晰路径下：

```powershell
python -m shoplift.eval.dcsass_eval --config shoplift/configs/pipeline.local.yml --dataset-root datasets/DCSASS_Shoplifting --output outputs/public_eval/dcsass_smoke --backend paddledet_pphuman --max-clips-per-label 2 --max-frames 10 --frame-stride 3 --continue-on-error
```

输出包括 `metrics.json`、`predictions.csv` 和每个 clip 的逐帧结果/事件文件；启用 debug 时，`predictions.csv` 会记录对应的 `debug_visualization.mp4` 路径。DCSASS 只有 clip 级二分类标签，因此该评测衡量端到端异常信号，不能单独证明手-商品-容器关系是否定位正确。

## 数据契约

- 核心数据结构：`shoplift/core/types.py`
- 跟踪类型兼容导出：`shoplift/tracking/track_types.py`
- PaddleDetection 适配器：`shoplift/adapters/paddledet_adapter.py`
- 人员门控：`shoplift/vision/person_gate.py`
- 手部 ROI：`shoplift/vision/pose_hand.py`
- 商品/容器检测适配：`shoplift/vision/object_container.py`
- 关系关联：`shoplift/tracking/association.py`
- 事件引擎：`shoplift/events/event_engine.py`
- 风险评分：`shoplift/rules/risk_score.py`
- 规则校验：`shoplift/rules/validators.py`
- 事件 JSON Schema：`shoplift/events/risk_event.schema.json`
- 事件示例：`shoplift/events/examples/risk_event.example.json`
- 契约说明：`shoplift/configs/schema.md`
- 离线 CLI：`shoplift/cli/offline_analyze.py`

PaddleDetection 适配器当前支持三类 P0 输出转换：

- 普通检测：`[class_id, score, x1, y1, x2, y2]` -> `DetectionBox`
- PP-Human/MOT：`[track_id, class_id, score, x1, y1, x2, y2]` 或 SDE `online_tlwhs/scores/ids` -> `Tracklet`
- 关键点：PP-Human `{"keypoint": [keypoints, scores]}` -> `HandRegion`

人员门控当前可从 `DetectionBox`、`Tracklet` 或适配器的帧结果中判断是否有人；无人帧返回 `skipped_heavy_modules=true`，同时累计 `skip_rate` 和 `trigger_rate`。

`pose_hand` 模块基于 COCO wrist/elbow 关键点生成左右手 ROI，支持低置信度 wrist 过滤，并将 `HandRegion.person_track_id` 绑定到对应人员轨迹。

`object_container` 模块将 `product/backpack/handbag/cart/pocket_region` 等模型类别归一到 `item/bag/basket/clothing_region` 等粗粒度内部类别，输出商品、容器和扩展区域分组结果。

P1 事件引擎当前支持 `bag_concealment`、`clothing_concealment`、`special_container_concealment`、`bulk_pickup_to_bag` 和 `near_body_suspicious`。风险评分由动作类型、容器类型、连续证据、模型置信度、区域风险和正常购物解释共同决定；规则校验会防止单帧接触、低可见度和购物篮/购物车正常放入被误升为高风险。

## 参考文档

- [技术方案](docs/shoplifting_detection_technical_solution.md)
- [PaddleDetection 能力支撑与补齐项分析](docs/paddledetection_capability_support_analysis.md)
- [开发路线 TODO](docs/TODO.md)
