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

P0-9 会补齐真实视频/帧目录分析链路。当前 P0 1-4 阶段可先用 dry-run 验证 CLI 参数和配置入口：

```powershell
python -m shoplift.cli.offline_analyze --config shoplift/configs/pipeline.example.yml --input data/samples/demo.mp4 --output outputs/shoplift --dry-run
```

完整离线分析目标命令将保持同一入口：

```powershell
python -m shoplift.cli.offline_analyze --config shoplift/configs/pipeline.example.yml --input data/samples/demo.mp4 --output outputs/shoplift
```

## 数据契约

- 核心数据结构：`shoplift/core/types.py`
- 跟踪类型兼容导出：`shoplift/tracking/track_types.py`
- PaddleDetection 适配器：`shoplift/adapters/paddledet_adapter.py`
- 人员门控：`shoplift/vision/person_gate.py`
- 事件 JSON Schema：`shoplift/events/risk_event.schema.json`
- 事件示例：`shoplift/events/examples/risk_event.example.json`
- 契约说明：`shoplift/configs/schema.md`

PaddleDetection 适配器当前支持三类 P0 输出转换：

- 普通检测：`[class_id, score, x1, y1, x2, y2]` -> `DetectionBox`
- PP-Human/MOT：`[track_id, class_id, score, x1, y1, x2, y2]` 或 SDE `online_tlwhs/scores/ids` -> `Tracklet`
- 关键点：PP-Human `{"keypoint": [keypoints, scores]}` -> `HandRegion`

人员门控当前可从 `DetectionBox`、`Tracklet` 或适配器的帧结果中判断是否有人；无人帧返回 `skipped_heavy_modules=true`，同时累计 `skip_rate` 和 `trigger_rate`。

## 参考文档

- [技术方案](docs/shoplifting_detection_technical_solution.md)
- [PaddleDetection 能力支撑与补齐项分析](docs/paddledetection_capability_support_analysis.md)
- [开发路线 TODO](docs/TODO.md)
