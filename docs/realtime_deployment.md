# 实时偷盗检测部署说明（realtime_infer）

> 本文档说明 `shoplift/cli/realtime_infer.py`（实时单流推理脚本）的模型权重、输入输出、监控指标与运行方式。
> 配套脚本：`shoplift/cli/monitor_dashboard.py`（浏览器实时监控面板）。

---

## 1. 概述

`realtime_infer` 把离线「两段式」链路合并成**一条流式单 pass**，从 USB 摄像头 / RTSP / 本地视频逐帧读取，跑完完整推理并持续采样资源占用，用于：

1. 实时偷盗事件检测（管线风险事件 + PP-TSM「拆包装」动作融合）；
2. 测量**单流**的 GPU 显存 / CPU 内存 / 时延 / FPS，据此外推 **10 流超市部署**的硬件配置。

每次启动会**自动新建带时间戳的输出目录**（`single_stream_YYYYMMDD_HHMMSS/`），不会覆盖上一次结果。

---

## 2. 单帧推理流水线

```
stream frame (source.next)
  ├─ PP-Human MOT        (行人检测 + 跟踪)          → person_tracks
  ├─ keypoint            (姿态 + 手部 ROI)          → body_poses / hand_regions
  ├─ person attribute    (手持状态 holding_product)  → proxy_item_regions
  ├─ item container det  (容器/包裹检测)            → items / containers
  ├─ 管线风险事件引擎      (RiskEvent)               → near_body_suspicious 等
  ├─ 增量 PP-TSM open    (人像 16 帧滑窗 @3.125fps)  → open 概率 → 事件化
  └─ 商品门 + 融合        (FusionEngine)             → package_opening 事件
```

---

## 3. 使用的模型与权重

| 模块 | 模型目录 / 权重 | 说明 |
|---|---|---|
| 行人检测 + 跟踪 (MOT) | `models/paddledetection/person_mot` | PP-Human 里的 MOT，静态图 predictor |
| 关键点姿态 | `models/paddledetection/person_keypoint` | COCO 17 点骨架，`derive_hand_regions=true` 派生手部 ROI |
| 人员属性（手持状态） | `models/shoplift/person_attribute/inference_hi384_v7_best` | 预测 `left/right_hand_state`（`holding_product` 手持商品），输入 384×512 |
| 容器检测 | `models/shoplift/item_container/rtdetr_r50vd_6x_container_det_classw102` | **9 类闭集容器检测**：bag / backpack / handbag / suitcase / basket / cart / plastic_bag / stroller / helmet |
| PP-TSM「拆包装」动作 | 配置 `shoplift/configs/paddlevideo/pptsm_open_v2_fight16.yaml`<br>权重 `outputs/wqh/paddlevideo/pptsm_open_v2_fight16/ppTSM_epoch_00026.pdparams` | 生产版 ep26，二分类 open/not-open，dygraph 动态图 |

**管线配置**：`shoplift/configs/pipeline.cc015_v7b_classw102_test.yml`
**融合配置**：`shoplift/configs/fusion.example.yml`（theta=0.6，min_consecutive=3，min_product_confirm_frames=14，timeline_fps=3.125，win=16，stride=2，crop_padding=0.15）

> 注意：`person_attribute`（人员属性）**不给物体分类**，它只判断「手是否在拿商品」（`holding_product` 状态）；把物体标成 `basket` 等类别的是 `item_container` 容器检测模型，该模型没有「水杯/商品/item」类，只能把非容器物体硬套进 9 个容器类别。

---

## 4. 输入（`--source`）

| 形式 | 写法 | 说明 |
|---|---|---|
| USB 摄像头索引 | `--source 0`（或 `/dev/video0`） | 训练服务器上的 USB 摄像头 |
| 网络流 | `--source rtsp://user:pass@ip:554/stream` | 也支持 `http://`、`rtmp://` |
| **本地视频文件** | `--source datasets/test/test_videos/xxx.mp4` | 用于**回放/复现**，测试阶段强烈建议先用视频验证 |

其它常用参数：
- `--camera-id`：摄像头标识（写进事件，如 `realtime-cam-001`）
- `--duration N`：运行 N 秒后停止（不填则一直跑）
- `--max-frames N`：处理 N 帧后停止
- `--frame-stride N`：跳帧（默认 1）
- `--source-fps N`：自动探测 fps 失败时手动指定

---

## 5. 输出

每次启动新建目录 `outputs/realtime/single_stream_YYYYMMDD_HHMMSS/`（同秒冲突追加 `_1`、`_2`），启动时打印 `[output-dir] <实际路径>`。

| 文件 | 说明 |
|---|---|
| `metrics.csv` | 每帧资源 / 时延样本（面板 + 分析用） |
| `summary.json` | 模型加载显存拆解 + 稳态/峰值/波动 + **10 流外推** |
| `events.jsonl` | 流式产出的融合事件（逐条追加） |
| `debug.mp4` | 全程标注视频（人框/骨架/手部/商品/容器） |
| `alert_*.jpg` | 报警瞬间截图（高亮报警那个人） |

---

## 6. 监控指标

### 6.1 metrics.csv（每帧一行，关键列）

| 字段 | 含义 |
|---|---|
| `frame_id` / `timestamp_ms` / `wall_s` | 帧号 / 时间戳 / 相对启动墙钟秒 |
| `pipeline_ms` / `event_ms` / `open_ms` / `fusion_ms` | 各阶段耗时（管线 / 事件引擎 / PP-TSM open / 融合） |
| `total_ms` / `fps` | 单帧总时延 / 吞吐 |
| `gpu_allocated_mb` / `gpu_reserved_mb` | paddle dygraph 已分配 / 预留显存（只有 PP-TSM open 是动态图） |
| `gpu_max_allocated_mb` / `gpu_max_reserved_mb` | paddle 统计的峰值 |
| `gpu_process_used_mb` | **进程级真实显存**（pynvml 或 `nvidia-smi --query-compute-apps` 兜底） |
| `gpu_total_mb` / `gpu_used_mb` | 整卡总量 / 整卡已用 |
| `cpu_rss_mb` | 进程 RSS 内存 |
| `n_tracks` / `n_open_windows` | 当前行人跟踪数 / open 窗口推断累计 |
| `n_pipeline_events` / `n_package_opening` | 管线事件累计 / 拆包装事件累计 |

### 6.2 summary.json

- `paddle_dygraph_*`：paddle 动态图（PP-TSM open）显存；
- `process_after_load_static_mb` / `process_peak_mb` / `process_dynamic_per_stream_mb`：进程级显存（加载后 / 峰值 / 每流动态增量）；
- `nvidia_smi_whole_gpu`：整卡快照；
- 10 流外推：`single_full`（单流满载）、`shared`（静态共享 + 10×动态）、`multi`（10×峰值）。

> 关键点：`paddle.device.cuda.memory_reserved()` 只看得到**动态图**（PP-TSM open），静态图 `create_predictor` 的显存它看不见，所以 10 流外推用**进程级 nvidia-smi** 口径，更接近真实部署。

### 6.3 实测参考（RTX 5070 Ti 16GB，单流 300s）

- 约 17.8 fps；时延 p50 52.9ms / p95 107.3ms；
- paddle dygraph reserved 约 1008 MB；进程级真实显存约 1.7 GB；CPU RSS 约 3.7 GB。

---

## 7. 报警逻辑

| 等级 | 终端输出 | 额外动作 |
|---|---|---|
| 高风险（管线 high + `package_opening`） | `[ALERT] ...` | **自动截图** `alert_<事件类型>_<person>_<时间戳>.jpg`（红框高亮该人） |
| 中风险（medium） | `[EVENT] ...` | 仅记入 `events.jsonl` |
| 低风险 | 无打印 | 仅记入 `events.jsonl` |

事件字段含 `person_track_id`（MOT 跟踪编号），可与 `debug.mp4` / `alert_*.jpg` 里的编号对应到画面里的人。**注意**：track_id 是跟踪编号，不是身份（人脸/姓名）；人走出画面再回来编号可能变。

---

## 8. 可视化（debug.mp4 框颜色图例）

| 元素 | 颜色 | 标签 |
|---|---|---|
| 人框 | 橙红 | `person-37`（track_id） |
| 报警人 | 红框加粗 | `ALERT person-37` |
| 姿态骨架 | 绿线 + 橙点 | 关键点连线 |
| 手部框 | 蓝 | `left_hand` / `right_hand` |
| 代理商品区（手持商品手部 ROI） | 黄 | `proxy_left` / `proxy_right` |
| 商品 | 绿 | `item` |
| 容器 | 品红 | `bag` / `basket` / `stroller` / `helmet` |

---

## 9. 运行命令

### 9.1 摄像头（生产实况）

```bash
python -m shoplift.cli.realtime_infer \
  --pipeline-config shoplift/configs/pipeline.cc015_v7b_classw102_test.yml \
  --camera-id realtime-cam-001 \
  --source /dev/video0 \
  --open-config shoplift/configs/paddlevideo/pptsm_open_v2_fight16.yaml \
  --open-weights outputs/wqh/paddlevideo/pptsm_open_v2_fight16/ppTSM_epoch_00026.pdparams \
  --paddlevideo-root third_party/PaddleVideo \
  --fusion-config shoplift/configs/fusion.example.yml \
  --output-dir outputs/realtime/single_stream \
  --per-model --duration 300 --summary-every 2
```

### 9.2 本地视频回放（复现 / 测试）

把 `--source` 换成视频路径即可，其余不变：

```bash
python -m shoplift.cli.realtime_infer \
  --pipeline-config shoplift/configs/pipeline.cc015_v7b_classw102_test.yml \
  --camera-id replay-cam-001 \
  --source datasets/test/test_videos/yulong_store/cc015bb3e050a8bc67ebbb2e6fd9951d.mp4 \
  --open-config shoplift/configs/paddlevideo/pptsm_open_v2_fight16.yaml \
  --open-weights outputs/wqh/paddlevideo/pptsm_open_v2_fight16/ppTSM_epoch_00026.pdparams \
  --paddlevideo-root third_party/PaddleVideo \
  --fusion-config shoplift/configs/fusion.example.yml \
  --output-dir outputs/realtime/single_stream \
  --per-model --summary-every 2
```

### 9.3 监控面板（自动指向最新一次 run）

```bash
python -m shoplift.cli.monitor_dashboard --latest outputs/realtime --port 8080
```

浏览器打开 `http://<服务器IP>:8080/`。也可显式指定某次 run：

```bash
python -m shoplift.cli.monitor_dashboard \
  --metrics outputs/realtime/single_stream_YYYYMMDD_HHMMSS/metrics.csv \
  --events  outputs/realtime/single_stream_YYYYMMDD_HHMMSS/events.jsonl \
  --port 8080
```

---

## 10. 注意事项

- 服务器**无显示环境**（headless），`cv2.imshow` 不可用；可视化靠 `debug.mp4` + `alert_*.jpg` + Web 面板。
- `debug.mp4` 编码是纯 CPU，放在时延/显存采样**之后**，不影响测量的显存和时延；但吞吐 fps 可能因编码略降。
- 多流部署时，每条流各起一个 `realtime_infer`（不同 `--camera-id` / `--output-dir`），面板对每个 run 各起一个（或扩展 `--metrics` 支持多文件）。
- 复现/调优建议先跑**本地视频回放**，确认事件与可视化符合预期后再接摄像头。
