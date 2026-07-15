# Shoplift 开发路线 TODO

本文档定义 `shoplift/` 开发区的正式建设路线、开发约束、任务拆解和验收标准。范围以当前已收敛的六类能力为准：人员检测/跟踪门控、人体关键点/手部区域分析、商品与容器检测、手-商品-容器轨迹关联、可疑动作识别、风险评分与规则校验。

## 1. 项目目标

### 1.1 总目标

基于 PaddleDetection 构建商超可疑行为分析引擎，将视频中的人、手、商品、容器和动作关系转化为可解释、可复核、可配置的结构化风险事件。

开发范围为算法与规则侧的六个核心能力：

1. 人员检测/跟踪门控
2. 人体关键点/手部区域分析
3. 商品与容器检测
4. 手-商品-容器轨迹关联
5. 可疑动作识别
6. 风险评分与规则校验

### 1.2 阶段性目标

| 阶段 | 目标 | 主要产物 |
|---|---|---|
| P0 离线原型 | 跑通单路离线视频分析链路，证明核心视觉证据可被提取 | 人员轨迹、手部区域、商品/容器框、关系事件 JSON、调试可视化 |
| P1 事件引擎 | 形成可复用的轨迹关联、动作状态机和风险评分模块 | `ShopliftingEventEngine`、规则配置、事件 schema、单元测试 |
| P2 试点准备 | 提升评测、配置、性能和接口稳定性，便于接入全栈系统 | 评测脚本、摄像头区域配置、批处理入口、性能报告、接口文档 |

### 1.3 非目标

- 不建设告警后台、复核页面、账号权限、运营报表。
- 不负责 POS/EAS/RFID/VMS 等外部系统集成，只预留字段或接口。
- 不做人脸识别、身份识别、顾客画像或自动执法判断。
- 不在初期追求 SKU 级商品识别，优先实现粗粒度商品与容器关系判断。

## 2. 开发约束和规范

### 2.1 代码边界

- 自有业务代码放在 `shoplift/`。
- PaddleDetection 上游代码放在 `src/PaddleDetection-release-2.9/`，优先作为依赖和参考使用。
- 如必须修改 PaddleDetection 源码，需在 PR/提交说明中写明原因、影响范围、回归方式，并优先提供可配置开关。
- 全栈系统负责视频服务、告警推送、短视频存储、复核后台和外部系统联动；`shoplift/` 只输出结构化事件与调试证据。

### 2.2 独立开发环境

`shoplift/` 必须具备独立、可复现、可校验的开发环境。开发、测试和离线分析入口应从仓库根目录执行，不应依赖开发者手动进入 `src/PaddleDetection-release-2.9/` 修改路径或临时安装包。

环境约束：

- 使用独立 `conda` 虚拟环境，环境名为 `shoplift-paddle`。
- Python、PaddlePaddle、PaddleDetection、OpenCV、NumPy 等关键依赖需要显式记录版本范围。
- PaddleDetection 作为上游视觉底座，通过适配层、配置和 `PYTHONPATH`/可编辑安装等方式接入，不能把 `shoplift/` 业务代码散落到 PaddleDetection 源码目录。
- CPU/GPU 依赖需要分层说明：基础开发环境先保证 CPU 可运行单元测试；GPU/CUDA/PaddlePaddle GPU 作为推理和性能测试环境单独说明。
- 模型权重、视频样本、推理输出和缓存不得提交到仓库，应通过配置路径引用，并在示例配置中使用占位路径。
- 需要提供环境自检脚本，至少检查 Python 版本、`paddle` 导入、`ppdet` 或 PaddleDetection 路径可用性、`cv2` 导入、GPU 可见性和关键配置文件存在性。

建议环境产物：

```text
.
├── requirements.txt
├── requirements-dev.txt
├── environment.yml
├── scripts/
│   └── check_env.py
└── shoplift/
    └── configs/
        ├── pipeline.example.yml
        ├── rules.example.yml
        └── env.example.yml
```

### 2.3 推荐目录结构

```text
shoplift/
├── __init__.py
├── configs/
│   ├── pipeline.example.yml
│   ├── rules.example.yml
│   ├── env.example.yml
│   └── schema.md
├── adapters/
│   └── paddledet_adapter.py
├── vision/
│   ├── person_gate.py
│   ├── pose_hand.py
│   └── object_container.py
├── tracking/
│   ├── track_types.py
│   └── association.py
├── events/
│   ├── event_engine.py
│   ├── state_machine.py
│   └── event_schema.py
├── rules/
│   ├── risk_score.py
│   └── validators.py
├── cli/
│   └── offline_analyze.py
└── tests/
```

### 2.4 数据与接口规范

- 所有模块间传递结构化数据，不传递裸散字典作为长期接口。
- 时间字段统一使用 `timestamp_ms`；帧序号统一使用 `frame_id`。
- 坐标统一为像素坐标 `[x1, y1, x2, y2]`，必要时额外提供归一化坐标。
- 跟踪 ID 分类型命名：`person_track_id`、`hand_track_id`、`item_track_id`、`container_track_id`。
- 事件必须包含 `event_id`、`camera_id`、`timestamp_ms`、`person_track_id`、`event_type`、`risk_score`、`risk_level`、`reason_tags`、`evidence`。
- 风险规则必须支持配置化阈值，禁止把关键阈值硬编码在业务逻辑中。

### 2.5 规则与模型约束

- 高风险事件必须来自连续帧证据，不允许单帧触发高风险藏匿判断。
- `item_disappeared` 只能作为原因标签之一，不能单独认定藏匿。
- 正常容器如购物篮、购物车、收银袋应支持豁免或降权。
- 规则输出必须可解释，原因标签需能映射到可视化证据。
- 模型置信度低、遮挡严重、多人重叠时应降级为 `low` 或 `medium`，而不是强行输出 `high`。

### 2.6 验证规范

- 每个规则模块至少包含正例、负例、边界条件单元测试。
- 每个事件类型至少准备一组离线视频或伪造轨迹样本用于回归。
- 输出 JSON 需通过 schema 校验。
- 调试可视化需能展示人员框、手部区域、商品/容器框、轨迹线和原因标签。
- 性能指标至少记录 FPS、端到端耗时、各模块耗时和跳过率。

## 3. TODOs

### 3.1 P0：离线原型与基础结构

- [x] 1. 初始化独立开发环境：
  - [x] 明确 Python、PaddlePaddle、PaddleDetection、OpenCV、NumPy 等依赖版本范围。
  - [x] 新增 `requirements.txt`，记录运行时依赖。
  - [x] 新增 `requirements-dev.txt`，记录测试、格式化、类型检查、文档校验等开发依赖。
  - [x] 新增 `environment.yml`，提供 Conda 环境创建入口。
  - [x] 新增 `shoplift/configs/env.example.yml`，配置 PaddleDetection 路径、模型权重路径、视频样本路径、输出目录和设备类型。
  - [x] 新增 `scripts/check_env.py`，检查 Python、Paddle、PaddleDetection、OpenCV、GPU 可见性和关键目录。
  - [x] README 中补充环境创建、依赖安装、环境自检和离线 CLI 运行命令。
  - [x] 确认单元测试可在不下载模型权重的 CPU 基础环境中运行。
- [x] 2. 初始化 `shoplift/` Python 包和推荐目录结构。
- [x] 3. 定义核心数据结构：`FrameMeta`、`DetectionBox`、`Tracklet`、`HandRegion`、`RelationEvidence`、`RiskEvent`。
- [x] 4. 定义事件 JSON schema 与示例文件。
- [ ] 5. 实现 PaddleDetection 适配器，将 PP-Human/MOT/检测/关键点输出转换为项目内部结构。
- [ ] 6. 实现人员检测/跟踪门控：
  - [ ] 无人画面跳过重模型。
  - [ ] 有人画面输出 `person_track_id`、人体框和轨迹。
  - [ ] 记录门控跳过率和触发率。
- [ ] 7. 实现人体关键点/手部 ROI 初版：
  - [ ] 基于 wrist/arm 关键点派生左右手区域。
  - [ ] 支持低置信度关键点过滤。
  - [ ] 输出手部区域与人员轨迹绑定关系。
- [ ] 8. 实现商品与容器检测适配：
  - [ ] 支持粗粒度类别 `item/product`、`bag/backpack/handbag`、`basket/cart`、`stroller`、`helmet`。
  - [ ] 支持后续扩展 `clothing_region/pocket_region`。
  - [ ] 输出商品和容器检测结果。
- [ ] 9. 实现离线分析 CLI：
  - [ ] 输入本地视频或帧目录。
  - [ ] 输出逐帧结构化结果。
  - [ ] 输出调试可视化视频或图片序列。

### 3.2 P1：关系建模与事件引擎

- [ ] 1. 实现手-商品接触关系 `hand_item_contact`：
  - [ ] 距离阈值。
  - [ ] bbox/ROI 重叠。
  - [ ] 连续帧稳定性。
  - [ ] 手与商品运动方向一致性。
- [ ] 2. 实现商品跟随人员关系 `item_follow_person`：
  - [ ] 商品轨迹归属到人员轨迹。
  - [ ] 处理短时漏检。
  - [ ] 处理多人靠近时的冲突归属。
- [ ] 3. 实现商品进入容器关系 `item_enter_container`：
  - [ ] 包袋/背包/手袋进入判断。
  - [ ] 购物篮/购物车正常容器判断。
  - [ ] 婴儿车/头盔等特殊容器判断。
  - [ ] 衣物/口袋区域进入判断的占位实现。
- [ ] 4. 实现商品进入后消失关系 `item_disappeared_after_entry`：
  - [ ] 消失帧数阈值。
  - [ ] 遮挡与漏检降置信策略。
  - [ ] 正常容器豁免逻辑。
- [ ] 5. 实现可疑动作状态机：
  - [ ] `observing`
  - [ ] `item_picked`
  - [ ] `near_body_or_container`
  - [ ] `suspected_concealment`
  - [ ] `confirmed_risk_event`
  - [ ] `resolved_or_downgraded`
- [ ] 6. 实现首批事件类型：
  - [ ] `clothing_concealment`
  - [ ] `bag_concealment`
  - [ ] `special_container_concealment`
  - [ ] `bulk_pickup_to_bag`
  - [ ] `near_body_suspicious`
- [ ] 7. 实现风险评分：
  - [ ] 动作类型权重。
  - [ ] 容器类型权重。
  - [ ] 连续帧证据权重。
  - [ ] 模型置信度权重。
  - [ ] 区域风险权重。
  - [ ] 正常购物解释降权。
- [ ] 8. 实现规则校验：
  - [ ] 高风险必须具备至少两个以上原因标签。
  - [ ] 购物篮/购物车正常放入默认不触发高风险。
  - [ ] 严重遮挡输出需降级并标记 `low_visibility`。
  - [ ] 单帧接触不得触发高风险。

### 3.3 P2：评测、配置与试点准备

- [ ] 1. 增加摄像头区域配置：
  - [ ] 货架区。
  - [ ] 收银/自助结账区。
  - [ ] 高风险货架区。
  - [ ] 盲区/低可信区域。
- [ ] 2. 增加规则配置文件：
  - [ ] 每类事件阈值。
  - [ ] 每个摄像头阈值覆盖。
  - [ ] 正常容器白名单。
  - [ ] 静默或降敏配置占位。
- [ ] 3. 建立离线评测脚本：
  - [ ] 输入标注事件与预测事件。
  - [ ] 输出召回率、误报率、事件级 Precision/Recall。
  - [ ] 输出按事件类型、摄像头、区域的失败统计。
- [ ] 4. 建立回归样本集规范：
  - [ ] 正常购物负样本。
  - [ ] 查看商品后放回负样本。
  - [ ] 整理衣服/拿手机/打开包找东西负样本。
  - [ ] 包袋藏匿正样本。
  - [ ] 衣物藏匿正样本。
  - [ ] 多件商品连续进入袋子正样本。
- [ ] 5. 增加接口文档：
  - [ ] 输入视频/帧接口。
  - [ ] 逐帧结果接口。
  - [ ] 事件输出接口。
  - [ ] 调试可视化输出接口。
- [ ] 6. 增加性能基准：
  - [ ] 单路离线视频 FPS。
  - [ ] 门控跳过率。
  - [ ] 关键点/检测/关系/规则模块耗时。
  - [ ] 不同模型配置下的性能对比。

### 3.4 暂不纳入本开发区的 TODO

- [ ] 短视频生成与对象存储。
- [ ] 告警队列、门店端推送和复核后台。
- [ ] 用户权限、审计日志、证据留存策略。
- [ ] POS/EAS/RFID/VMS 联动实现。
- [ ] 跨摄像头同人关联。
- [ ] SKU 级商品识别和价格欺诈判断。

以上任务可由全栈、平台或后续专项接入，本开发区仅预留事件字段和配置扩展点。

## 4. 验收标准

### 4.1 P0 验收标准

- 能对至少 1 段本地视频或帧目录执行离线分析。
- 人员门控可输出 `person_track_id`，无人片段能够跳过后续重模型。
- 每帧结果包含人员、手部 ROI、商品/容器检测结果中的可用子集。
- 能生成结构化逐帧 JSON 和调试可视化结果。
- 基础数据结构和 schema 有最小单元测试。

### 4.2 P1 验收标准

- 能从伪造轨迹或离线视频结果中生成 `RiskEvent`。
- 至少支持 `bag_concealment`、`clothing_concealment`、`special_container_concealment` 三类高优先级事件中的两类。
- 高风险事件必须包含不少于两个 `reason_tags`。
- 正常放入购物篮/购物车的样本不得输出高风险事件。
- 单帧接触、短时遮挡、低置信度检测不得直接触发高风险。
- 规则、状态机和风险评分模块具备单元测试。

### 4.3 P2 验收标准

- 支持通过配置定义摄像头区域和规则阈值。
- 离线评测脚本能输出事件级指标和失败样本列表。
- 事件输出字段稳定，可被全栈系统直接解析。
- 调试可视化能展示人员、手部、商品、容器、轨迹和原因标签。
- 形成性能报告，至少包含 FPS、端到端耗时、模块耗时和门控跳过率。

### 4.4 项目级验收标准

项目进入试点联调前，应满足：

- `shoplift/` 内核心模块职责清晰，PaddleDetection 依赖被适配层隔离。
- 事件输出具备可解释性，人工复核人员能根据 `reason_tags` 和可视化证据理解触发原因。
- 误报高风险场景有明确降权或豁免策略。
- 文档、配置样例、schema、离线 CLI 和测试用例齐备。
- 不输出任何人脸识别、身份识别或自动定性盗窃结论。

## 5. 相关上下文

- [技术方案](docs/shoplifting_detection_technical_solution.md)
- [PaddleDetection 能力支撑与补齐项分析](docs/paddledetection_capability_support_analysis.md)
- [PaddleDetection_v2.9](src\PaddleDetection-release-2.9)
