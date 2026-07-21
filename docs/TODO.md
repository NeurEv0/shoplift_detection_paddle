# Shoplift 开发路线 TODO

本文档定义 `shoplift/` 开发区的正式建设路线、开发约束、任务拆解和验收标准。范围以当前已收敛的七类能力为准：人员检测/跟踪门控、人体关键点/手部 ROI 与身体区域近似、人员属性预测、容器检测、持商品手部代理区域与容器/身体区域轨迹关联、可疑动作识别、风险评分与规则校验。

## 1. 项目目标

### 1.1 总目标

基于 PaddleDetection 构建商超可疑行为分析引擎，将视频中的人、手部 ROI、左右手持商品属性、代理商品区域、容器和动作关系转化为可解释、可复核、可配置的结构化风险事件。

开发范围为算法与规则侧的七个核心能力：

1. 人员检测/跟踪门控
2. 人体关键点/手部 ROI 与身体区域近似
3. 人员属性预测
4. 容器检测
5. 持商品手部代理区域-容器/身体区域轨迹关联
6. 可疑动作识别
7. 风险评分与规则校验

### 1.2 阶段性目标

| 阶段 | 目标 | 主要产物 |
|---|---|---|
| P0 离线原型 | 跑通单路离线视频分析链路，证明核心视觉证据可被提取 | 人员轨迹、手部 ROI、姿态派生身体区域、人员属性、代理商品区域、容器框、关系事件 JSON、调试可视化 |
| P1 事件引擎 | 形成可复用的轨迹关联、动作状态机和风险评分模块 | `ShopliftingEventEngine`、规则配置、事件 schema、单元测试 |
| P2 试点准备 | 提升评测、配置、性能和接口稳定性，便于接入全栈系统 | 评测脚本、摄像头区域配置、批处理入口、性能报告、接口文档 |

### 1.3 非目标

- 不建设告警后台、复核页面、账号权限、运营报表。
- 不负责 POS/EAS/RFID/VMS 等外部系统集成，只预留字段或接口。
- 不做人脸识别、身份识别、顾客画像或自动执法判断。
- 不在初期追求 SKU 级商品识别或精确商品框检测，优先使用人员属性预测判断左右手是否持有商品，并以手部 ROI 近似商品位置。
- 本版本身体/衣物/口袋区域来自姿态关键点和人员框的几何近似；后续可引入实例分割模型提升区域精度。
- `shoplift/hand_state_classification` 下的模型不纳入本版本方案路线，相关能力以本文档定义的商超人员属性预测模型为准。

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
- 跟踪 ID 分类型命名：`person_track_id`、`hand_track_id`、`proxy_item_track_id`、`container_track_id`。后续如接入精确商品检测，可再补充 `item_track_id`。
- 事件必须包含 `event_id`、`camera_id`、`timestamp_ms`、`person_track_id`、`event_type`、`risk_score`、`risk_level`、`reason_tags`、`evidence`。
- 风险规则必须支持配置化阈值，禁止把关键阈值硬编码在业务逻辑中。

### 2.5 规则与模型约束

- 高风险事件必须来自连续帧证据，不允许单帧触发高风险藏匿判断。
- `holding_product_lost_after_entry` 或 `item_disappeared` 只能作为原因标签之一，不能单独认定藏匿。
- 正常容器如购物篮、购物车、收银袋应支持豁免或降权。
- 规则输出必须可解释，原因标签需能映射到可视化证据。
- 模型置信度低、遮挡严重、多人重叠时应降级为 `low` 或 `medium`，而不是强行输出 `high`。

### 2.6 验证规范

- 每个规则模块至少包含正例、负例、边界条件单元测试。
- 每个事件类型至少准备一组离线视频或伪造轨迹样本用于回归。
- 输出 JSON 需通过 schema 校验。
- 调试可视化需能展示人员框、姿态骨架、手部 ROI、代理商品区域、容器框、轨迹线和原因标签，并明确区分代理区域与真实商品框。
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
- [x] 5. 实现 PaddleDetection 适配器，将 PP-Human/MOT/检测/关键点输出转换为项目内部结构。
- [x] 6. 实现人员检测/跟踪门控：
  - [x] 无人画面跳过重模型。
  - [x] 有人画面输出 `person_track_id`、人体框和轨迹。
  - [x] 记录门控跳过率和触发率。
- [x] 7. 实现人体关键点/手部 ROI 初版：
  - [x] 基于 wrist/arm 关键点派生左右手区域。
  - [x] 实现前臂方向导向裁剪（Forearm-Guided Cropping），利用 elbow->wrist 向量外推手部 ROI 中心并按前臂长度设定裁剪边长。
  - [x] 支持低置信度关键点过滤。
  - [x] 支持 `(0, 0)` 占位关键点过滤和单一 pose score 兼容，避免 PaddleDetection 部分输出格式导致手部 ROI 误判。
  - [x] 输出手部区域与人员轨迹绑定关系。
- [x] 8. 实现商品与容器检测适配占位：
  - [x] 支持粗粒度类别 `item/product`、`bag/backpack/handbag`、`basket/cart`、`stroller`、`helmet`。
  - [x] 支持后续扩展 `clothing_region/pocket_region`。
  - [x] 输出商品和容器检测结果。
  - [x] 当前路线将 `item/product` 精确检测降为后续增强项，P0/P1 闭环优先使用持商品手部代理区域。
- [x] 9. 实现离线分析 CLI：
  - [x] 输入本地视频或帧目录。
  - [x] 输出逐帧结构化结果。
  - [x] 输出调试可视化视频或图片序列。

### 3.2 P1：关系建模与事件引擎

- [x] 1. 实现手-商品接触关系 `hand_item_contact`：
  - [x] 距离阈值。
  - [x] bbox/ROI 重叠。
  - [x] 连续帧稳定性。
  - [x] 手与商品运动方向一致性。
- [x] 2. 实现商品跟随人员关系 `item_follow_person`：
  - [x] 商品轨迹归属到人员轨迹。
  - [x] 处理短时漏检。
  - [x] 处理多人靠近时的冲突归属。
- [x] 3. 实现商品进入容器关系 `item_enter_container`：
  - [x] 包袋/背包/手袋进入判断。
  - [x] 购物篮/购物车正常容器判断。
  - [x] 婴儿车/头盔等特殊容器判断。
  - [x] 衣物/口袋区域进入判断的占位实现。
- [x] 4. 实现商品进入后消失关系 `item_disappeared_after_entry`：
  - [x] 消失帧数阈值。
  - [x] 遮挡与漏检降置信策略。
  - [x] 正常容器豁免逻辑。
- [x] 5. 实现可疑动作状态机：
  - [x] `observing`
  - [x] `item_picked`
  - [x] `near_body_or_container`
  - [x] `suspected_concealment`
  - [x] `confirmed_risk_event`
  - [x] `resolved_or_downgraded`
- [x] 6. 实现首批事件类型：
  - [x] `clothing_concealment`
  - [x] `bag_concealment`
  - [x] `special_container_concealment`
  - [x] `bulk_pickup_to_bag`
  - [x] `near_body_suspicious`
- [x] 7. 实现风险评分：
  - [x] 动作类型权重。
  - [x] 容器类型权重。
  - [x] 连续帧证据权重。
  - [x] 模型置信度权重。
  - [x] 区域风险权重。
  - [x] 正常购物解释降权。
- [x] 8. 实现规则校验：
  - [x] 高风险必须具备至少两个以上原因标签。
  - [x] 购物篮/购物车正常放入默认不触发高风险。
  - [x] 严重遮挡输出需降级并标记 `low_visibility`。
  - [x] 单帧接触不得触发高风险。

### 3.3 P1*：缺口

- [ ] 0.5. 最高优先级：接入商超人员属性预测并替代精确商品检测闭环：
  - [ ] 训练或接入 `docs/supermarket_person_attribute_model.md` 定义的真实人员属性模型，输出左右手 `holding_product`、`holding_object`、`empty`、`uncertain`。
  - [x] 新增属性数据结构、后处理模块和 PP-Human 后端 Paddle Inference 接入点。
  - [x] 输出左右手可见性、人体朝向和遮挡等级，并写入逐帧 JSON。
  - [x] 将 `holding_product + hand_roi` 转换为 `proxy_item_region`，明确标记 `is_precise_item_bbox=false`。
  - [x] 以 `person_track_id + hand_side` 聚合 `proxy_item_track_id`，形成代理商品轨迹。
  - [x] 明确忽略 `shoplift/hand_state_classification` 下的模型，不把该路径作为本版本属性模型来源。
- [x] 0. 最高优先级：补齐姿态识别 Module 与 OR/AND 探索基础：
  - [x] 将 PP-Human KeyPoint 输出提升为独立 `body_poses` 结构化证据，而不是只作为手部 ROI 的中间变量。
  - [x] 输出 COCO 17 点关键点、逐点置信度、骨架边、姿态置信度和 `person_track_id` 绑定关系。
  - [x] 在推理可视化中展示人体框和姿态骨架关键点，并用其支持后续人工判断姿态理解与手部 ROI 的互补性。
  - [x] 在 `datasets/test_videos` 批量推理管线中保留姿态 JSONL 和可视化结果。
  - [x] 完成 pose-only 基线探索，并回溯确认 `hand_item_contact` 必须依赖 `HandRegion` 证据。
  - [x] 基于 `datasets/test_props/origins` 可视化复核，确认前臂导向手部 ROI 定位准确，并保留 crop/overlay 测试输出。
  - [x] 将测试推理与 DCSASS 专用配置切换为姿态和手部 ROI 并发输出：`pose_hand.enabled=true`、`keypoint.derive_hand_regions=true`。
- [ ] 1. 补齐容器检测与姿态派生身体区域闭环：
  - [ ] 准备商超场景粗粒度容器类别标注规范，至少覆盖 `bag/backpack/handbag`、`basket/cart`、`stroller`、`helmet`、可疑袋等。
  - [ ] 训练或接入可本地导出的 PaddleDetection 容器检测模型。
  - [ ] 使用姿态关键点和人员框派生 `clothing_region/pocket_region/sleeve_region/torso_region` 等身体区域近似，不在本阶段依赖实例分割。
  - [ ] 在 `pipeline.local.yml` 中开启容器检测、姿态关键点、手部 ROI、人员属性预测并完成端到端验证。
  - [ ] 输出容器、姿态派生身体区域、手部 ROI、代理商品区域的置信度、类别映射和失败样本。
  - [ ] 将精确 `item/product` 检测保留为后续增强项，不作为 P1* 闭环前置条件。
- [ ] 2. 打通真实视频中的视觉证据到事件闭环：
  - [ ] 在真实视频中稳定产生 `hand_holding_product` 关系。
  - [ ] 在真实视频中稳定产生 `proxy_item_enter_container_or_body_region` 关系。
  - [ ] 在真实视频中验证 `holding_product_lost_after_entry` 与遮挡降级逻辑。
  - [ ] 将人员轨迹、姿态骨架、手部 ROI、人员属性、代理商品区域、容器框和关系证据共同写入事件 evidence。
- [ ] 3. 扩展公开数据集评测：
  - [ ] 保留 DCSASS clip-level smoke eval，用于端到端回归和资源路径检查。
  - [ ] 增加 DCSASS 全量或分组评测，按 `original_video` 分组统计结果。
  - [ ] 输出正样本漏报、负样本误报和失败原因清单。
  - [ ] 补充具备属性级、框级或事件级标注的数据集/自建样本，用于验证左右手持商品属性、代理商品区域、容器和藏匿关系定位。
- [ ] 4. 增强复核可视化证据：
  - [ ] 在 debug 视频中展示人员轨迹线、姿态骨架、手部 ROI、代理商品区域、容器框、姿态派生身体区域和关系连线。
  - [ ] 展示事件时间段、风险分数、风险等级和原因标签。
  - [ ] 在 `predictions.csv` 或评测报告中记录必要可视化资源路径。
  - [ ] 约定公开数据集验证输出路径，例如 `outputs/public_eval/<dataset>_<run_name>/`。
- [ ] 5. 引入动作/时序补充能力：
  - [ ] 评估是否接入动作识别、时序分类或轻量轨迹特征模型。
  - [ ] 区分拿起、查看、放回、靠近身体、放入包内、遮挡消失等时序片段。
  - [ ] 保持规则引擎可解释性，动作模型输出只能作为证据之一。
- [ ] 6. 建立误报压制与阈值校准样本集：
  - [ ] 正常购物、放入购物篮/购物车、收银袋等正常容器负样本。
  - [ ] 查看商品后放回、整理衣服、拿手机、打开包找东西等易误报负样本。
  - [ ] 包袋藏匿、衣物藏匿、特殊容器藏匿、多件商品连续放入袋子等正样本。
  - [ ] 基于样本集校准阈值、降权规则和 `low_visibility` 输出策略。

### 3.4 P2：评测、配置与试点准备

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
- 每帧结果包含人员、姿态骨架、手部 ROI、人员属性、代理商品区域、容器检测结果中的可用子集。
- 能生成结构化逐帧 JSON 和调试可视化结果。
- 基础数据结构和 schema 有最小单元测试。

### 4.2 P1 验收标准

- 能从伪造轨迹或离线视频结果中生成 `RiskEvent`。
- 至少支持 `bag_concealment`、`clothing_concealment`、`special_container_concealment` 三类高优先级事件中的两类。
- 高风险事件必须包含不少于两个 `reason_tags`。
- 正常放入购物篮/购物车的样本不得输出高风险事件。
- 单帧持商品、短时遮挡、低置信度属性或检测不得直接触发高风险。
- 规则、状态机和风险评分模块具备单元测试。

### 4.3 P2 验收标准

- 支持通过配置定义摄像头区域和规则阈值。
- 离线评测脚本能输出事件级指标和失败样本列表。
- 真实 PP-Human + 人员属性预测 + 容器检测后端能在公开数据集或自建样本上产生非空视觉证据；身体/衣物/口袋区域在本版本由姿态关键点近似产生。
- 至少完成一次公开数据集 smoke eval，并以简洁路径保留 `metrics.json`、`predictions.csv` 和必要 debug 可视化资源。
- 事件输出字段稳定，可被全栈系统直接解析。
- 调试可视化能展示人员、姿态骨架、手部 ROI、代理商品区域、容器、姿态派生身体区域、轨迹、关系证据和原因标签。
- 形成性能报告，至少包含 FPS、端到端耗时、模块耗时和门控跳过率。

### 4.4 项目级验收标准

项目进入试点联调前，应满足：

- `shoplift/` 内核心模块职责清晰，PaddleDetection 依赖被适配层隔离。
- 事件输出具备可解释性，人工复核人员能根据 `reason_tags` 和可视化证据理解触发原因。
- 人员属性预测、手部 ROI、代理商品区域、容器检测、姿态派生身体区域、关系证据和事件引擎形成真实视频闭环，不依赖伪造检测结果或精确商品框证明核心能力。
- 公开数据集评测和业务回归样本能分别覆盖端到端异常信号、框级证据和事件级证据。
- 误报高风险场景有明确降权或豁免策略。
- 文档、配置样例、schema、离线 CLI 和测试用例齐备。
- 不输出任何人脸识别、身份识别或自动定性盗窃结论。

## 5. 相关上下文

- [技术方案](docs/shoplifting_detection_technical_solution.md)
- [商超人员属性模型设计](docs/supermarket_person_attribute_model.md)
- [PaddleDetection 能力支撑与补齐项分析](docs/paddledetection_capability_support_analysis.md)
- [商超容器检测数据集与模型调研](docs/container_detection_dataset_model_research.md)
- [PaddleDetection_v2.9](src\PaddleDetection-release-2.9)
