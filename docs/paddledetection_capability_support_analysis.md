# PaddleDetection 对商超偷盗行为检测方案的能力支撑与补齐项分析

## 1. 分析范围

本文基于现有方案文档 `docs/shoplifting_detection_technical_solution.md`，以及本地仓库 `src/PaddleDetection-release-2.9/` 的代码、配置和文档进行分析，目标是判断：若要把当前商超偷盗行为检测方案推进到可商业落地级项目，PaddleDetection 能作为哪些能力层的支撑，哪些能力仍需项目侧补齐。

结论先行：

- PaddleDetection 可以作为本项目的视觉算法底座、训练调优底座、视频分析原型底座和多端部署底座。
- 它已经覆盖人员检测、多人跟踪、人体关键点、人体属性、若干行为识别范式、小目标/商品检测训练、模型导出、压缩和推理部署。
- 它不能直接提供“商超偷盗”业务闭环。手-商品-容器关系建模、商品消失/藏匿推理、风险评分、告警短视频、人工复核、POS/EAS/RFID 联动、数据闭环和生产级视频服务，需要在 PaddleDetection 之上单独建设。

## 2. PaddleDetection 可提供的核心能力层

| 方案能力层 | PaddleDetection 支撑能力 | 可复用程度 | 对商超偷盗检测的价值 | 主要限制 |
|---|---|---:|---|---|
| 视频输入与基础 Pipeline | `deploy/pipeline` 支持图片、视频、视频目录、本地摄像头、单/多路 RTSP；支持 RTSP 结果推流；配置文件可按模块启停。 | 高 | 可快速搭建门店摄像头视频分析原型，验证多模型串联流程。 | 不是生产级流媒体平台，缺少断流重连、队列背压、资源调度、健康检查和告警服务治理。 |
| 人员检测 | PP-Human 使用 PP-YOLOE 行人检测模型；`configs/pphuman` 提供 CrowdHuman/业务数据训练配置。 | 高 | 可作为“有人才进入后续推理”的前置门控，也可输出顾客 bbox。 | 商超遮挡、俯视角、货架窄通道仍需门店数据微调。 |
| 多目标跟踪 | PP-Tracking/MOT 支持 ByteTrack、OC-SORT、BoT-SORT、DeepSORT、FairMOT、JDE、CenterTrack；Pipeline 已按 track id 收集结果。 | 高 | 可为每个顾客生成 `track_id`，支撑后续按人聚合动作、商品和容器关系。 | 长时间遮挡、跨摄像头同人关联、多人拥挤场景需要额外调优；商品级跟踪要另做。 |
| 跨镜头关联 | PP-Human/PP-Tracking 提供 ReID 与 MTMCT 流程。 | 中 | 可作为跨摄像头轨迹关联的起点。 | 现有 ReID 面向通用行人开源/业务数据，商业门店需要摄像头拓扑、时间同步、质量评估和隐私策略。 |
| 人体关键点 | 支持 HRNet、HigherHRNet、Lite-HRNet、PP-TinyPose；PP-Human 已集成“跟踪 -> 关键点 -> ST-GCN 行为”的流程。 | 中高 | 可用于判断手腕、手臂、躯干方向，以及取放动作的时序趋势。 | COCO 17 点只有手腕，没有手指；遮挡、低分辨率、俯拍会影响手部细节。 |
| 手部关键点 | 仓库存在 `configs/keypoint/tiny_pose/tinypose_256x256_hand.yml`，支持 21 点手部关键点训练配置。 | 中 | 可作为“手-商品接触”和“放入口袋/包”的细粒度能力起点。 | Pipeline 未直接集成手部关键点模型；需要手部检测/裁剪、训练数据、推理串联和鲁棒性验证。 |
| 人体属性与随身物 | PP-Human 属性识别包含正面持物、手提包、单肩包、背包、长外套、裤装等属性。 | 中 | 可作为风险先验：是否携带包、是否正面持物、衣物类型等。 | 属性是人级分类，不给出包口、口袋、衣物边缘等精确区域。 |
| 商品/容器检测 | 支持 PP-YOLOE+、PP-PicoDet、RT-DETR、Faster RCNN、Mask RCNN 等检测/分割模型；`configs/ppyoloe/application` 包含 SKU110K 商超商品密集检测模型。 | 高 | 可训练商品、包、购物篮、购物车、婴儿车、头盔、货架区域等检测器。 | 通用模型不能直接识别门店自有商品和藏匿容器细节；需要大量门店视角标注数据。 |
| 小目标检测 | `configs/smalldet` 提供 PP-YOLOE-SOD、SAHI 切图/拼图、数据分布统计工具。 | 高 | 商品通常小、密集、遮挡严重，小目标方案很关键。 | 切图会增加延迟；实时视频中需要平衡精度、吞吐和重复框融合。 |
| 实例分割 | 支持 Mask RCNN、Cascade Mask RCNN、SOLOv2、PP-YOLOE-Seg 等。 | 中 | 可用于更精确地得到包袋、衣物、购物篮、人体区域边界。 | PP-Human Pipeline 默认没有把分割作为偷盗关系模块串起来。 |
| 行为识别范式 | PP-Human 支持基于骨骼点、图像分类、目标检测、行人轨迹、视频分类的行为识别；现有动作包括摔倒、打电话、吸烟、打架、闯入。 | 中 | 可复用“按 track id 裁剪、积累时序、输出动作状态”的工程模式。 | 没有现成的 shoplifting/concealment 模型；必须训练自定义偷盗动作模型或规则。 |
| 区域规则 | PP-Human/MOT 支持出入口计数、自定义多边形区域闯入。 | 中 | 可定义货架区、自助结账区、高风险区域、员工区等 ROI。 | 仅有基础区域进入/离开逻辑，不支持复杂购物流程和结账逻辑。 |
| 数据训练与二次开发 | 支持 COCO/VOC 自定义检测数据、关键点 COCO/MPII 数据、MOT 数据、半监督检测；提供训练、评估、推理、导出脚本。 | 高 | 能支撑自定义商品、容器、姿态、跟踪模型的持续迭代。 | 不包含标注平台、数据版本管理、主动学习闭环和事件级样本管理。 |
| 半监督与低标注成本训练 | `configs/semi_det` 支持 DenseTeacher、ARSL、RTDETR-SSOD；文档中包含 SKU110K 半监督实验。 | 中高 | 商超大量无标注视频可用于降低商品/容器检测标注成本。 | 行为事件仍需要高质量人工标注，半监督主要解决检测类任务。 |
| 模型压缩与性能优化 | `configs/slim` 支持剪裁、量化、离线量化、蒸馏、联合策略；`deploy/auto_compression` 支持自动化压缩。 | 高 | 可为边缘盒子、Jetson、CPU 或低功耗部署降延迟、降成本。 | 压缩后必须按门店场景重新评估漏报/误报。 |
| 多端部署 | 支持 Paddle Inference Python/C++、Paddle Serving、Paddle Lite、ONNX、TensorRT、FastDeploy；FastDeploy 覆盖 X86、NVIDIA GPU、ARM、昇腾、昆仑、瑞芯微、晶晨、算能等。 | 高 | 可覆盖云端、边缘服务器、嵌入式盒子、移动端等商业部署形态。 | 业务 Pipeline、告警系统和多模型编排仍需工程化封装。 |
| Benchmark 与可视化 | 提供 Benchmark 脚本、Pipeline 计时器、可视化输出视频。 | 中 | 便于验证帧率、延迟、模型模块耗时和可解释画面。 | 不等于生产监控，需要补 GPU/CPU 指标、队列延迟、告警链路 SLA。 |

## 3. 与现有偷盗检测方案的逐层映射

### 3.1 人员门控层

现有方案提出“无人时跳过后续重模型”。PaddleDetection 可以直接支撑这一层：

- 使用 PP-Human 的 DET/MOT 模块做行人检测。
- 使用轻量模型如 PP-YOLOE-s、PP-YOLOE+ Tiny 或 PP-PicoDet 行人模型做低成本门控。
- 视频无人时只保留低频检测；有人时再进入商品、容器、关键点和行为识别链路。

这一层是 PaddleDetection 最成熟、最应优先复用的能力。

### 3.2 顾客跟踪层

偷盗行为不是单帧判断，而是同一顾客在多帧中的动作链。PaddleDetection 的 MOT/PP-Human 可以提供：

- `track_id`、人体框、轨迹中心点。
- OC-SORT/BoT-SORT/ByteTrack 等可切换跟踪器。
- `DataCollector` 按 track id 收集帧、框、属性、关键点、ReID 特征和动作结果。

项目侧应在此基础上扩展“商品轨迹”和“容器轨迹”，并建立 `person_track_id -> hand_track -> item_track -> container_track` 的关联表。

### 3.3 人体姿态与手部动作层

PaddleDetection 已有全身关键点和手部关键点训练配置，但商业偷盗场景需要更细：

- 全身关键点可用于判断手腕位置、手臂运动方向、身体朝向。
- `tinypose_256x256_hand.yml` 可作为 21 点手部关键点训练起点。
- PP-Human 的骨骼点行为流程可复用为“按人累计关键点序列 -> 时序分类”的工程模式。

需要补齐：

- 手部检测/裁剪模块，尤其是从人体 wrist 关键点向手部 ROI 扩展。
- 手指/手掌与商品 bbox/mask 的接触判断。
- 口袋、袖口、衣物边缘、包口等精细区域定位。
- 低清摄像头和遮挡场景下的稳定性策略。

### 3.4 商品与容器检测层

PaddleDetection 对这一层的支撑很强，但必须训练自定义模型：

- SKU110K 可作为密集商品检测参考，但并不等同于本门店商品与手持商品检测。
- PP-YOLOE+、RT-DETR、PP-PicoDet 可用于商品、包、购物篮、购物车、婴儿车、头盔等检测。
- PP-YOLOE-SOD 和 SAHI 适合货架小商品、远距离商品、密集陈列。
- Mask RCNN/SOLOv2 可用于包袋、衣物、购物篮区域的更精确边界。

建议初期不要追求 SKU 级识别，而是采用粗粒度可落地类别：

- `person`
- `hand`
- `item/product`
- `bag/backpack/handbag`
- `basket/cart`
- `stroller`
- `helmet`
- `shelf_zone`
- `checkout_container`
- `clothing_region/pocket_region`，可先由人体框/关键点派生，后续再训练分割或关键点。

### 3.5 关系建模与行为判断层

这是 PaddleDetection 最不能直接替代项目侧的部分。PaddleDetection 能提供检测、跟踪、关键点和动作识别框架，但“偷盗事件”需要自定义关系推理：

- 手是否接触商品。
- 商品是否从货架区域离开。
- 商品是否与同一顾客 track 持续关联。
- 商品是否进入包口、衣物区域、购物车/购物篮等容器。
- 商品进入非正常容器后是否消失。
- 动作是否持续足够长、是否重复、是否发生在高风险区域。
- 是否存在正常购物解释，例如放入购物篮、放回货架、结账区扫描。

建议项目侧新增独立的 `ShopliftingEventEngine`，输入来自 PaddleDetection 的结构化视觉结果，输出事件和风险分数。

## 4. PaddleDetection 可直接改造的工程切入点

| 切入点 | 位置 | 建议改造方式 |
|---|---|---|
| 视频分析总入口 | `deploy/pipeline/pipeline.py` | 复用配置化 Pipeline，新增 `SHOPLIFTING` 模块开关。 |
| 模块结果容器 | `deploy/pipeline/datacollector.py` | 在 `Result` 中新增 `item`、`container`、`hand`、`relation`、`shoplifting_event` 等结果类型。 |
| 按 track id 聚合 | `DataCollector.append()` | 从人级结果扩展到商品/容器级结果，并保存跨帧关系。 |
| 行人裁剪动作识别 | `pphuman/action_infer.py` | 参考 `DetActionRecognizer`、`ClsActionRecognizer` 新增“藏匿动作分类/检测”模型。 |
| 骨骼点时序缓存 | `pphuman/action_utils.py` | 参考 `KeyPointBuff` 实现手-商品-容器轨迹缓存。 |
| 区域规则 | Pipeline 参数 `region_polygon` | 扩展为多区域配置：货架、出口、自助结账、盲区、高风险货架。 |
| 模型部署 | `deploy/python`、`deploy/cpp`、`deploy/fastdeploy` | 原型阶段用 Python，商业部署阶段按硬件选 C++/FastDeploy/TensorRT/Lite。 |

## 5. 商业落地还必须补充的能力

### 5.1 商超垂直数据资产

PaddleDetection 只能提供训练框架，不能替代真实业务数据。需要建设：

- 门店摄像头采集规范：角度、高度、分辨率、帧率、货架覆盖、隐私遮挡。
- 事件级视频片段数据集：正样本、误报样本、正常购物样本、疑似但未确认样本。
- 多层标注：人框、person_id、手部关键点、商品框/掩码、容器框/掩码、货架/结账区域、事件起止时间、动作类型、复核结果。
- 难负样本库：正常放入购物篮、查看商品后贴身、整理衣服、拿手机、打开包找东西、儿童/婴儿车场景。
- 门店/摄像头/区域分层评测集，避免只在少数视频上调参。

### 5.2 手-商品-容器关系引擎

这是商业可用性的核心。建议新增关系引擎，至少包含：

- `hand_item_contact`：手与商品距离、重叠、运动一致性。
- `item_follow_person`：商品轨迹是否跟随同一人员。
- `item_enter_container`：商品中心点/掩码是否进入包、衣物、婴儿车、头盔等区域。
- `item_disappear_after_entry`：商品进入容器后是否长时间不可见。
- `normal_container_check`：购物篮、购物车、收银台袋子等正常容器豁免或降权。
- `temporal_consistency`：连续帧证据累计，避免单帧误报。
- `explainable_reason_tags`：输出 `item_disappeared`、`concealment_to_bag`、`near_pocket` 等可复核原因。

### 5.3 风险评分与事件状态机

需要把视觉结果转成商业告警：

- 按 `person_track_id` 管理事件状态：观察中、疑似拿取、疑似藏匿、证据增强、告警、结束。
- 按动作类型、容器类型、商品数量、区域风险、持续时间、模型置信度加权。
- 支持不同门店/摄像头/货架的阈值配置。
- 支持静默时段、非营业时段、员工区域白名单。
- 输出结构化事件：`event_id`、`camera_id`、`timestamp`、`person_track_id`、`event_type`、`risk_level`、`reason_tags`、`clip_uri`。

### 5.4 短视频告警与人工复核系统

PaddleDetection 能保存可视化视频，但商业系统需要：

- 环形视频缓存，支持告警前后 N 秒自动截取。
- 告警队列、门店端推送、复核页面。
- 复核状态：待复核、确认、误报、忽略、升级。
- 可解释画面：人、手、商品、容器、轨迹、风险原因叠加。
- 审计日志、权限控制、证据留存周期。
- 反馈回流训练集和阈值配置。

### 5.5 生产级视频服务

Pipeline 原型不等于商业视频平台。还需补充：

- RTSP/RTMP 断线重连、超时、黑屏、卡顿检测。
- 多路流调度、GPU/CPU 资源隔离、动态降级。
- 队列背压和丢帧策略，保障实时性。
- 多模型异步执行：人员门控高频，商品/关键点/行为低频或触发式执行。
- 摄像头配置中心：区域多边形、货架类型、模型版本、阈值。
- 指标监控：FPS、端到端延迟、GPU 显存、队列长度、告警量、误报率。

### 5.6 外部系统联动

部分偷盗方式不能仅靠视频闭环确认，需要联动：

- POS/自助收银：扫描事件、商品清单、结账时间线。
- EAS/RFID：防盗门报警、标签状态。
- 库存系统：高损商品、盘点差异。
- 门店排班/员工系统：员工白名单、巡店通知。
- 摄像头/NVR/VMS：取流、录像回放、证据归档。

### 5.7 MLOps 与持续优化

商业落地后，误报和漏报会持续出现，需要：

- 数据版本管理、标注版本管理、模型版本管理。
- 按门店/摄像头/区域的离线评测报表。
- 灰度发布、A/B 测试、模型回滚。
- 主动学习：高风险未确认、误报、高不确定性片段优先送标注。
- 漂移检测：摄像头角度变化、货架改造、季节服装变化、促销陈列变化。

### 5.8 合规、隐私和运营边界

方案应保持“AI 辅助防损”，不能把视觉告警当成盗窃定性：

- 不做人脸识别或身份识别，除非业务和法规明确允许。
- 告警必须人工复核。
- 明确保留周期、访问权限、日志审计。
- 对误报、顾客争议、员工操作有标准流程。
- 对儿童、婴儿车、残障人士等敏感场景避免激进自动判断。

## 6. 推荐技术路线

### 阶段 0：离线可行性验证

目标是验证“能否在本门店摄像头视角下稳定看见关键动作”。

- 使用 PP-Human MOT 得到人员轨迹。
- 训练粗粒度商品/容器检测模型，类别先控制在 6 到 10 个。
- 使用全身关键点和手部 ROI 估计，验证手-商品接触。
- 离线实现第一版规则：商品从货架离开、跟随手移动、靠近包/衣物后消失。
- 输出离线评测：召回率、误报率、每路视频 FPS、主要失败原因。

### 阶段 1：门店试点

目标是形成“可复核告警”，不是直接自动处置。

- 接入 2 到 5 路真实 RTSP。
- 加入环形缓存和短视频告警。
- 建立复核后台和误报反馈。
- 针对每个摄像头配置货架区、结账区、盲区。
- 重点上线高确定性场景：放入口袋、放入包、放入婴儿车/头盔、多件连续进入袋子。

### 阶段 2：商业化工程

目标是规模化部署和持续优化。

- 服务化拆分视频接入、推理、事件引擎、告警、反馈、训练数据回流。
- 使用 TensorRT/FastDeploy/Paddle Lite 做硬件适配。
- 建立模型版本、灰度发布和回滚机制。
- 与 POS/EAS/RFID/VMS 做复合事件判断。
- 形成按门店、摄像头、货架、时段的运营报表。

## 7. 建议优先补齐清单

| 优先级 | 能力 | 原因 |
|---|---|---|
| P0 | 商超事件级数据集与标注规范 | 没有高质量本域数据，任何模型和规则都无法商业可靠。 |
| P0 | 手-商品-容器关系引擎 | 这是从“检测到人/物”走向“检测到可疑行为”的关键。 |
| P0 | 短视频告警与人工复核闭环 | 商业价值来自可复核事件，而不是离线检测框。 |
| P0 | 摄像头区域配置与风险状态机 | 不同区域的动作含义不同，必须有业务上下文。 |
| P1 | 商品/容器/手部定制模型 | 支撑核心视觉证据，决定召回上限。 |
| P1 | 生产级多路视频服务 | 决定是否能在门店稳定运行。 |
| P1 | 误报样本回流与评测体系 | 决定能否从试点走向规模化。 |
| P2 | POS/EAS/RFID 联动 | 提升复杂场景可信度，尤其是自助结账漏扫、标签替换等。 |
| P2 | 模型压缩与边缘部署优化 | 决定单店硬件成本和可维护性。 |
| P2 | 合规、权限、审计和数据留存 | 决定能否被门店和法务长期接受。 |

## 8. 关键风险判断

- 商品过小、货架密集、手部遮挡会显著降低单纯视觉方案的可靠性。
- “商品消失”不等于“盗窃”，可能是遮挡、放入购物篮、被其他人遮住或模型漏检。
- 自助结账漏扫、标签替换、价格欺诈不能只靠视频动作确认，必须联动 POS/EAS/RFID 或人工复核。
- PP-Human 现有行为识别模型不能直接迁移为偷盗模型，只能复用工程范式。
- 门店摄像头安装质量会直接决定上限，算法不能弥补过远、背光、俯角不合理、货架遮挡严重的问题。

## 9. 本仓库中重点参考的文件与目录

- `src/PaddleDetection-release-2.9/README_cn.md`：模型库、PP-Human、PP-Tracking、PP-TinyPose、部署能力总览。
- `src/PaddleDetection-release-2.9/deploy/pipeline/`：PP-Human/PP-Vehicle 视频分析 Pipeline。
- `src/PaddleDetection-release-2.9/deploy/pipeline/config/infer_cfg_pphuman.yml`：PP-Human 各模块配置。
- `src/PaddleDetection-release-2.9/deploy/pipeline/config/tracker_config.yml`：跟踪器配置。
- `src/PaddleDetection-release-2.9/deploy/pipeline/pipeline.py`：Pipeline 串联逻辑。
- `src/PaddleDetection-release-2.9/deploy/pipeline/datacollector.py`：按 track id 收集结果的数据结构。
- `src/PaddleDetection-release-2.9/deploy/pipeline/pphuman/action_infer.py`：骨骼点、检测式、分类式动作识别实现。
- `src/PaddleDetection-release-2.9/deploy/pipeline/pphuman/action_utils.py`：关键点时序缓存和动作可视化状态。
- `src/PaddleDetection-release-2.9/configs/pphuman/README.md`：PP-Human 行人、属性、行为相关模型。
- `src/PaddleDetection-release-2.9/configs/keypoint/tiny_pose/tinypose_256x256_hand.yml`：21 点手部关键点训练配置。
- `src/PaddleDetection-release-2.9/configs/ppyoloe/application/README.md`：PP-YOLOE+ 下游任务与 SKU110K 商超商品检测。
- `src/PaddleDetection-release-2.9/configs/smalldet/README.md`：小目标检测、SAHI 切图/拼图。
- `src/PaddleDetection-release-2.9/configs/semi_det/README.md`：半监督检测。
- `src/PaddleDetection-release-2.9/configs/slim/README.md`：剪裁、量化、蒸馏和模型压缩。
- `src/PaddleDetection-release-2.9/deploy/README.md`、`deploy/python/README.md`、`deploy/cpp/README.md`、`deploy/fastdeploy/README.md`：部署形态与硬件支持。

