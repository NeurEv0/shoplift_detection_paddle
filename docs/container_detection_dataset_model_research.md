# 商超容器检测数据集与模型调研

本文面向当前 P1* 缺口：训练或接入一个 2D 容器检测模型，用于在商超监控画面中检测私有容器和正确结账/正常购物容器，并支撑 `proxy_item_enter_container_or_body_region` 关系判断。调研优先覆盖 English datasets，不局限于 PaddleDetection。

## 1. 结论

当前没有发现一个公开英文数据集可以直接覆盖本项目所需的完整闭环：商超 CCTV 视角、人员手部动作、商品代理区域、私有容器、正常购物容器、藏匿事件起止时间和框级标注。可行路线应拆成三层：

1. 用大规模通用检测数据集预训练或抽取相关类别，先解决 `bag/backpack/handbag/suitcase/helmet/stroller/shopping cart/basket` 的通用视觉能力。
2. 用开放词汇检测模型在真实门店视频帧上伪标注容器框，再进行人工复核，快速得到商超视角的少量高质量训练集。
3. 用 PaddleDetection 内的 PP-YOLOE+ / RT-DETR / PP-PicoDet 或框架外 YOLO / DETR 系列做监督微调，最终导出到本项目适配层。

推荐优先级：

| 优先级 | 用途 | 推荐资源 |
|---|---|---|
| P0 | 立即建立容器检测基线 | COCO 预训练 + PaddleDetection PP-YOLOE+ 或 RT-DETR；用 COCO 类别覆盖 `backpack/handbag/suitcase` |
| P0 | 补齐购物车、购物篮、婴儿车、头盔等类别 | Open Images V7、LVIS、Objects365 按类抽取 |
| P0 | 低标注成本启动真实场景数据 | GroundingDINO / YOLO-World / OWL-ViT 对真实帧伪标注，人工复核 |
| P1 | 提升商超域泛化 | 真实门店帧 + 公开数据混合训练，加入正常购物篮/购物车负样本和个人包袋正样本 |
| P1/P2 | 商品/货架辅助，不作为容器主数据 | SKU-110K、RPC、MVTec D2S、GroZi-120 |
| 评测辅助 | 行为异常端到端 smoke eval | DCSASS、UCF-Crime、ShanghaiTech 等弱标注异常视频集 |

## 2. 目标类别建议

先控制在 7 到 10 个粗粒度类别，避免一开始追求细分材质或 SKU 级别：

| 项目内部类 | 外部类别映射建议 | 风险含义 |
|---|---|---|
| `bag` | bag, handbag, purse, tote bag, plastic bag, shopping bag | 私有或不确定容器 |
| `backpack` | backpack | 私有容器 |
| `suitcase` | suitcase, luggage | 私有容器 |
| `basket` | shopping basket, basket | 正常购物容器，默认降权或豁免 |
| `cart` | shopping cart, trolley, cart | 正常购物容器，默认降权或豁免 |
| `stroller` | stroller, baby carriage | 特殊容器 |
| `helmet` | helmet, motorcycle helmet | 特殊容器 |
| `checkout_bag` | checkout bag, paper bag, plastic shopping bag, cashier bag | 结账后正常容器，需要区域/POS/时序辅助 |
| `suspicious_container` | open bag, booster bag, unknown container | 不确定容器，人工复核优先 |

`checkout_bag` 很难仅靠单帧外观判断，建议由检测类 + 收银区 ROI + POS/时序上下文共同生成，不建议单独依赖视觉类别。

## 3. 可用数据集

### 3.1 Open Images V7

Open Images 是最值得优先抽取的英文通用检测数据源之一，原因是类别词表大，通常能覆盖包、背包、手袋、行李、篮子、推车、头盔等长尾容器类。它的优势是框级标注规模大、类别丰富，适合作为容器检测的外部预训练/混合训练数据。

使用方式：

- 按类别抽取 `Backpack`、`Handbag`、`Suitcase`、`Bag`、`Plastic bag`、`Basket`、`Cart`、`Helmet` 等相近类。
- 对类别做内部合并，例如 `Handbag/Purse/Tote bag/Plastic bag -> bag`。
- 训练时需要控制类别不平衡，避免常见包类压过少见的购物篮/婴儿车/头盔。

局限：

- 不是商超 CCTV 视角，遮挡、俯拍、远距离小目标分布与门店监控不同。
- `cart` 可能包含非购物车语义，需要人工抽检或通过开放词汇模型二次过滤。

### 3.2 LVIS

LVIS 的价值在于长尾类别细，很多 COCO 没有的细分类可以在 LVIS 中找到，例如购物车、购物篮、头盔、包袋相关类别。适合作为“补类别”的数据源，尤其是当 COCO 只有 `backpack/handbag/suitcase` 这类粗容器而缺少 `shopping cart/basket/stroller/helmet` 时。

使用方式：

- 抽取与本项目类别相关的 LVIS 类别，再映射为项目内部 7 到 10 类。
- 可用于训练一个比 COCO 更懂长尾容器的教师模型，给真实门店帧打伪标签。

局限：

- 标注分布来自互联网图像，商超监控域偏移明显。
- 长尾类别样本量可能不足，不能完全替代真实门店样本。

### 3.3 Objects365

Objects365 也是适合做通用预训练的大规模检测数据集。它的类别数量比 COCO 多，通常对日常物体、包、交通/生活类容器覆盖更好。若下载和训练成本可接受，可以作为 Open Images / LVIS 的补充。

使用方式：

- 抽取本项目相关类，混合真实门店数据进行微调。
- 若训练资源有限，不必全量训练，可使用已在 Objects365 预训练过的检测器权重作为初始化。

局限：

- 数据体量较大，下载、清洗、转换 COCO 格式的工程成本高。
- 类别命名和项目类别之间需要清晰映射。

### 3.4 COCO

COCO 不能完整覆盖本任务，但非常适合作为第一版 baseline，因为主流检测模型几乎都有 COCO 预训练权重。COCO 中对 `backpack`、`handbag`、`suitcase` 等私有容器有直接类别，但缺少商超购物篮/购物车/结账袋等关键类别。

使用方式：

- 第一版直接复用 COCO 预训练检测器，先验证 `bag/backpack/handbag/suitcase` 在真实视频中能否产生非空证据。
- 后续只作为 backbone/head 初始化，不作为最终类别体系。

局限：

- 正常购物容器覆盖不足，容易把“包袋藏匿”做出来，却无法稳定降权“放入购物篮/购物车”。

### 3.5 SKU-110K

SKU-110K 是商超/零售货架密集商品检测数据集，更适合商品/货架密集检测，不适合作为容器检测主数据源。本项目当前路线已经降低精确商品框优先级，因此 SKU-110K 应作为后续增强项，而不是 P1* 容器闭环前置条件。

可用价值：

- 训练或验证货架密集商品检测、扫货/批量拿取辅助信号。
- 与人员属性模型结合，辅助判断 `holding_product` 的上下文。

不建议：

- 不要把 SKU-110K 当作私有容器/购物容器检测数据。

### 3.6 RPC、MVTec D2S、GroZi-120

这些零售/商品数据集适合商品识别、货架/结账台商品检测或分割，不直接覆盖本项目容器检测。

可用价值：

- RPC：商品级 retail checkout 识别，可辅助未来自助结账漏扫研究。
- MVTec D2S：实例分割质量较高，可用于商品/货架分割方法参考。
- GroZi-120：货架商品识别参考价值高，但年代较早，直接迁移价值有限。

### 3.7 DCSASS、UCF-Crime、ShanghaiTech 等异常视频集

这些数据集可用于端到端异常行为 smoke eval，但不是容器检测训练集。它们通常是视频级或片段级异常标签，缺少容器框、手部属性、商品代理区域和事件级关系标注。

使用方式：

- 保留 DCSASS / UCF-Crime 中的 shoplifting/stealing 类别做端到端回归。
- 不把它们作为容器检测模型的训练来源。

## 4. 模型架构与预训练参数

### 4.1 PaddleDetection 内可落地模型

| 模型 | 推荐用途 | 优点 | 风险 |
|---|---|---|---|
| PP-YOLOE+ | 第一版商超容器检测主力 | PaddleDetection 原生支持，COCO 权重和导出链路成熟，速度/精度均衡 | 对购物篮/购物车等非 COCO 类需再训练 |
| RT-DETR / RT-DETRv2 | 精度优先或中等实时场景 | DETR 系列定位稳定，PaddleDetection 已有配置 | 训练和推理成本可能高于轻量 YOLO |
| PP-PicoDet | CPU/边缘端轻量部署 | 小模型、速度快 | 小目标和遮挡容器召回可能不足 |
| Mask RCNN / Mask RT-DETR | 需要包口/篮口区域时 | 可提供 mask，用于更精细进入关系 | 标注成本高，P1* 不建议作为主路线 |

建议第一版使用 PP-YOLOE+ 或 RT-DETR：

- 若目标是尽快接入现有 PaddleDetection 适配层，优先 PP-YOLOE+。
- 若真实视频中容器遮挡多、定位质量要求高，可以并行试 RT-DETR。
- 若部署端算力弱，再蒸馏/裁剪到 PP-PicoDet 或小型 YOLO。

### 4.2 框架外模型

| 模型 | 推荐用途 | 为什么适合当前缺口 |
|---|---|---|
| GroundingDINO | 伪标注、开放词汇召回、长尾类别探索 | 可以用文本提示直接找 `shopping basket`、`stroller`、`helmet`、`open handbag` 等训练集中缺的类 |
| YOLO-World | 实时开放词汇检测和伪标注 | 比 GroundingDINO 更适合批量跑视频帧，适合快速生成候选框 |
| OWL-ViT / OWLv2 | 开放词汇 baseline | 适合小规模验证文本 prompt 是否能召回目标容器 |
| YOLOv8/YOLO11/RT-DETR 系列 | 非 Paddle 监督训练 baseline | 生态成熟、训练快，可作为和 PaddleDetection 的横向对照 |
| Detic / GLIP | 大词表检测参考 | 对长尾类别有价值，但工程接入成本更高 |

推荐做法：

1. 用 GroundingDINO 或 YOLO-World 在真实视频抽帧上跑 prompt：
   - `backpack`
   - `handbag`
   - `shopping bag`
   - `plastic bag`
   - `shopping basket`
   - `shopping cart`
   - `baby stroller`
   - `helmet`
   - `open bag`
2. 过滤低置信度框，并按人员附近、手部 ROI 附近、货架/收银区 ROI 做二次筛选。
3. 人工复核 1k 到 5k 张关键帧，形成 COCO 格式训练集。
4. 用 PP-YOLOE+ / RT-DETR 在该训练集上微调，输出 PaddleDetection 权重。

## 5. 推荐训练路线

### 5.1 最小可行版本

目标：一周内让真实视频中产生非空、可视化、可调参的容器框。

1. 先用 COCO 预训练 PP-YOLOE+ 跑 `backpack/handbag/suitcase` 类。
2. 用 GroundingDINO/YOLO-World 伪标注真实视频帧中的 `shopping basket/cart/stroller/helmet/bag`。
3. 人工复核少量关键帧，输出 COCO 格式：
   - 每类至少 200 到 500 个高质量框。
   - 正常购物篮/购物车必须覆盖不同视角、遮挡和空/满状态。
   - 个人包袋必须覆盖手提、肩背、打开、半遮挡状态。
4. 微调 PP-YOLOE+ small 或 medium。
5. 接入 `shoplift/vision/object_container.py`，输出内部粗粒度类别和置信度。

### 5.2 推荐数据配比

| 数据来源 | 占比建议 | 作用 |
|---|---:|---|
| 真实门店帧人工复核 | 50% 到 70% | 解决域偏移，是最终效果上限 |
| Open Images / LVIS / Objects365 抽类 | 20% 到 40% | 补长尾外观和类别多样性 |
| COCO 相关类 | 0% 到 20% | 维持通用包类能力 |
| SKU/RPC/D2S 商品数据 | 0% 到 10% | 仅在需要商品上下文时加入 |

### 5.3 标注规范

标注时必须将“风险含义”和“外观类别”分开：

- `basket/cart` 是正常容器，但进入关系应保留，只是在规则层默认降权。
- `bag/backpack/handbag/suitcase` 默认是私有容器，但在收银区或结账后可能成为正常容器。
- `checkout_bag` 不建议仅按外观判断，应由收银区 ROI 和时序/POS 上下文生成。
- `stroller/helmet` 属特殊容器，框可以只标可容纳商品的主体区域，不需要标完整人/车结构。
- 同一画面中人的衣物/口袋区域不属于容器检测模型 P1* 主任务，仍由姿态关键点和人员框近似生成。

## 6. 与当前代码路线的适配建议

1. 配置层新增 `container_detector`：
   - `backend: paddledet`
   - `config_path`
   - `weights_path`
   - `label_map_path`
   - `score_threshold`
   - `class_aliases`
2. 内部类别保持粗粒度：
   - `bag`
   - `backpack`
   - `suitcase`
   - `basket`
   - `cart`
   - `stroller`
   - `helmet`
   - `suspicious_container`
3. 在逐帧 JSON 中输出：
   - `container_bbox`
   - `container_class`
   - `container_role`: `private` / `normal_shopping` / `special` / `unknown`
   - `is_checkout_context`: 由 ROI/POS/时序填充，初期默认为 false
4. 规则层继续使用现有豁免逻辑：
   - `basket/cart` 默认不触发高风险。
   - `bag/backpack/handbag/suitcase/stroller/helmet` 进入后若 `holding_product` 消失或变为 `uncertain`，提高风险。

## 7. 风险与补救

| 风险 | 影响 | 补救 |
|---|---|---|
| 公开数据视角与 CCTV 差异大 | 真实视频误检/漏检 | 真实门店帧必须占最终训练集多数 |
| 购物篮/购物车和普通篮/推车语义混淆 | 正常容器降权失败 | 类别映射后人工抽检，必要时拆成 `shopping_basket` 与 `generic_basket` |
| 结账袋和私有购物袋外观相似 | 收银后误报 | 不靠单帧类别判断，加入收银区 ROI、时间线、POS 上下文 |
| 包口区域无法靠 bbox 精确判断 | `enter_container` 关系噪声大 | P1* 先用 bbox + 手部代理区域，P2 再考虑包口/篮口 mask 或关键点 |
| 开放词汇伪标注误检多 | 训练集污染 | 只把伪标注作为候选，关键类必须人工复核 |

## 8. 建议的下一步

1. 建立 `docs/container_labeling_spec.md`，固定内部类别、别名映射和正负样本规则。
2. 抽取 30 到 60 分钟真实门店视频关键帧，按人员附近/手部附近优先采样。
3. 用 GroundingDINO 或 YOLO-World 生成候选容器框。
4. 人工复核第一批 1k 到 2k 张帧，导出 COCO 格式。
5. 用 PaddleDetection 的 PP-YOLOE+ small/medium 做第一版微调。
6. 跑 `pipeline.local.yml` 端到端验证，检查：
   - 容器框非空率
   - 私有容器召回
   - 购物篮/购物车正常容器误报
   - `proxy_item_enter_container_or_body_region` 是否能稳定产生

## 9. 参考来源

- Open Images V7 download and class metadata: https://storage.googleapis.com/openimages/web/download_v7.html
- LVIS dataset: https://www.lvisdataset.org/dataset
- Objects365 dataset: https://www.objects365.org/overview.html
- COCO dataset: https://cocodataset.org/
- SKU-110K repository: https://github.com/eg4000/SKU110K_CVPR19
- RPC dataset: https://rpc-dataset.github.io/
- MVTec D2S dataset: https://www.mvtec.com/company/research/datasets/mvtec-d2s
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- YOLO-World: https://github.com/AILab-CVC/YOLO-World
- Ultralytics YOLO documentation: https://docs.ultralytics.com/
- PaddleDetection repository and model zoo: https://github.com/PaddlePaddle/PaddleDetection
