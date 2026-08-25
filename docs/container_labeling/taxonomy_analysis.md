# 容器标签体系与商品偷盗检测上报任务适配性分析

> 目的：评估现有容器分类标签（9 类）是否覆盖商超主要场景、是否适配商品偷盗检测上报任务，并给出标签增补与角色调整建议。本文档供组长决策与标注执行参考。
>
> 结论先看第 1 节；详细论证见后文；代码侧待办见第 6 节。
>
> **当前状态（2026-02 修订）**：标签体系维持 **9 类不变**。`trolley`（顾客自带拉杆/折叠购物车）与 `cooler_bag`（保温袋/冰袋）列为**候选增补，待调研与讨论后定稿**，暂未加入标签。

## 1. 结论摘要

1. **现有 9 类外观标签基本覆盖商超 90%+ 容器场景**，与 5 类上报事件（`bag_concealment`、`clothing_concealment`、`special_container_concealment`、`bulk_pickup_to_bag`、`near_body_suspicious`）的映射总体成立，**不建议推倒重来**。
2. **高优先级问题（未解决，待决策）：`plastic_bag` 的角色语义冲突**。代码中 `plastic_bag` 被映射为「正常购物容器」（等同购物篮），但顾客**自带塑料袋装商品是最高发藏匿场景之一**，外观上无法与散称袋区分。若一律按正常购物处理会**系统性漏报**。→ 建议：标注照标 `plastic_bag`（不拆类），风险角色改为「默认可疑，由区域上下文（散称区/结账区）降权」。
3. **候选增补 1：`trolley`（顾客自带拉杆/折叠购物车，private）**。现有 `cart` 只有「店用购物车」语义，顾客自带小拉车会被误判为 `cart`（正常购物）而漏报。**待调研**：出现频率、与 `cart` 的外观可分性、标注成本 → 组长讨论后决定。
4. **候选增补 2：`cooler_bag`（保温袋/冰袋，private）**：生鲜/冷冻区藏匿高发、外观独特（厚/反光/拉链）。**待调研**：出现频率、标注成本 → 组长讨论后决定。
5. **不建议新增**：`waist_bag`（目标过小）、`paper_bag` 单独类（与 `bag` 外观难分）、`checkout_bag`（与 `bag` 外观重复，用结账上下文规则处理）、`clothing_region`/`pocket_region`（由姿态关键点派生，不进入检测模型）、`item/product`（本阶段聚焦容器，商品位置用手部 ROI 代理）。
6. **落地清单**：本次已交付标签体系分析、手工标注规范 `manual_labeling_spec.md`、VOC→COCO 转换脚本 `scripts/labelimg_voc_to_coco.py`；**标签列表保持 9 类未变**；代码侧待办见第 6 节，待确认后实施。

## 2. 现状标签 → 商超场景覆盖矩阵

现状 9 类及其在偷盗检测中的角色（角色来源：`shoplift/vision/object_container.py` 与 `shoplift/tracking/association.py`）：

| id | 标签 | 风险角色 | 覆盖场景 |
|---:|---|---|---|
| 0 | `bag` | private（私有/可疑） | 顾客自带的购物袋、帆布袋、纸袋、编织袋、不明软包（含保温袋等软质袋） |
| 1 | `backpack` | private | 双肩包/书包/单肩背包，含背在身上 |
| 2 | `handbag` | private | 手提包/钱包/手抓包/小托特包 |
| 3 | `suitcase` | private | 行李箱/拉杆箱（低频但存在） |
| 4 | `basket` | normal（正常购物） | 店用购物篮/手提篮/带轮小篮 |
| 5 | `cart` | normal | 店用购物车/带儿童座椅购物车 |
| 6 | `plastic_bag` | normal（**有争议，见 4.1**） | 塑料袋：散称袋/顾客自带袋/收银装袋 |
| 7 | `stroller` | special（特殊容器） | 婴儿车/童车 |
| 8 | `helmet` | special | 摩托车/自行车/硬壳头盔 |

| 场景 | 是否覆盖 | 说明 |
|---|---|---|
| 商品放入个人包袋（包/背包/手提包/行李箱） | ✅ | bag/backpack/handbag/suitcase 全覆盖 |
| 商品放入顾客自带塑料袋 | ⚠️ 半覆盖 | 有 `plastic_bag` 类，但被当作正常购物处理 → 漏报风险 |
| 商品放入散称区提供的塑料袋 | ✅ | `plastic_bag`，正常购物 |
| 商品放入购物篮/购物车（正常） | ✅ | basket/cart |
| 商品放入顾客自带小拉车/折叠购物车 | ❌ 缺失（候选 `trolley`，待定） | 会被误判为 cart（正常）→ 漏报 |
| 商品放入婴儿车/头盔 | ✅ | stroller/helmet |
| 商品放入保温袋/冰袋 | ⚠️ 部分覆盖（候选 `cooler_bag`，待定） | 目前按 `bag` 标注，损失类别信号但风险角色一致（private） |
| 商品塞入口袋/衣物/袖口 | 设计外 | 由姿态关键点派生 `clothing_region`/`pocket_region`，不依赖检测模型（见 4.5） |
| 可疑袋（booster bag 金属衬袋等） | ⚠️ 外观不可分 | 外观即普通购物袋，标注仍按 `bag`，风险由区域/上下文规则判定 |

## 3. 现状标签 → 偷盗上报事件映射

上报事件定义见 `shoplift/events/`（P1 事件引擎）：

| 上报事件 | 依赖的容器证据 | 现状是否成立 | 备注 |
|---|---|---|---|
| `bag_concealment`（个人包藏匿） | 代理商品进入 private 容器（bag/backpack/handbag/suitcase） | ✅ 基本成立 | 若后续加入 `trolley` 更完整 |
| `clothing_concealment`（衣物藏匿） | 代理商品进入衣物/口袋区域 | ✅ 成立 | 区域由姿态派生，与检测标签无关 |
| `special_container_concealment`（特殊容器藏匿） | 代理商品进入 stroller/helmet | ✅ 成立 | `cooler_bag` 若加入可归入 special 或 private |
| `bulk_pickup_to_bag`（批量拿取入袋） | 多商品轨迹进入同一 private 容器 | ✅ 成立 | 依赖容器检测持续可见 |
| `near_body_suspicious`（贴身可疑） | 不需要容器证据 | ✅ 成立 | — |
| （隐含）正常购物豁免 | 进入 basket/cart 时降权 | ✅ 成立 | `plastic_bag` 是否豁免需重定义（见 4.1） |

## 4. 发现的问题

### 4.1 `plastic_bag` 角色语义冲突（高优先，待决策）

- 现状：`shoplift/vision/object_container.py` 中 `plastic_bag` 被映射到 `NORMAL_CONTAINER_CATEGORY`（= `basket`），即「正常购物容器」；`association.py` 也将其视为 normal。
- 冲突：商超偷盗检测中，**顾客把商品放进自己携带的塑料袋**（非散称区场景）是典型藏匿行为；而「散称区提供的塑料袋」才是正常购物。两者**外观几乎无法区分**（都是透明/半透明塑料袋）。
- 影响：若 `plastic_bag` 一律按 normal 处理，自带塑料袋装商品会被「正常购物豁免」降权 → 系统性漏报。
- 建议（三选一，推荐 A）：
  - **A（推荐）**：保留 `plastic_bag` 单一外观类，标注照标；风险角色改为**默认 private/special**，当区域上下文（散称区、收银/结账区）证明为正常时再由规则降权。改动落在 `object_container.py` 与区域配置，不改标签体系。
  - B：拆成 `plastic_bag`（散称袋，normal）与 `shopping_bag`（自带袋，private）——**不推荐**，标注者无法可靠区分，会造成标注方差和模型混淆。
  - C：保持现状——**不推荐**，漏报风险不可接受。
- 标注口径影响：标注者看到塑料袋一律标 `plastic_bag`，**不需要判断用途**。

### 4.2 `cart` 无法区分店用/顾客自带（高优先，调研中）

- 中国商超常见场景：顾客（尤其老年顾客）自带**折叠购物车/拉杆小车**，外观与店用购物车不同（体积小、可折叠、布料/塑料材质、带拉杆或背带），但现有标签只有 `cart` 且为 normal。
- 影响：自带小拉车被检测为 `cart` → 正常购物豁免 → 顾客把整袋商品放入自带车后离开被漏报。
- 候选方案：**新增 `trolley`（顾客自带拉杆/折叠购物车，private）**。外观可分（小型、折叠结构、非金属篮筐），标注者判定成本低。
- **待调研项**：① 该场景在已有门店视频中的出现频率；② 与 `cart` 的外观可分性（会不会导致标注方差）；③ 是否可先用规则/区域配置覆盖而不加类。

### 4.3 `bag` 定义过宽，标注方差大（中优先）

- 现状定义「Generic personal or unknown soft container」会把购物袋、帆布袋、纸袋、编织袋、保温袋、不明软包全归入 `bag`。
- 影响：`bag` 内部混杂不影响风险角色（都是 private），但会降低类内一致性；若后续需要细分（如按袋型统计）会受限制。
- 建议：规范中给出**类别优先级判定顺序**（见 `manual_labeling_spec.md` 第 4 节），宁粗勿错；不为此拆分标签。

### 4.4 `suitcase` 低频（低优先）

- 商超中行李箱少见，但存在（顾客带行李箱购物/搬运）。保留即可，不删除（删类会打乱 id 与已有数据）。

### 4.5 衣物/口袋区域不在检测范围（设计说明）

- `clothing_region`/`pocket_region` 是姿态关键点+人员框的几何近似（`pose_hand`/`body_pose` 派生），不属于容器检测模型的标注目标。规范中明确「不标」，避免标注者把衣服、口袋、袖口画框。

## 5. 增补建议（调研中，暂未加入标签）

### 5.1 候选：`trolley`（顾客自带拉杆/折叠购物车）

- 理由：见 4.2，解决店用/自带购物车混淆导致的漏报；中国商超高频场景（待调研确认频率）。
- 角色：private（私有容器，藏匿/带走高发）。
- 标注口径（若加入）：顾客自带的拉杆购物车、折叠购物车、带轮购物袋（非店用）；与 `cart`（店用大购物车）区分。
- **状态：待调研 + 组长讨论，暂不加入标签**。

### 5.2 候选：`cooler_bag`（保温袋/冰袋）

- 理由：生鲜/冷冻区藏匿高发（待调研确认频率）；外观独特（厚壁、反光内衬、拉链、方形），与 `bag` 容易区分；独立成类可支撑 `special_container_concealment` 或 `bag_concealment` 上报。
- 角色：private 或 special（实现时二选一，倾向 private）。
- 标注口径（若加入）：保温袋、冰袋、冷藏袋、野餐保温包；普通软袋不算。
- **状态：待调研 + 组长讨论，暂不加入标签**；当前按 `bag` 标注。

### 5.3 语义调整：`plastic_bag` 角色（见 4.1，方案 A）

- 标签不变，标注照标；代码角色由 normal → 默认可疑、区域上下文降权。

### 5.4 不推荐新增及理由

| 候选 | 不推荐理由 |
|---|---|
| `waist_bag` / `fanny_pack` 腰包 | 目标过小（<20px），检测收益低；贴身藏匿多由姿态/贴身规则覆盖 |
| `paper_bag` 单独类 | 与 `bag` 外观重叠，标注与模型都难区分 |
| `checkout_bag` 结账袋 | 外观与 `bag` 相同，正常性应由结账区域上下文判定，不靠外观类 |
| `clothing_region` / `pocket_region` | 由姿态派生，不进入检测模型 |
| `item` / `product` 商品类 | 本阶段聚焦容器；商品位置使用「持商品手部 ROI 代理区域」，不依赖精确商品框 |
| `booster_bag` 等可疑袋 | 外观与普通购物袋不可分，标注无意义；风险由区域/行为规则判定 |

### 5.5 当前标签列表（9 类，无新增）

```text
bag          # 0  private  通用软包/购物袋/纸袋/编织袋（含保温袋等软质袋）
backpack     # 1  private  双肩/单肩背包
handbag      # 2  private  手提包/钱包
suitcase     # 3  private  行李箱
basket       # 4  normal   店用购物篮
cart         # 5  normal   店用购物车
plastic_bag  # 6  上下文   塑料袋（散称/自带/收银，角色由区域判定，待决策）
stroller     # 7  special  婴儿车
helmet       # 8  special  头盔
```

> ⚠️ 若后续确定新增 `trolley`/`cooler_bag`，必须**追加到列表末尾**（9、10），不能插入中间，否则已导出的 category id 映射全部错位。训练配置 `num_classes` 随之调整。

## 6. 落地清单

### 本次已交付（可直接用于手工标注）

| 交付物 | 位置 |
|---|---|
| 标签体系分析（本文件） | `docs/container_labeling/taxonomy_analysis.md` |
| 手工标注规范（labelImg / Pascal VOC，9 类） | `docs/container_labeling/manual_labeling_spec.md` |
| VOC XML → COCO JSON 转换脚本（标签由运行时读取） | `scripts/labelimg_voc_to_coco.py` |
| 转换脚本单元测试 | `shoplift/tests/test_labelimg_voc_to_coco.py` |
| `read_label_list` UTF-8 BOM 兼容修复 | `scripts/container_det_groundingdino_pipeline.py`、`scripts/labelimg_voc_to_coco.py` |

> `datasets/container_det/label_list.txt` 保持 9 类，未变更。

### 待确认后实施（代码侧）

| 改动 | 位置 | 说明 |
|---|---|---|
| `plastic_bag` 角色：normal → 默认可疑/区域降权 | `shoplift/vision/object_container.py`、`shoplift/tracking/association.py` | 方案 A（见 4.1），需同步更新 `test_object_container.py` 断言 |
| （若加入 `trolley`/`cooler_bag`）纳入容器类别集合 | `shoplift/vision/object_container.py`（`ITEM_AND_CONTAINER_CLASSES`、`CONTAINER_CATEGORIES`、`CANONICAL_ITEM_CONTAINER_CLASSES`） | 待调研讨论定稿后实施 |
| （若加入）GroundingDINO 提示词增补 | `scripts/container_det_groundingdino_pipeline.py` 的 `DEFAULT_PROMPT_ALIASES` | 辅助标注/预标注阶段 |
| （若加入）导出模型后 `class_id_to_category` 更新 | 运行时 `pipeline.*.yml` 的 `backend.item_container.class_id_to_category` | 训练+导出后 |

## 7. 待确认决策点（供组长拍板）

1. **`trolley` 是否加入**：完成调研（出现频率、与 `cart` 可分性、规则替代可行性）后讨论决定。
2. **`cooler_bag` 是否加入**：完成调研（出现频率、标注成本）后讨论决定；加入前保温袋按 `bag` 标注。
3. **`plastic_bag` 角色**：是否按方案 A（默认可疑 + 区域降权）调整代码？
4. 手工标注数据是否已有部分 COCO 标注？如有，后续新增类必须追加到末尾，避免 id 错位。
