# 容器检测标注任务：改动汇总报告

> 任务：① 评估并增补容器分类标签（私有/公有容器）适配商超场景与商品偷盗检测上报任务；② 制定一套面向 **labelImg** 手工标注的标注规范；③ 保证标注结果与后续 RT-DETR 训练适配。
>
> **当前标签体系：9 类（未变更）**。`trolley`（顾客自带拉车）、`cooler_bag`（保温袋）为候选类别，组长要求**先调研、后讨论**，定稿前不加入标签。
>
> 执行人：本分支作者（按组长要求完成分析、规范与配套工具；手工标注由本人后续执行）。

## 1. 分支信息

- 工作分支：**`feat/container-labeling`**（从 `main` 新建，避免直接改动主分支）
- 当前所有改动均在 `feat/container-labeling` 上，尚未提交；确认无误后由作者自行 commit/提 PR。

## 2. 改动清单

### 2.1 新增文件（9 个）

| 文件 | 类型 | 作用 |
|---|---|---|
| `docs/container_labeling/taxonomy_analysis.md` | 分析文档 | 现有 9 类标签 vs 商超场景/上报事件的适配性分析；问题发现（`plastic_bag` 角色冲突、`cart`/自带拉车混淆等）；`trolley`/`cooler_bag` 候选增补与「不推荐新增」理由；代码侧待办清单 |
| `docs/container_labeling/manual_labeling_spec.md` | **标注规范（核心交付）** | 面向 labelImg/Pascal VOC 的完整手工标注规范：9 类逐类定义与标注口径、全局「标/不标」规则、`difficult` 标记规则、框绘制规则、类别优先级判定、labelImg 操作规范、质量控制与数据分布目标、标注→训练→辅助标注闭环 |
| `scripts/labelimg_voc_to_coco.py` | 工具脚本 | 把 labelImg 的 Pascal VOC XML 转换为 PaddleDetection RT-DETR 训练的 COCO JSON（`images/{train,val,test}/` + `annotations/instances_*.json` + `export_summary.json`）；支持 difficult 过滤、空图负样本保留、train/val/test 划分；**标签由运行时读取 label_list，不硬编码** |
| `scripts/launch_labelimg_container_det.ps1` | 启动脚本 | 一键启动 labelImg 容器标注：自动传入图片目录 / `label_list.txt`（类别文件）/ 保存目录，`-Sequence` 参数切换序列，含前置检查与自动创建标注目录；内置 `setValue(float)` 兼容包装（与源码补丁双保险；UTF-8 BOM，兼容 Windows PowerShell 5.1） |
| `scripts/patch_labelimg_pyqt5.py` | 修复脚本 | 修复 labelImg × PyQt5（sip 6）的 float 参数崩溃：把安装目录里 labelImg 的 8 处 float 调用点包上 `int()`（`setValue` ×4、`drawRect`/`drawLine` ×3、`drawText` ×1）；自动定位 site-packages，幂等；重装 labelImg 后重跑即可 |
| `shoplift/tests/test_labelimg_voc_to_coco.py` | 单元测试 | 7 个用例覆盖：label list 读取、XML 解析、difficult 过滤、未知类别报错、缺图跳过、端到端 COCO 输出、CLI 入口 |
| `scripts/generate_empty_voc_xmls.py` | 工具脚本 | 负样本兜底：labelImg 不自动保存未改动的空图，本脚本扫描图片目录、列出/批量生成"空 XML"（含正确 `<size>`、无 `<object>`），配合转换脚本 `--include-empty` 进入 COCO；默认只列不写，`--generate` 才生成 |
| `docs/container_labeling/training_guide.md` | 训练指南 | 标注 → 转换 → 数据落地 → 服务器环境确认 → RT-DETR-R50VD 训练/评估/导出 → 数据迭代闭环；含第一批数据核对结论与关键坑（dataset_dir 相对 CWD 解析、服务器需同结构 checkout） |
| `docs/container_labeling_task_report.md` | 本报告 | 任务改动汇总 |

### 2.2 修改文件（1 个）

| 文件 | 改动 | 说明 |
|---|---|---|
| `scripts/container_det_groundingdino_pipeline.py` | `read_label_list` 改用 `utf-8-sig` | 防御 Windows 记事本/PowerShell 保存标签文件带 UTF-8 BOM（`\ufeffbag`）导致首类解析失败的问题 |

> - 分析/规范文档位于 `docs/container_labeling/`（`datasets/` 为服务器挂载目录，原始媒体与 `label_list.txt` 等数据仍在该目录）。
> - `datasets/container_det/label_list.txt`：**未变更**（保持 9 类；曾临时追加 `trolley`/`cooler_bag` 已回退）。
> - `docs/collaboration.md` 的修改与 `nfs_fix/` 目录为工作区中**既有未提交改动**，非本次任务产生，未做处理。

## 3. 标签体系结论（核心决策摘要）

### 3.1 当前状态

```text
9 类：bag backpack handbag suitcase basket cart plastic_bag stroller helmet
```

| id | 标签 | 风险角色 | 状态 |
|---:|---|---|---|
| 0–3 | bag / backpack / handbag / suitcase | private（私有/可疑） | 不变 |
| 4–5 | basket / cart | normal（正常购物） | 不变 |
| 6 | plastic_bag | **上下文判定**（现状代码为 normal，建议调整） | 语义重定义待决策 |
| 7–8 | stroller / helmet | special（特殊） | 不变 |
| — | trolley（顾客自带拉车） | private（候选） | **待调研 + 组长讨论，暂不加** |
| — | cooler_bag（保温袋） | private（候选） | **待调研 + 组长讨论，暂不加** |

### 3.2 关键分析结论（详见 `container_labeling/taxonomy_analysis.md`）

1. **高优先问题（待决策）：`plastic_bag` 角色冲突**。代码中它被映射为「正常购物容器」，但顾客自带塑料袋装商品是最高发藏匿场景之一，且外观无法与散称袋区分 → 建议标注照标，风险角色改为「默认可疑，由散称区/结账区上下文降权」。
2. **候选 `trolley`**：顾客自带拉杆/折叠小车会被误判为 `cart`（正常）而漏报；待调研出现频率与外观可分性后讨论。
3. **候选 `cooler_bag`**：生鲜/冷冻区藏匿高发、外观独特；待调研后讨论；**定稿前保温袋按 `bag` 标注**。
4. **不推荐新增**：`waist_bag`（过小）、`paper_bag`/`checkout_bag`（外观与 bag 难分，用上下文规则）、`clothing_region`/`pocket_region`（姿态派生，不进检测）、`item/product`（本阶段不标商品，商品位置用手部 ROI 代理）。

## 4. 标注规范要点（详见 `container_labeling/manual_labeling_spec.md`）

- **标注目标**：可作为藏匿目标或正常购物装载目标的容器，9 类；标注者**不做「是否偷盗」判断**，风险角色由下游规则层按区域/上下文判定。
- **明确不标**：商品本身、人/头/手、衣物/口袋/袖口（姿态派生）、货架/收银台/标牌/影子/反射、员工工服口袋、收银台待售新袋、完全不可见/只露背带的容器。
- **候选类别处理**：顾客自带拉车（候选 `trolley`）**不标注**（避免错标成 `cart`）；保温袋（候选 `cooler_bag`）暂按 `bag` 标注；两者都记入争议样本清单，为调研提供频率数据。
- **标**：完全可见；遮挡 ≥50% 但类型可判（按完整外接框）；被携带/使用中的容器；多实例逐个标。
- **不标**：可见 <40%；短边 < 图像短边 × 2%；严重模糊不可判；只出边缘 <20%；无法归类的对象（不设 unknown 类）。
- **`difficult` 标记**：遮挡 40–70%、极小目标（2%–4%）、极端角度/模糊但类型可判 → 标框 + 勾 Difficult（转换脚本默认过滤，保证训练集质量）。
- **类别优先级判定顺序**：明显背着→`backpack`；挽臂小包→`handbag`；其余软袋→`bag`；带拉杆箱体→`suitcase`（开放式布/塑料车体→候选，不标）；店用大金属车→`cart`（个人小车→候选，不标）；透明塑料→`plastic_bag`；保温材质→暂按 `bag`；兜底选 `bag`（宁粗勿错）。
- **质量与分布**：抽查类别准确率 ≥95%、框 IoU ≥0.7；每批 10% 交叉复核；负样本（无容器帧）10–20%。

## 5. 与后续训练适配

### 5.1 标注工具与格式

- 工具：labelImg，保存格式选 **PascalVOC**（XML）。
- `classes.txt` = `label_list.txt` 内容与顺序（9 类，id 从 0 起，与 COCO category id 一致）。

### 5.2 转换命令

```powershell
python scripts/labelimg_voc_to_coco.py `
  --xml-dir datasets/container_det/manual_work/xml `
  --image-dir datasets/container_det/manual_work/jpg `
  --label-list datasets/container_det/label_list.txt `
  --output datasets/container_det `
  --val-ratio 0.2 --include-empty
```

输出与 GroundingDINO `export-coco` 相同的 COCO 布局（`annotations/instances_{train,val,test}.json` + `images/{split}/`），可直接用于现有 RT-DETR 基线训练；`--keep-difficult` 可保留困难框（`ignore=1`）。

### 5.3 训练配置

- 标签保持 9 类，`container_det_detection.yml` 的 `num_classes: 9` **无需改动**。

### 5.4 代码侧待办（本次未改，等组长确认后实施）

| 待办 | 位置 |
|---|---|
| `plastic_bag` 角色：normal → 默认可疑 + 区域降权（方案 A） | `shoplift/vision/object_container.py`、`shoplift/tracking/association.py`（需同步改 `test_object_container.py` 断言） |
| （若 `trolley`/`cooler_bag` 调研通过）纳入容器类别集合 | `shoplift/vision/object_container.py` |
| （若加入）GroundingDINO 预标注提示词增补 | `scripts/container_det_groundingdino_pipeline.py` 的 `DEFAULT_PROMPT_ALIASES` |
| （若加入）导出模型后更新 `class_id_to_category` | 运行时 `pipeline.*.yml` |

## 6. 验证情况

- `scripts/labelimg_voc_to_coco.py` 用合成 VOC 数据**端到端跑通**（真实 CLI）：difficult 过滤、train/val 划分、空图保留、xyxy→xywh 转换、`export_summary.json` 统计均正确；标签由运行时读取，9 类/11 类均可。
- 单元测试 **7/7 通过**（直接调用测试函数验证；本机沙箱限制 pytest 临时目录清理，`pytest` 命令行在此环境无法完整收尾，属环境问题而非测试失败）。
- `scripts/launch_labelimg_container_det.ps1`：UTF-8 BOM，Windows PowerShell 5.1 解析通过，前置检查友好报错路径实测。
- **labelImg × PyQt5 兼容（Python 3.10）已彻底修复**：实测 PyQt5 5.15.8/5.15.9 + sip 12.11/12.12/12.19 对 `setValue(float)` 全部严格（降级路线无效），故对安装目录源码打补丁——`scripts/patch_labelimg_pyqt5.py` 覆盖 8 处 float 调用点并已应用，`py_compile` 通过；offscreen 端到端实测：`Canvas.paintEvent` 浮点坐标正常、MainWindow 启动加载图片无崩溃。重装 labelImg 后重跑补丁脚本即可。
- 新增/修改均为增量内容，未改动任何现有业务代码逻辑，不影响现有测试套件。

## 7. 待办与决策点（供组长拍板）

1. **`trolley` 调研**：出现频率（建议结合标注过程争议清单统计）、与 `cart` 的外观可分性、能否用规则/区域配置替代——完成后讨论是否加入。
2. **`cooler_bag` 调研**：出现频率、标注成本——完成后讨论是否加入；定稿前保温袋按 `bag` 标注。
3. **`plastic_bag` 角色**：是否按方案 A（默认可疑 + 区域降权）调整代码。
4. 手工标注数据是否已有部分 COCO 标注？如有，后续新增类必须**追加到标签列表末尾**（9、10…），避免 id 错位。
