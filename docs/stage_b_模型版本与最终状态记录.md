# Stage B 人员属性 + 容器模型:配置变更全史与最终状态记录

> 记录范围:从**第一次跑管线**(配置未做任何改动时)到**最终确认模型**
> (属性 `hi384_v7_best` + 容器 `classw101`)之间所有配置/数据/规则改动、
> 改动目的、中间命令、最终文件位置,以及最终模型在 cc015 / D03 上的表现与已知短板。
> 整理日期:2026-09-05。服务器:10.200.6.16(e-ai2@ZHITAI_2tb,项目根
> `/home/e-ai2/ZHITAI_2tb/shoplift_detection_paddle`,conda 环境
> `/home/e-ai2/ZHITAI_2tb/conda_envs/shoplift-paddle`,GPU 0,CC 12.0,PIR)。

> **2026-09-09 更新(详见文末 §8)**: 最终容器模型由 classw101 升级为 **classw102**
> (v2exp 数据 + 权重 [1,2,1.2,1,1,1,1,1,1]), 修复 classw101 在 cc015 37-50s 的
> 贴篮背包幻影误检(贴篮 backpack 743→1); 属性模型 hi384_v7_best 与规则
> rules.hi384_geometry_v1.yml 未变。注意: 事件层新增的 person-71 bag_concealment
> HIGH(0.891@23040ms) 经核实为**规则自重叠假阳性**(手持"商品"= 被属性模型误判的背包,
> 私有"容器"= 同一背包的检出框, 无真商品入私有容器), 非真事件 —— 见 §8.4.1。
> classw101 章节保留为历史。

---

## 0. 一句话现状

**最终管线 = 人员属性 `models/shoplift/person_attribute/inference_hi384_v7_best`(384×512,6 头)
+ 容器 `models/shoplift/item_container/rtdetr_r50vd_6x_container_det_classw101`(backpack 权重 3.0)
+ 规则 `rules.hi384_geometry_v1.yml`(margin80)。
最终验证输出:cc015_v7b_classw101(64 事件,0 concealment)与 d03_v7b_m80_classw101
(472 真事件 HIGH 保持、316 手机误报修复;遗留 69/76/422 误报,见 §6)。**

---

## 1. 首次跑管线的配置(改动起点)

第一次端到端管线测试配置 = `shoplift/configs/pipeline.container_pa_test.yml`:

| 项 | 首版取值 |
|---|---|
| 输入 | yulong_store/cc015…mp4(92 s@25 fps,1280×720) |
| 人员属性模型 | `models/shoplift/person_attribute/inference`(**192×256**,6 头早期小模型) |
| 容器模型 | `models/shoplift/item_container/rtdetr_r50vd_6x_container_det`(RT-DETR R50vd 6x,无自定义 class 权重) |
| 规则 | `rules.example.yml`(几何规则的雏形) |
| 输出 | `outputs/wqh/container_pa_test/cc015/` |

后续主线改动分三轴:**① 属性模型(架构+数据+权重档) ② 容器模型(class 权重) ③ 几何规则(margin 参数)**。

---

## 2. 属性模型演进(数据每轮增量)

训练配方(hi384 族,自 v1 起基本不变):384×512、batch 32、class_weights [1,1,3,0.7]、
60 ep、lr 5e-4、warmup 5、seed 2026、pphgnetv2-S;6 头标签:
hand_state(empty/holding_object/holding_product/uncertain)、hand_visibility(clear/partial_occluded/not_judgable)、
body_orientation(front/side/back/unknown)、occlusion(none/light/heavy);
loss 加权:左手/右手状态 ×2.0,可见性 ×1.5,朝向 ×1.0,遮挡 ×0.5。

`class_weights [1.0, 1.0, 3.0, 0.7]`(按 empty / holding_object / holding_product / uncertain 顺序)
的由来(源自 Stage B 方案设计,详见 `docs/session_report_closure_and_attribute_retrain.md` §6.3):
- **定位**:head 权重(手状态 2.0 / 可见性 1.5 / 朝向 1.0 / 遮挡 0.5)负责**任务间**平衡;
  CE 类别权重是**任务内**类别不均衡修正——每类样本的损失乘以其权重;
- **holding_product ×3.0**:偷盗判定的关键类,在数据中极稀有(反频率约 1/0.036≈28,
  平方根反频≈5.3),取 3.0 是保守提振召回,防止模型塌缩到 empty 多数类、漏掉真持商品帧;
- **uncertain ×0.7**:约占样本 63%、且多为"看不清/兜底"标签,轻微降权(不屏蔽——
  运行时仍需保留 uncertain 的保守语义,配合 visibility 门控与规则兜底);
- **副作用与盯守**:提召回可能略降 precision → 训练/评估始终用逐头 per-class
  recall/precision 双指标盯守,若 precision 崩则考虑降到 2.0;
- 该权重自 hi384 v1 起沿用至今(v4-v7 均未改动),历版左右手状态 test acc 0.83-0.86,
  未见 precision 崩溃,故维持 3.0。

| 版本 | 训练数据增量(相对上一版) | 数据切分目录 | 导出目录 | 选中权重(协议) |
|---|---|---|---|---|
| v1(hi384) | 基线全量 reviewed | person_attribute(v1) | inference_hi384(v1-v4 共用,后被 v4 覆盖) | best_val_loss |
| v4 | + **D03 person-472 39 帧**(偷盗者窗口,force→train) | person_attribute_v4?→(见 v5 行) | 同上(覆盖) | best_val_loss |
| v5 | + **cc015 person-71 41 帧**(帧 530-570,force→train) | **person_attribute_v4_cc015/splits**(新目录,沿用至今) | inference_hi384_v5 | best_val_loss(ep12,val loss 4.978) |
| v6 | cc015 **f568/f569/f570 三帧重标**:holding_product→uncertain / partial_occluded→not_judgable(修正过度自信 0.99) | 同 v4_cc015(仅同步标签) | inference_hi384_v6 | best_val_loss(ep8) |
| v7 | + **D03 person-316 8 帧手机样本**(f11555-11562,left=holding_object;修复手机被误判 holding_product) | 同 v4_cc015(train 追加 8 行=2160,val/test 不变 259/259) | inference_hi384_v7(bvl) 与 **inference_hi384_v7_best**(best@57) | bvl(ep9) / best(ep57) |
| **v7_best(最终)** | = v7 数据 | person_attribute_v4_cc015 | **inference_hi384_v7_best** | **best.pdparams(ep57,val acc 0.8700 最高)** |

选型结论:min-val-loss 早停点(bvl@ep8-9)在场景上不稳定(见 §6 95/390 误报、472 掉 LOW),
最终采用 **val acc 峰值 ep57(best.pdparams)** —— 唯一同时保住 D03 person-472 真事件并修复
person-316 手机误报的权重。

数据主线(合并后即 v7 训练集构成):
```
基线全量 reviewed 标注
+ D03 person-472 39 帧(force train)
+ cc015 person-71 帧 530-570(41 行,force train;含旧 @720 1 行自然落 train)
+ f568-570 标签修正(uncertain/not_judgable)
+ D03 person-316 8 帧手机样本(holding_object,clean 顺序解码抽取)
= datasets/person_attribute_v4_cc015/splits(train 2160 / val 259 / test 259)
```

### 属性关键命令(项目根)
```bash
# 训练(容器训练同法,cd src/PaddleDetection-release-2.9 + PYTHONPATH=. 仅容器需要)
PY=/home/e-ai2/ZHITAI_2tb/conda_envs/shoplift-paddle/bin/python
nohup $PY -m shoplift.models.person_attribute.train_hi \
  --config shoplift/configs/person_attribute_hi384_v7.yml \
  > outputs/wqh/person_attribute/hi384_v7_train.log 2>&1 &
# 评估(三权重对比用)
$PY -m shoplift.models.person_attribute.evaluate_hi \
  --config shoplift/configs/person_attribute_hi384_v7.yml \
  --checkpoint outputs/wqh/person_attribute/hi384_v7/best.pdparams --split val
# 导出最终模型(concat 格式 → 目录可直接被管线 model_dir 引用)
$PY -m shoplift.models.person_attribute.export \
  --config shoplift/configs/person_attribute_hi384_v7.yml \
  --weights outputs/wqh/person_attribute/hi384_v7/best.pdparams \
  --output-dir models/shoplift/person_attribute/inference_hi384_v7_best --format concat
```

---

## 3. 容器模型演进(class 权重)

RT-DETR R50vd 6x,9 类 [bag0..helmet8],backpack 为 idx1。训练 config 在
`shoplift/configs/paddledetection/container_det/`。

| 版本 | backpack 权重 | 改动目的 | 导出目录 |
|---|---|---|---|
| 基线 | 默认(≈1) | 原始训练(无自定义权重) | rtdetr_r50vd_6x_container_det |
| classw100 | **4.0** | 提升挎包检出(cc015 person-71 黑挎包)→ 引发 person-103 白袋误报 | …_classw100(已删) |
| **classw101(最终)** | **3.0** | 用户拍板降权:保留挎包检出同时消除 103 白袋误报 | **…_classw101** |

代码侧改动(已部署,带备份 .bak_classw100):`src/PaddleDetection-release-2.9/ppdet/…/detr_loss.py`
与 `utils.py` 增加 `class_weights` / per-query 权重注入;eval 用
`EvalReader collate_batch:false + eval_with_loss True`(新机器 --eval 不崩的关键)。

容器关键命令(在 src/PaddleDetection-release-2.9 内):
```bash
PY=/home/e-ai2/ZHITAI_2tb/conda_envs/shoplift-paddle/bin/python
PYTHONPATH=. $PY tools/train.py -c ../../../shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det_classw101.yml --eval
# 导出 best_model → models/shoplift/item_container/rtdetr_r50vd_6x_container_det_classw101
# (final: val AP 0.709,idx53)
```

---

## 4. 几何规则演进

| 版本 | margin | 说明 |
|---|---|---|
| rules.example.yml | 120(早期) | 私有容器框顶扩张 120 px → D03 误报多 |
| **rules.hi384_geometry_v1.yml(最终)** | **80** | 扩张 80 px:滤掉 person-95 式误报,保留 472 真事件;配套事件判定:entry ≥3 连续帧、item 轨缺失 ≤5 帧、进入后消失 ≥10 帧 |

最终管线 `rules_config: ./shoplift/configs/rules.hi384_geometry_v1.yml`。

---

## 5. 最终文件/数据位置(留存清单)

**服务器 10.200.6.16 项目根**
- 最终属性模型导出:`models/shoplift/person_attribute/inference_hi384_v7_best/`
  (inference.pdmodel/pdiparams/infer_cfg.yml)
- 属性训练源(保留,供复现/重导):`outputs/wqh/person_attribute/hi384_v7/`
  (best@57、best_val_loss、last、test_summary.json)+ `hi384_v7_train.log`
- 最终容器模型导出:`models/shoplift/item_container/rtdetr_r50vd_6x_container_det_classw101/`
  (model.json + model.pdiparams 169MB)
- 容器训练源(保留):`outputs/wqh/container_det/rtdetr_r50vd_6x_container_det_classw101/`
- 最终训练数据:`datasets/person_attribute_v4_cc015/splits/`
  (train.csv 2160 / val.csv 259 / test.csv 259 + images/train|val|test)
- 标注 master(含最终标签):`datasets_annotation/person_attribute/person_attribute_yulong_store_crops/`
  (full.csv + images/full;含 f568-570 修正与 person-316 8 行手机样本)
- 最终配置:`shoplift/configs/pipeline.cc015_v7b_classw101_test.yml`、
  `pipeline.d03_v7b_m80_classw101_test.yml`、`person_attribute_hi384_v7.yml`、
  `rules.hi384_geometry_v1.yml`、`…/container_det/rtdetr_r50vd_6x_container_det_classw101.yml`
- 最终验证 run(完整保留含 debug):`outputs/wqh/container_pa_test/cc015_v7b_classw101/`(≈1.9GB)、
  `outputs/wqh/container_pa_test/d03_v7b_m80_classw101/`(≈41GB,debug_frames + debug_visualization.mp4)

**本地工作区**
- `shoplift/configs/`(首版 + 最终端点配置;中间版本已清理)
- 本文档:`docs/stage_b_模型版本与最终状态记录.md`

历史版本(models 导出 v1-v7、训练输出 hi384~hi384_v6、容器基线/classw100、旧 run 输出、
旧数据集 person_attribute~v3、中间窗口 crops 等)已按确认清单删除。

---

## 6. 最终模型表现(cc015 / D03,2026-09-05,最终配置)

### cc015(container_pa_test/cc015_v7b_classw101)
| 指标 | 数值 |
|---|---|
| 总事件 | 64(near_body medium 58 / low 6) |
| **bag_concealment** | **0** ✅ |
| person-103(白袋误报历史主角) | 5× near_body medium,无 concealment ✅(classw101 降权后持续为 0) |
| person-71(黑挎包人物) | 1× near_body medium 0.495,**无事件** ⚠️ 见下 |

### D03(container_pa_test/d03_v7b_m80_classw101)
| person | 结果 | 判定 |
|---|---|---|
| person-472(标注真偷) | concealment **HIGH 0.85** @576480-576920 | ✅ 真事件保住 |
| person-316(持手机) | 无 concealment(v5 误报 HIGH 0.895 已消除) | ✅ 手机负样本生效 |
| person-95 / person-390 | 无 concealment | ✅(v7_bvl@9 的误报在 best@57 消失) |
| person-69 / person-76 | concealment HIGH 0.867/0.805 | ⚠️ 见下 |
| person-422 | concealment HIGH 0.786 @536280-536880 | ⚠️ 见下 |
| 总事件 | 57 | — |

### 效果不好的项与原因
1. **cc015 person-71 无事件(最老问题,与模型无关)**:MOT 在 f552-562/567-569/571-585+
   存在跟踪空洞(人物贴右缘 x2=1280 被裁),item 轨与容器关联断裂 → 事件引擎看不到完整
   "入包"链。修法在 MOT 稳定性/贴边处理,不在容器或属性模型。
2. **D03 person-69/76/422 误报 HIGH**:v5 曾被压到 LOW,v7_best 回归 HIGH。疑似这三段是
   "正常把商品放进自备购物篮/车/袋"被语义化为 concealment(basket/cart 判 normal 的语义
   规则尚未落地,遗留项"69/76 bag-in-cart 语义规则")。未定论:待目检 debug
   (69≈1:14、76≈1:17、422≈8:56)确认后,应在规则层用容器类别门控修正,而非改模型。
3. **属性早停点不稳定**:min-val-loss 的早停(bvl@ep8-9)在不同场景上行为跳跃大
   (v7_bvl 在 D03 丢 472、新增 95/390),val loss 对场景行为预测力弱 → 选型以场景实测为准,
   最终选 val acc 峰值(best@57)。

### 数据抽取备注(踩坑记录)
D03_堂食区_修复版.mpg 实为带损坏参考帧的 HEVC 封装:**cv2 随机跳帧
(CAP_PROP_POS_FRAMES)解码会产出灰白坏帧**,必须**从第 0 帧顺序解码**取帧
(管线全程顺序读所以 debug 帧是干净的)。属性训练图抽取(如 person-316 8 帧)一律顺序解码;
手工追加 splits csv 行时 image_path 需用 `train/` 前缀且图片放 `splits/images/train/`。

---

## 7. 后续工作:拆包装识别(待办)

**目标**:识别"偷盗者把商品**拆掉外包装/撕开包装**后藏匿或丢弃包装"的行为——这是
concealment 之前常见的一步(拆包 → 商品贴身/入私袋 → 包装丢弃/留在货架),当前事件引擎
只能看到"持商品 → 入袋",看不到"拆包"动作本身。

**待确认的语义边界**(开工前先与需求方对齐):
- 动作定义:撕开/打开包装取出商品(包装与商品分离)才算;普通"从货架拿整盒商品"不算;
- 输出形态:新增事件类型(如 `package_opening` / `unpackaging`)还是作为既有
  concealment 的前置强化信号(拆包后紧跟入私袋 → 直接 HIGH)?

**初步思路(未验证,待方案评审)**:
1. **数据**:抽 D03/cc015 及新增视频中的拆包帧(双手撕包装、包装落地/被丢弃),
   属性侧需区分 holding_product(未拆,整件)与拆包中(双手 + 包装变形/分离),可能加标签或
   交给动作链判断,不进 6 头属性模型;
2. **几何/事件链**:复用 item-follow/入容器链,新增"包装分离"关系——检测到与商品同源的
   第二物体(包装盒/塑料袋)短暂出现后被丢弃/静止(弃置物检测),或人物双手都在包装上持续
   数帧(双手合拢动作 + 高 motion);
3. **弃置物(包装)检测**:把"地面/货架残留的小盒/包装袋"建模为短生命周期对象,与人物
   proxy-item 结束点关联;
4. **验证集**:优先在 D03(已知偷盗段)与 cc015 person-71 窗口标注拆包帧,先出 recall 基线。

> 依赖前置:① 69/76/422 的"正常放篮 vs 拆包"语义规则定案;② MOT 空洞修复(否则窗口内
> 动作链仍会断)。本小节为待办占位,方案与结论后续在此追加。

---

## 8. 续:classw102 回合(2026-09-09,治 cc015 贴篮背包幻影)

**结论(2026-09-09 复核更正): 检测层(classw102 模型)达成目标并通过 —— 贴篮幻影
压制、真背包检出质量提升、D03 零回归;事件层新增的 person-71 HIGH 经核实为
规则自重叠 FP(见 §8.4.1), 不计为真能力, 事件层验收结论改为"待规则修复后再评"。
属性模型 hi384_v7_best 与规则 rules.hi384_geometry_v1.yml 未动,**本轮零新增规则
代码**(事件触发均为既有规则)。

### 8.1 背景与路线

- classw101(backpack 权重 3.0)在 cc015 37-50s 段于篮筐上方持续误报 backpack
  (贴篮幻影:人离开后仍报,如 f1400 一帧 8 个贴篮检出 0.23-0.55);
- 根因:训练集 backpack 仅 13 正例、零负例、allow_empty:false,权重硬顶致对
  "篮子上方类背包纹理"过拟合;
- 用户拍板走**重训路线(档2:数据增强 + 重训)**,不继续堆规则;标注由用户本地
  labelImg 完成后回传服务器合并。

### 8.2 数据:v2exp 数据集

- **新标注 master**:`datasets_annotation/container_det/cc015_window_v2exp/`
  (72 帧 f375-1440 stride15,images/full + full.csv + annotations/72 个 VOC XML,
  规则: f375-585 标真背包+可见容器, f585-1440 只标可见容器不标背包);
- **合并**:`datasets/container_det_v2exp/` = container_det 全量复制 + 合并
  72 帧中与旧 train 不重复的 **60 帧**(12 帧 stride90 重叠自动跳过);
- train **223→283 图、1089→1325 标注**,backpack **13→35**(含远离篮区真例 +
  ~58 帧"篮子可见无包"难负例),handbag 占比 26%→21.4%;val/test 保持不变;
- 旧数据零覆盖(全在 v2exp 新目录)。

### 8.3 训练/导出

- 配置:`shoplift/configs/paddledetection/container_det/rtdetr_r50vd_6x_container_det_classw102.yml`
  - class_weights `[1.0, 2.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
    (backpack 3.0→2.0 数据补足后退让;handbag 1.0→1.2 护 D03 占比稀释);
  - TrainDataset 指向 v2exp,`allow_empty: false`(同 classw101 —— 旧 json 含
    19 张 D03 零标注图,allow_empty:true 会把它们放进训练导致空 gt collate 崩溃);
  - 其余(100ep,milestones[66,88],lr 5e-5,PIR)与 classw101 一致;
- 训练日志:`outputs/wqh/container_det/rtdetr_r50vd_6x_container_det_classw102/train.log`
  (15:51→16:18 ~27min,100 轮 eval,best @ep99);
- 导出:`models/shoplift/item_container/rtdetr_r50vd_6x_container_det_classw102/`
  (best_model.pdema,PIR:infer_cfg.yml + model.json + model.pdiparams)。

### 8.4 验收

**A. 类别 AP(同 val 26 图,--classwise,best_model.pdparams)**

| 类别 | classw101 | classw102 | Δ |
|---|---|---|---|
| handbag | 0.785 | 0.796 | +1.1pt(**门控>2pt 未触发,PASS**) |
| plastic_bag | 0.736 | 0.753 | +1.7pt |
| basket | 0.800 | 0.800 | 0 |
| cart | 0.955 | 0.949 | -0.6pt(噪声) |
| bag | 0.271 | 0.181 | val 仅 1-2 GT,噪声级 |
| backpack | nan | nan | val 无 GT → 场景帧把关 |
| mAP | 0.709 | 0.696 | -1.3pt(预期内) |

**B. cc015 场景回归**(cc015_v7b_classw102 vs cc015_v7b_classw101,同规则/属性/MOT,
仅容器模型不同;frame_results 按 attributes.source_category 统计)

| 窗口 | cw101 backpack | cw102 backpack | 贴篮(x>1050,y>400) 101→102 |
|---|---|---|---|
| 0-20s | 1940 | 247 | 582→172 |
| 20-23s 真包 | 470(max0.55) | 81(max0.83) | 227→81(全真位) |
| 23-37s | 1709 | 255 | 550→61 |
| 37-45s 幻A | 1517 | 200 | 422→**0** |
| 45-51s 幻B | 989 | 83 | 321→**1** |
| 51-92s 离场 | 6263 | 932 | 2760→476 |
| **合计** | **12888** | **1798(-86%)** | **4862→791(-84%)** |

- 真包 20-23s 检出不降反升(0.55→0.83)且更聚焦;
- **事件 64→66**:新增 2 条(均非真事件,详见 §8.4.1):
  - `person-71 bag_concealment HIGH 0.891 @23040ms`(f576) = **规则自重叠假阳性**
    (无商品入私有容器);classw101 同段检出仅 0.20-0.27 且抖断,凑不齐 3 帧
    entry 一致性故未触发 —— 非"检测变强抓到真事",而是检测变强把规则漏洞点亮;
  - `person-339 bag_concealment LOW 0.136 @71200ms`(正常购物豁免路径,无害);
  - 其余 64 条逐条不变(时间/分数一致)。

### 8.4.1 核实更正:person-71 HIGH 为"自己碰自己"自重叠 FP(2026-09-09 用户质疑后核实)

**质疑**: 事件语义为"商品进私有容器",但画面里没有任何商品进私有容器,为何触发?

**机制(代码 + 帧证据,全部属实)**:
1. 分类学(`shoplift/vision/object_container.py` + `shoplift/tracking/association.py`):
   `backpack/handbag/suitcase → bag`(PERSONAL_CONTAINER_CATEGORY),
   `PRIVATE_CONTAINER_CATEGORIES = {"bag"}` → **一切 backpack 检出 = 私有容器**;
   容器模型 9 类**无商品类**,引擎里的 "item" 全部来自人员属性模块的
   "手持物"代理区域(`proxy_item_regions`),即 person_attribute 的 holding 信号;
2. 帧证据 f565(classw102):
   - person-71 属性:`left holding_object 1.0 / right holding_product 0.99`
     —— 属性模型把他**握着的背包**误判为"手持商品";
   - `proxy-item-person-71-right`(右手代理) box [1197,525,1234,562];
   - `item_container_det-565-3`(backpack 检出,私有容器) box [1196,537,1231,563];
   - 两者是**同一个物理背包**,重叠 ≈90%;
3. 触发链: 代理"商品"(=背包)→ 进入"私有容器"(=同一背包的检出框,
   center inside + 3 帧一致)→ f567+ 包入篮检出分下滑被解读为
   `item_disappeared_after_entry` + `gap_filled` → HIGH 0.891@23040ms;
4. 为什么 classw101 不触发: 同段仅 0.20-0.27 抖断检出(如 f563 三个 0.2-0.27 框),
   中心入框一致性凑不满 3 帧;classw102 稳定 0.78-0.83 → 暴露既有规则漏洞;
5. 同机制在 D03 person-472/422/69/76 成立的差异: 那里代理 item 是**真商品**。
   区分信号缺失 = 引擎无法区分"手持物是自己的包"vs"手持物是货架商品"。

**决策(2026-09-09 用户拍板)**: 修复方向 a)/b) **均不做** —— person-71 的 HIGH
属**人员属性模型 holding_product 小误检**(把"拿自己背包"判成"手持商品"), 影响低,
不为此改规则; cc015 的偷盗(私有容器进公有容器)检出**交由后续"私有容器进公有容器"
两段式新规则**承载(2026-09-10 **已实现并通过验收**,见 §8.6)。

**C. D03 回归:57 条事件逐条不变**(person-472 真 HIGH concealment 0.85 保持;
person-69 0.87→0.86;person-316 手机误报不复发;person-95 状态不变)→ 零回归。

**残留观察项**:cc015 51-92s 贴篮 backpack ~476(0.47/帧,max 0.86,含 f1395-1440
用户标注的真实背包段),事件层无一人+入篮关联故不触发;后续其他视频回归时顺带观察。

### 8.5 文件清单

- 标注 master:`datasets_annotation/container_det/cc015_window_v2exp/`(72 帧 +
  full.csv + annotations/72 xml)
- 合并脚本(已执行,勿删):`.tmp/merge_v2exp.py`;合并结果
  `datasets/container_det_v2exp/{images,annotations,label_list.txt}`
- 对比脚本:`.tmp/compare_cw101_vs_cw102.py`
- 验收输出:`outputs/wqh/container_pa_test/cc015_v7b_classw102/` 与
  `outputs/wqh/container_pa_test/d03_v7b_m80_classw102/`
  (events.json/frame_results.jsonl/run.log/debug_visualization.mp4)
- 配置:`shoplift/configs/pipeline.cc015_v7b_classw102_test.yml`、
  `shoplift/configs/pipeline.d03_v7b_m80_classw102_test.yml`

### 8.6 已实现:"私有容器进公有容器"两段式规则(2026-09-10 定稿并通过服务器验收)

用户 2026-09-09 决策:cc015 的偷盗(把私有容器 bag/backpack 放入公有容器 basket/cart
后再转移商品)由**新规则**检出,不修属性模型的 holding_product 小误检(§8.4.1 决策)。
2026-09-10 该规则完成实现、本地全量调参与服务器跑批验收。

#### 8.6.1 触发语义(两段式)

1. **第一段(武装)**:私有容器(canonical `bag`)进入**店内篮**(`source_category=basket`)
   且包底边越过篮口线 `mouth_margin_px=20` 连续 `min_entry_frames=3` 帧 → 进入 hiding;
   随后包**检出消失**达 `hide_confirm_frames=2` 帧 → **armed**(武装),并绑定 **owner**。
2. **第二段(触发)**:武装后**该包的主人**累计"手中心落在篮框内"的帧数
   ≥ `owner_dwell_frames=115` 帧(**4.6s**,25fps) → 触发
   `private_container_concealment`(HIGH 0.78)。
3. 解除条件:武装后包再次可见 > `blip_tolerance_frames=40` 帧(视为取出/暴露),
   或武装后 owner 无手活动超 `idle_expire_ms=12s`。

#### 8.6.2 关键机制

- **owner 绑定**:包可见期间做短时跟踪(帧间 IoU≥0.3、gap≤3),把"手中心距包中心
  <80px"的人登记为该包主人;武装时用最近消失的包 track 的 owner,兜底回看武装前
  40 帧"手入篮"多数者。→ cc015 绑定成功为 **person-71**(f510-552 携带期)。
- **锚点去重**:每帧 basket 检测先 NMS(同类 IoU>0.35);锚点跨帧匹配 IoU≥0.2,
  重叠锚(IoU>0.35)合并状态 → 消除同一物理篮的**双事件**(实测 f829/f830 类重复)。
- **防误报闸门**:只认 `source_category=basket`。管线把容器 category 归一为
  `basket`,但 `attributes.source_category` 保留原值 → **购物车 `cart`(d03 34247 个)
  与顾客自有袋 `plastic_bag`(d03 45135 个)被天然排除**;d03 中 `source=basket`
  的检测仅 2 个 → 0 武装、0 误报。
- **避开降档关键字**:reason tags 使用 `private_container_entered_public_container` /
  `private_container_hidden` / `owner_hand_long_dwell_in_public_container` /
  `entry_temporal_consistent`,不含 `low_visibility`/`possible_occlusion`/
  `container_kind=normal` 等会被 `RiskRuleValidator` 压到 medium 的键。

#### 8.6.3 用户决策记录(2026-09-10)

| 决策点 | 结论 |
| --- | --- |
| 事件名 | `private_container_concealment` |
| 公容器范围 | **仅 `basket`**(购物车 cart 内私包可见→不需要规则;顾客自有 plastic_bag 排除) |
| 包消失才武装 | 是(可见→不算,契合"偷盗被看见则不需规则") |
| margin | 必须用(20px,防"搁在篮口"被判进篮) |
| 触发信号 | **主人累计手在篮内停留**(cc015 为长时间停留,非频繁进出) |
| 阈值 | **115 帧 = 4.6s**(以秒为单位、在保住 cc015 真事件前提下取最大:上限 117 帧,留 2 帧缓冲) |
| MOT 同篮接续(71→172→183→208) | **不做**(用户 2026-09-10 选 B):避免事件 id 漂移与双事件,71 自身累计 117 帧为上限 |

#### 8.6.4 代码与配置

- 新增 `shoplift/events/nested_concealment.py`(`NestedConcealmentConfig` +
  `NestedConcealmentDetector`,锚点状态机 idle→hiding→armed);
- `shoplift/events/event_engine.py`:构造参数 `nested_concealment_config`,
  `process_frame` 内跑探测器并把事件并入结果(过 `RuleValidator` 后仍为 HIGH);
- `shoplift/cli/offline_analyze.py`:把 `rules.nested_concealment` 传入引擎;
- `shoplift/configs/rules_loader.py`:`_SECTIONS`/`RulesConfigBundle` 增
  `nested_concealment` 段(**同时修掉 `from_mapping` 漏传该字段导致 yml 参数不生效的 bug**);
- `shoplift/configs/rules.hi384_geometry_v1.yml`:新增 `nested_concealment:` 段
  (Q=115)、`risk_scoring.action_type_weights.private_container_concealment: 0.38`、
  `event_types.supported` 增该类型;
- 单测 `shoplift/tests/test_nested_concealment.py`(6 项:真事件 HIGH、非 owner 不计、
  停留不足不触发、cart source 不武装、包持续可见不触发、引擎级集成)→ 全绿。

#### 8.6.5 服务器验收(2026-09-10,classw102 模型)

跑批命令(注意 `video_infer_visualize` 的 `--input` 默认值 `datasets/test_videos`
会覆盖 yml,必须显式传视频路径):

```bash
SHOPLIFT_USER=wqh nohup <env-python> -m shoplift.cli.video_infer_visualize \
  --config shoplift/configs/pipeline.cc015_v7b_classw102_test.yml \
  --input datasets/test/test_videos/yulong_store/cc015bb3e050a8bc67ebbb2e6fd9951d.mp4 \
  --output outputs/wqh/container_pa_test/cc015_v7b_classw102_v13test \
  > outputs/wqh/container_pa_test/cc015_v7b_classw102_v13test/run.log 2>&1 &
# d03 同理:--config pipeline.d03_v7b_m80_classw102_test.yml
#          --input datasets/test/test_videos/yulong_store/D03_堂食区_修复版.mpg
#          --output outputs/wqh/container_pa_test/d03_v7b_m80_classw102_v13test
```

验收结果(输出目录 `*_v13test/`,不覆盖已验收的 classw102 基线目录):

| 场景 | 基线 events | v13 events | 结果 |
| --- | --- | --- | --- |
| cc015 | 66(64 near_body + 2 bag_concealment) | **67** | **+1**:`private_container_concealment` / person-71 / **33360ms(33.36s)** / **high** / 0.78;既有 66 条 key 完全一致(新增 1、消失 0) |
| d03 | 57(53 near_body + 4 bag_concealment) | 57 | events.json **字节级完全相同**(md5 `8f616215cf40`)→ **0 新增误报** |

事件内部状态:`anchor=nested-pub-12, arm_frame_id=617, owner=person-71,
owner_dwell_frames=115, public_container_kind=store_basket`。
本地真实数据回放(2300 帧 cc015 / 22601 帧 d03)与服务器结果一致;
左缘人群区的 6 个早期误触发(person-379/340/385/386/422)与 person-378 边界误报
(累计仅 50-59 帧 < 115)均被 owner 绑定 + 阈值 + source 闸门清除。

> 后续可选项(未做,留待需要时):同人 MOT 接续(71→172→183→208,累计可达 ~250 帧),
> 可使阈值提到 7-8s,但事件会在延续段触发且需处理事件 id 归属。

> 前置依赖: §8.4.1 的自重叠 FP 未影响本规则验收(已被 source 闸门与 owner 绑定消化)。
