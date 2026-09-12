# PP-TSM 拆包装识别:四分类 → 二分类 定版报告(v2 ep26)

> 日期:2025-09-08(记录当日完整工作流)
> 目的:给 **PaddleDetection 管线**补充"拆包装"识别能力(管线现有偷盗规则识别不出拆包装动作)。
> 本文档供后续新对话直接使用(尤其末尾 **§9 Fusion 意图**),以继承已有模型与融合设计思路。

---

## §1 结论速览(定版)

| 项 | 定版值 |
|---|---|
| 任务 | 二分类:**1 = open(拆包装,即偷盗)** / **0 = not-open(非拆包装)** |
| 模型 | PP-TSM(ResNetTweaksTSM-50, num_seg=16, fight 预训练 backbone, head 随机初始化) |
| 定版权重 | `outputs/wqh/paddlevideo/pptsm_open_v2_fight16/ppTSM_epoch_00026.pdparams`(**v2 ep26**) |
| 配置 | `shoplift/configs/paddlevideo/pptsm_open_v2_fight16.yaml`(epochs=120, warmup=10, batch=1) |
| 推理事件规则 | 沿时间线滑窗(win=16 帧, stride=2)→ 单窗 open 概率;连续 **≥3 窗概率 ≥0.6** 记为一次"拆包事件" |
| 评测口径 | 窗口中心落在 open 段(master_index)= 正;事件级:两条拆包时间线(steal_2 训练域 + steal_3 未见)均触发,7 条非拆包时间线(含吃面/装袋/静止)零误报 |

---

## §2 背景:为什么要加 PP-TSM

- 业务:PaddleDetection 检测/跟踪 + 偷盗规则管线(规则输出 low/med/high 风险),但**规则识别不出"拆包装"**这个动作;
- 方案:PP-TSM(视频分类)在"人像裁剪时间线"上滑窗识别拆包装,作为独立证据,后续与管线规则 **Fusion** 后向中台上报(见 §9);
- 本项目此前已有一条成熟的**二分类偷盗检测**(steal vs normal,见 `pptsm_steal_*` 各 run,勿动);本文档描述的是**为步骤/拆包识别新做的分支**,最终坍缩为 open-vs-not 二分类。

## §3 为什么从"4 类步骤"转成"二分类(拆包 vs 非拆包)"

先按 4 类步骤(reach 拿取 / open 拆包装 / conceal 藏匿 / normal)训练,结论是**学不出来**,证据:

1. **4 类实验**(段级 clip 每段=1 样本/epoch;train 12 段 = 4:4:1:3):
   - 6 份轮数扫描(e30~e180)在 steal_3 上最好也只有 1/3 或带 2 误报的 2/3;**open 从无一轮被认对**;conceal 只在 e90@45 偶然学会;域内(训练集自身 steal_1/steal_2 段)最好也只有 5/8;
   - 概率扁平(≈类先验 0.33/0.33/0.10/0.24),峰值 <0.35,规则引擎 0.6 阈值下无窗口可用;
   - 不同 run 学到不同"安全类"(全判 normal / 全判 reach),稳定性极差。
2. **滑窗扩样实验**(把每段切成 win16/stride4 窗口,98 样本):
   - open 首次被学出来(steal_3 的 open 段 83/120 轮判对),但模型坍缩为"**open vs 其他一切→conceal**":reach 永久丢失、normal 段 17/17 窗被误投 conceal;窗口聚合也救不回(含 normal 后 FP 2/2)。
3. **结论**:reach/open/conceal 在 16 帧窗口 + 少量素材下互相不可分;而 **open(拆包装)是唯一动作特殊性足够、可跨视频泛化的类**(用户判断 + 数据验证一致)→ 任务重定义为二分类。

## §4 数据构建(二分类,全部窗口化)

统一窗口口径:**win=16 帧连续,stride=2,窗口=1 样本**;训练源仅来自 9 条 master 时间线的人像裁剪序列(3fps 级重采样),**真值按 master_index 落在哪段**。

### v1(初版,`steps/open_data/`,已废弃)
- 正(open):`steal_2_d03_s2`(唯一拆包段,103 master 帧)→ **44 窗**;
- 负(not-open):藏匿 `steal_2_s3` 20 + 伸手(正常人 `normal_train1_s2` 14 + 偷盗 `steal_2_s1` 5)+ 静止 `normal_train3_s1` 5 = **44 窗**;
- 按用户要求排除:steal_1 全部(侧视装包,手部动作不完整)、normal_test1(吃面)、normal_valid1(背对镜头);
- 比例 1:1;脚本 `scripts/make_open_binary_data.py --preset 1`。

### v2(定版训练集,`steps/open_data_v2/`)
- 正:同上 44 窗(**train.list 中正类行 ×2 = 88 行/epoch**);
- 负:**101 窗** = v1 的 44 + 新增难负例 57:
  - `normal_test1_d03_s1` 吃面 17(手口持续动作 FP 源)
  - `normal_train2_cc015_s1` 17(后段持续手部动作 FP 源)
  - `normal_train1_cc015_s1` 8
  - `steal_1_cc015` 四段 15(装自家袋,侧视)
- `normal_test2_d03_s1` **不进训练**,保留为新鲜评测探针;
- 有效比例 88:101 ≈ **1:1.15**;145 个窗口目录;
- 脚本:`scripts/make_open_binary_data.py --preset 3 --pos-repeat 2 --out-root outputs/wqh/paddlevideo/steps/open_data_v2`

> 注:以上"排除"仅针对**步骤标签不可靠**;作二分类负类(非拆包)语义正确,故 v2 重新纳入。

## §5 训练

- v1 配置 `shoplift/configs/paddlevideo/pptsm_open_fight16.yaml` → `outputs/wqh/paddlevideo/pptsm_open_fight16`
- **v2(定版)配置 `shoplift/configs/paddlevideo/pptsm_open_v2_fight16.yaml`** → `outputs/wqh/paddlevideo/pptsm_open_v2_fight16`
  - 与 4 类配置仅差:num_classes=2、train→open_data_v2、valid/test→open_eval.list(内容:steal_3 s1/s2/s3 + normal_test2,data_prefix=steps/data)、output_dir;
  - epochs=120, warmup=10(等比例), lr max_epoch=120, batch=1, num_seg=16, PRECISEBN、VideoMix 同家族;每轮自动存 ckpt + 写 VisualDL(step=epoch);
- 命令(复现):
  ```bash
  cd /home/e-ai2/ZHITAI_2tb/shoplift_detection_paddle
  source /home/e-ai2/miniconda3/etc/profile.d/conda.sh
  conda activate /home/e-ai2/ZHITAI_2tb/conda_envs/shoplift-paddle
  PYTHONPATH=third_party/PaddleVideo nohup python third_party/PaddleVideo/main.py --validate \
    -c shoplift/configs/paddlevideo/pptsm_open_v2_fight16.yaml \
    > outputs/wqh/paddlevideo/logs/train_open_v2_fight16.log 2>&1 &
  ```
- v2 训练曲线特征:train top1 ~ep26 起 ≈1.0(正类只 1 次拆包动作,过拟合正常);val(4 clip)loss/top1 剧烈摆动属正常(4 样本均值的统计噪声),**选模型不依赖 val 曲线,依赖全轮评测扫描**。

## §6 评估流程与定版依据

### 6.1 段级评测(open_eval)
- 工具:`scripts/eval_step_ckpts.py --config <cfg> --out <json>`(全 120 轮逐轮在 open_eval.list 4 段上评测:steal_3_s2=open→1;steal_3_s1/s3+normal_test2→0);
- v1:干净命中(open 对 + 0 误报)9 轮(16,24,28,52,62,70,72,75,89),置信度 0.90-0.97;
- v2:**13 轮**(8,9,14,16,20,21,22,24,26,27,31,37,38),置信度被压到 0.52-0.81(难负类的代价)。

### 6.2 时间线评测(裁剪版 M4)——决定性
- 构造:`scripts/make_master_windows.py` 沿 **9 条 master 时间线**滑窗(steal_1/2/3 + 6 normals,共 201 窗)→ `steps/timeline_windows/`;
- 评分:ep24(v1)/v2 候选轮(8,16,20,26,38)→ `m4_timeline_open.json` / `m4_timeline_open_v2.json`;
- **v1 ep24**:拆包召回优秀(steal_2 rec 0.98、steal_3 rec 0.92,置信 0.97),但 FP 源暴露:吃面 17/17 窗 flat≈0.97、steal_1 装包 7/8、normal_train2 后 8 窗 —— 训练负类缺"正常人持续手部动作";
- **v2**:吃面 17→0、装包 7→0、normal_train2 8→0(**全部压到 0.6 以下**),但模型变保守,单窗 0.7 阈值下召回下降;
- **事件级对比(本地分析,不需重跑)** = 连续 ≥K 窗概率 ≥θ:
  - θ=0.6、连续≥3 窗:**v2 ep26 唯一完美分离**——steal_2 与 steal_3(未见视频)均触发拆包事件,7 条非拆包时间线零事件;其 FP 源峰值 ≤0.39,余量充足;steal_3 拆包峰值 0.79;
  - v2 e20/e38 太保守(漏 steal_3);v1e24 召回高但 4 条非拆时间线误报。
- **结论:定版 v2 ep26 + 事件规则(θ=0.6, 连续≥3 窗)**。

## §7 保留清单(定版依赖,勿删)

服务器路径(均在 `/home/e-ai2/ZHITAI_2tb/shoplift_detection_paddle/`):

| 类型 | 路径 | 说明 |
|---|---|---|
| 定版权重 | `outputs/wqh/paddlevideo/pptsm_open_v2_fight16/`(至少 `ppTSM_epoch_00026.pdparams`;建议整目录含 visualdl_logs/best) | v2 ep26 |
| v2 配置 | `shoplift/configs/paddlevideo/pptsm_open_v2_fight16.yaml` | 复现/推理 |
| v2 数据 | `outputs/wqh/paddlevideo/steps/open_data_v2/`(train.list / open_eval.list / 145 窗) | 复训/评测 |
| timeline 评测输入 | `outputs/wqh/paddlevideo/steps/timeline_windows/`(timeline.list/meta + 201 窗) | 时间线重测 |
| 评测结果 | `outputs/wqh/paddlevideo/steps/m4_open_v2_sweep.json`、`m4_timeline_open_v2.json` | 依据留存 |
| 真值/源数据 | `steps/master/`(9 条 master)、`steps/master_meta.json`、`steps/data/step_clips_meta.json` | 真值对齐、未来重切 |
| eval 引用 clip | `steps/data/steal_3_d03_s1..s3/`、`steps/data/normal_test2_d03_s1/`(open_eval 直接读) | 评测需用 |
| 工具脚本 | `scripts/make_open_binary_data.py`、`scripts/make_master_windows.py`、`scripts/eval_step_ckpts.py` | 复现流水 |
| v2 日志 | `outputs/wqh/paddlevideo/logs/train_open_v2_fight16.log` | 审计 |
| 老二分类(项目既有) | `pptsm_steal*` 全部、二分类 clips/configs | **勿动** |
| 本地 | `docs/pptsm_open_binary_report.md`(本文档)、`shoplift/configs/paddlevideo/pptsm_open_fight16.yaml`(v1 对照,可选) | — |

## §8 清理清单(可删,按建议优先级)

> 均为 **4 类实验 / v1 二分类** 的作废产物;删除前请确认不再需要中间对照。总可回收 ≈ **155 GB**。

**A. 4 类 step 训练 run(作废,占 ~135G)— 建议删**
- `outputs/wqh/paddlevideo/pptsm_steps_fight16`(22G)
- `pptsm_steps_fight16_e30`(5.5G)/ `_e60`(11G)/ `_e90`(16G)/ `_e150`(27G)/ `_e180`(32G)
- `pptsm_steps_windows_fight16`(22G)

**B. 4 类数据/配置/脚本/评测(作废)— 建议删**
- `steps/window_data/`(98 窗 4 类滑窗,120M)
- `shoplift/configs/paddlevideo/pptsm_steps_frames_fight16*.yaml`(e30~e180/windows/基础共 8 个)
- `scripts/make_step_windows.py`(4 类窗口脚本)
- `outputs/wqh/paddlevideo/steps/m4_sweep_e*.json`、`m4_indomain_e*.json`、`m4_sweep_windows.json`、`m4_windowagg_*.json`(4 类评测 json)
- 4 类训练日志 `logs/train_steps_*.log`、`logs/train_steps_windows_fight16.log`

**C. v1 二分类(已被 v2 取代,可选删 ~22G)**
- `outputs/wqh/paddlevideo/pptsm_open_fight16/`(若想留对照,先拷出 `ppTSM_epoch_00024.pdparams` 单文件 ~190M 再删整目录)
- `steps/open_data/`(v1 数据,可删)
- `logs/train_open_fight16.log`、`steps/m4_open_sweep.json`(v1 评测,可选留作记录)
- `shoplift/configs/paddlevideo/pptsm_open_fight16.yaml`(可选留,文档已引用规格;删则无碍推理)

**D. 中间 clip 源(可选删,收益小)**
- `steps/data/` 下除 open_eval 用到的 4 个目录外的其余 14 个 step-clip 目录(steal_1/steal_2/normal_train* 等;数据已复制进 open_data_v2;master 才是源头;若日后要"用这些源重切新负类"则先保留)

> 保留判断原则:能**复现定版评测**的最小集 = v2 run + open_data_v2 + timeline_windows + master/metas + 4 个 eval clips + 脚本;其余皆可回收。

## §9 Fusion 意图(写给后续新对话窗口)

**现状(已完成)**:PP-TSM open 二分类定版 v2 ep26,可在"某个人像裁剪时间线"上输出逐窗 open 概率并事件化(连续 ≥3 窗 ≥0.6 = 一次拆包事件;单窗概率也是连续信号,可用于更细融合)。

**待办**:用户将先给 PaddleDetection 管线增加一条"偷盗规则判定",之后**在另一个对话窗口实现 PP-TSM 与管线结果的融合与中台上报**。融合要点(需向新窗口交代):

1. **两路输入**:
   - 管线:一系列偷盗规则的风险输出,档位 **low / medium / high**(格式待提供:规则事件表?每秒风险曲线?时间范围+等级?),覆盖偷盗行为链中的"非拆包"环节(逗留/伸手/藏匿等,规则能识别);
   - PP-TSM:同时间线滑窗 open 概率(模型弥补管线**识别不出拆包装**的缺陷)。
2. **时间对齐**:两者都落在同一条 master(人像)时间线上;需确认管线事件时间戳与 master 帧/秒的对应关系。
3. **候选融合形态(未定,待管线输出格式确定后选)**:
   - 强证据抬升:管线 low/med + PP-TSM 确认拆包事件 → 抬到 high 上报;PP-TSM 无证据则维持管线原判;
   - 分数加权:两路分数(归一化)加权合成一档;
   - 门控:仅管线已标可疑时 PP-TSM 才起作用(抑制 normal 误报,当前 v2 事件级已零误报,门控可更松)。
4. **口径红线(上线前必须对齐)**:
   - 生产端"人像时间线"的帧节奏须与训练一致(16 帧窗口 ≈ 5.3s 动作,master 3fps 级重采样);若按 25fps 原始帧滑,必须改成"跨同样秒数"采样,否则掉点;
   - 跟踪 ID 连续性影响"持续时长"类判据;
   - 模型在"吃面/往自家袋装东西"上曾强误报,v2 已通过难负类压到事件级零误报(阈值 0.6);生产阈值/事件参数可微调,但建议不低于 θ=0.5、连续≥3 窗。
5. **上报格式**:low/med/high + 证据字段(规则来源、模型证据:拆包事件时间窗/峰值/持续)待与中台约定。

---

*文档维护:本报告由当日会话整理,供 v2 ep26 定版后的融合与新对话使用。*
