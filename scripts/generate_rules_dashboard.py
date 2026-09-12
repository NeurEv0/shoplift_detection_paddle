"""生成"商店盗窃检测规则总览"静态单文件仪表盘 (rules_dashboard.html)。

用途: 把最新管线的所有规则(触发链路 tags + 评分标准)可视化, 方便团队查看,
      不再依赖单个 markdown 文档。

产出: 仓库根目录 rules_dashboard.html (自包含, 无后端/无外部 CDN, 浏览器直开)。

数据来源:
  1. 参数动态读取 shoplift/configs/rules.hi384_geometry_v1.yml (association /
     risk_scoring / rules / nested_concealment / event_types), 保证与管线同步;
  2. 规则链路模型(关系→tags→状态机→事件)与 tag 语义维护在本文件 RULE_MODEL /
     TAG_GLOSSARY 中, 对应 shoplift/tracking/association.py、
     shoplift/events/state_machine.py、shoplift/events/event_engine.py、
     shoplift/events/nested_concealment.py、shoplift/rules/risk_score.py、
     shoplift/rules/validators.py 的常量与逻辑。

重新生成: python scripts/generate_rules_dashboard.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - 仅在缺少 pyyaml 的环境提示
    print("需要 pyyaml: pip install pyyaml", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]
RULES_YML = ROOT / "shoplift" / "configs" / "rules.hi384_geometry_v1.yml"
OUTPUT_HTML = ROOT / "rules_dashboard.html"

# ---------------------------------------------------------------------------
# 规则链路模型 (手维护, 与源码常量对齐)
# step.kind: relation(关系) | tag(标签) | state(状态机) | event(事件)
# ---------------------------------------------------------------------------
RULE_MODEL = [
    {
        "id": "near_body_suspicious",
        "name": "近身可疑 (near_body_suspicious)",
        "zh": "手接触商品 + 跟随/进容器(未确认藏匿) → 近身可疑",
        "family": "接触/跟随",
        "target_level": "low / medium",
        "chain": [
            {"kind": "relation", "label": "hand_item_contact", "note": "手与商品连续接触 ≥ min_contact_frames(3) 帧"},
            {"kind": "tag", "label": "temporal_consistent", "note": "连续计数(帧间隔 ≤ max_frame_gap)"},
            {"kind": "tag", "label": "motion_aligned", "note": "手/商品运动方向余弦 ≥ min_motion_cosine(0.35) 时加分"},
            {"kind": "relation", "label": "item_follow_person / item_enter_container / item_disappeared_after_entry", "note": "三种关系任一(需 ≥2 种关系类型)"},
            {"kind": "state", "label": "item_picked → near_body_or_container", "note": "未进入 confirmed_risk_event"},
            {"kind": "event", "label": "near_body_suspicious", "note": "基础权重 0.18"},
        ],
    },
    {
        "id": "bag_concealment",
        "name": "私包藏匿 (bag_concealment)",
        "zh": "商品拿取 → 进入私有容器(包) → 商品消失 → 确认藏匿",
        "family": "藏匿",
        "target_level": "high",
        "chain": [
            {"kind": "relation", "label": "hand_item_contact", "note": "拿取: 手接触商品"},
            {"kind": "state", "label": "item_picked", "note": "接触成立进入取货态"},
            {"kind": "relation", "label": "item_enter_container", "note": "商品进入私有容器(bag, 含包口扩张 mouth_margin=80)"},
            {"kind": "tag", "label": "entered_private_container + entry_temporal_consistent", "note": "连续 ≥ min_entry_frames(3) 帧"},
            {"kind": "state", "label": "suspected_concealment", "note": "进入私/特/衣容器 → 疑似藏匿"},
            {"kind": "relation", "label": "item_disappeared_after_entry", "note": "进入后商品检出消失 ≥ disappeared_after_entry_frames(10) 帧"},
            {"kind": "tag", "label": "item_disappeared + after_container_entry + after_private_container_entry", "note": "消失 + 入私容器后消失"},
            {"kind": "state", "label": "confirmed_risk_event", "note": "确认风险事件"},
            {"kind": "event", "label": "bag_concealment", "note": "权重 0.32 + private 0.16"},
        ],
    },
    {
        "id": "clothing_concealment",
        "name": "衣物藏匿 (clothing_concealment)",
        "zh": "商品进入衣物/口袋扩展区 → 消失 → 确认藏匿",
        "family": "藏匿",
        "target_level": "high",
        "chain": [
            {"kind": "relation", "label": "item_enter_container", "note": "进入扩展区(clothing_region/pocket_region)"},
            {"kind": "tag", "label": "entered_clothing_region", "note": "衣物区域入口"},
            {"kind": "state", "label": "suspected_concealment", "note": ""},
            {"kind": "relation", "label": "item_disappeared_after_entry", "note": ""},
            {"kind": "tag", "label": "after_clothing_region_entry", "note": "入衣物区后消失"},
            {"kind": "state", "label": "confirmed_risk_event", "note": ""},
            {"kind": "event", "label": "clothing_concealment", "note": "权重 0.34 + clothing 0.18"},
        ],
    },
    {
        "id": "special_container_concealment",
        "name": "特殊容器藏匿 (special_container_concealment)",
        "zh": "商品进入特殊容器(婴儿车/头盔) → 消失 → 确认藏匿",
        "family": "藏匿",
        "target_level": "high",
        "chain": [
            {"kind": "relation", "label": "item_enter_container", "note": "进入 special 容器(stroller/helmet)"},
            {"kind": "tag", "label": "entered_special_container", "note": ""},
            {"kind": "state", "label": "suspected_concealment", "note": ""},
            {"kind": "relation", "label": "item_disappeared_after_entry", "note": ""},
            {"kind": "tag", "label": "after_special_container_entry", "note": ""},
            {"kind": "state", "label": "confirmed_risk_event", "note": ""},
            {"kind": "event", "label": "special_container_concealment", "note": "权重 0.34 + special 0.18"},
        ],
    },
    {
        "id": "bulk_pickup_to_bag",
        "name": "批量拿取入袋 (bulk_pickup_to_bag)",
        "zh": "同人同容器窗口内累计 ≥3 件商品藏匿 → 批量拿取",
        "family": "批量",
        "target_level": "high",
        "chain": [
            {"kind": "state", "label": "confirmed_risk_event (≥1 藏匿事件)", "note": "先有藏匿类事件(同 person+container)"},
            {"kind": "tag", "label": "bulk_item_count ≥ bulk_item_count_threshold(3)", "note": "窗口内同容器累计商品数 ≥3"},
            {"kind": "event", "label": "bulk_pickup_to_bag", "note": "权重 0.38; 数量≥3 时 score 抬到 ≥0.78"},
        ],
    },
    {
        "id": "normal_container_placement",
        "name": "正常容器放置 (normal_container_placement)",
        "zh": "商品进入正常容器(篮/购物车/结账袋) → 豁免降级, 不产出事件",
        "family": "豁免 · 不产出事件",
        "target_level": "豁免",
        "chain": [
            {"kind": "relation", "label": "item_enter_container", "note": "进入 normal 容器(basket/cart/checkout_bag)"},
            {"kind": "tag", "label": "entered_normal_container + normal_container", "note": ""},
            {"kind": "state", "label": "resolved_or_downgraded", "note": "正常容器 → 解除/降级"},
            {"kind": "tag", "label": "normal_container_exempted", "note": "消失时打豁免标签"},
            {"kind": "event", "label": "不产出事件 (仅 suggested 类型)", "note": "不在 supported 清单; 权重 0.05 仅作兜底, 正常购物 score ≤0.44"},
        ],
    },
    {
        "id": "private_container_concealment",
        "name": "私包藏入店内篮 (private_container_concealment)",
        "zh": "两段式: 私包放入店内篮(source=basket)后消失(武装) → 包主人累计伸手入篮 ≥4.6s → 事件",
        "family": "两段式新规则",
        "target_level": "high",
        "chain": [
            {"kind": "relation", "label": "bag 底边越过篮口线 (mouth_margin=20px) 连续 ≥3 帧", "note": "私有容器(canonical bag)进店内篮(仅 source=basket)"},
            {"kind": "tag", "label": "private_container_entered_public_container", "note": "进入公有容器"},
            {"kind": "relation", "label": "包检出消失 ≥ hide_confirm_frames(2) 帧", "note": "被篮壁遮挡 → 武装, 并绑定包主人"},
            {"kind": "tag", "label": "private_container_hidden", "note": "消失确认 → armed"},
            {"kind": "relation", "label": "owner 手中心落在篮框内累计 ≥ owner_dwell_frames(115 帧 = 4.6s)", "note": "主人(放入前手近袋者)长停留; 非 owner 不计"},
            {"kind": "tag", "label": "owner_hand_long_dwell_in_public_container", "note": ""},
            {"kind": "event", "label": "private_container_concealment", "note": "固定分 0.38+0.15+0.15+0.10 = 0.78 (high)"},
        ],
        "note": "触发信号 = 仅\"主人手在篮内长时间停留\"(owner 累计 ≥115 帧=4.6s)。"
                "\"手频繁进出(≥3 次 × 单次≥15 帧)\"是早期 v1.2 草案机制, 按 2026-09-10 定稿移除"
                "(cc015 手频繁进出很少, 以长时间停留触发); 详见 docs/stage_b §8.6.3 决策表。",
    },
]

# ---------------------------------------------------------------------------
# reason_tags 语义表 (与源码 emitted tags 对齐)
# ---------------------------------------------------------------------------
TAG_GLOSSARY = [
    ("hand_item_distance_close", "接触", "手框与商品框距离 ≤ 阈值(max_contact_distance_px 与尺寸比取大)"),
    ("hand_item_overlap", "接触", "手/商品框 IoU ≥ min_contact_iou 或中心互入"),
    ("hand_item_near", "接触", "未命中距离/重叠时的兜底接触标签"),
    ("temporal_consistent", "接触/入口", "连续计数跨帧成立(帧间隔 ≤ max_frame_gap)"),
    ("motion_aligned", "接触", "手与商品运动方向余弦 ≥ min_motion_cosine"),
    ("low_confidence", "接触/入口", "手或商品/容器置信度 < low_confidence_score(0.35)"),
    ("item_owned_by_person", "跟随", "商品已归属某人的标记(候选得分 >0)"),
    ("hand_item_contact_owner", "跟随", "归属得分来自该人先前的接触证据"),
    ("previous_item_owner", "跟随", "该人在前一帧已是该商品主人"),
    ("item_near_person", "跟随", "商品中心落入扩展人体框"),
    ("follow_motion_aligned", "跟随", "商品与人体运动方向一致"),
    ("gap_filled", "跟随", "商品短时缺失后按主人延续补齐(max_missing_frames 内)"),
    ("entered_private_container", "入口", "商品进入私有容器(bag, 含包口扩张)"),
    ("entered_special_container", "入口", "商品进入特殊容器(stroller/helmet)"),
    ("entered_clothing_region", "入口", "商品进入衣物/口袋扩展区"),
    ("entered_normal_container", "入口", "商品进入正常容器(basket/cart/checkout_bag)"),
    ("normal_container", "入口", "正常容器种类标记(伴随 entered_normal_container)"),
    ("entered_container", "入口", "未知容器种类入口兜底标签"),
    ("entry_temporal_consistent", "入口", "入口连续 ≥ min_entry_frames(3) 帧"),
    ("item_disappeared", "消失", "商品进入容器后检出消失 ≥ disappeared_after_entry_frames(10) 帧"),
    ("after_container_entry", "消失", "消失发生在此前有过容器入口"),
    ("after_private_container_entry", "消失", "入私容器后消失"),
    ("after_special_container_entry", "消失", "入特殊容器后消失"),
    ("after_clothing_region_entry", "消失", "入衣物区后消失"),
    ("normal_container_exempted", "消失/豁免", "入正常容器后消失 → 豁免(正常购物)"),
    ("low_visibility", "降级", "低可见度/严重遮挡(或入口低置信度转化) → 降档"),
    ("single_frame_contact", "降级", "单帧接触证据 → 降档 low"),
    ("bulk_pickup", "批量", "批量拿取 ≥ bulk_item_count_threshold(3)"),
    ("bulk_item_count", "批量", "窗口内同容器累计商品数标签"),
    ("bulk_pickup_to_bag", "批量", "批量拿取入袋事件类型标签"),
    ("private_container_entered_public_container", "两段式", "私包进入店内篮(公有容器)"),
    ("private_container_hidden", "两段式", "私包进入后被篮壁遮挡 → 武装"),
    ("owner_hand_long_dwell_in_public_container", "两段式", "包主人手在篮内累计停留 ≥ 阈值"),
    ("entry_temporal_consistent", "两段式", "两段式入口连续帧成立(与入口 tag 同名复用)"),
    ("normal_shopping", "校验", "校验器对正常容器事件补打的正常购物标签"),
]

STATE_GLOSSARY = [
    ("observing", "观察中: 无证据或已解除"),
    ("item_picked", "已取货: hand_item_contact 或 item_follow_person 成立"),
    ("near_body_or_container", "近身/进容器: item_enter_container 但非私/特/衣容器"),
    ("suspected_concealment", "疑似藏匿: 商品进入私/特/衣容器"),
    ("confirmed_risk_event", "确认风险: 入容器后商品消失(私/特/衣)"),
    ("resolved_or_downgraded", "解除/降级: 正常容器或低可见度/遮挡"),
]

RELATION_GLOSSARY = [
    ("hand_item_contact", "手-商品接触: 距离/重叠判定, 连续 ≥ min_contact_frames"),
    ("item_follow_person", "商品归属人: 跟随/接触/运动一致性打分, ≥ follow_score_threshold"),
    ("item_enter_container", "商品进入容器: 中心入框或重叠比 ≥ min_entry_overlap_ratio"),
    ("item_disappeared_after_entry", "进入后商品消失: 连续缺失 ≥ disappeared_after_entry_frames"),
]


def _load_rules() -> dict:
    return yaml.safe_load(RULES_YML.read_text(encoding="utf-8"))


def _fmt(v) -> str:
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def _build_payload(rules: dict) -> dict:
    association = rules.get("association", {})
    risk = rules.get("risk_scoring", {})
    rl = rules.get("rules", {})
    nested = rules.get("nested_concealment", {})
    event_types = rules.get("event_types", {}).get("supported", [])
    return {
        "meta": {
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_yaml": str(RULES_YML.relative_to(ROOT)) if RULES_YML.is_relative_to(ROOT) else str(RULES_YML),
        },
        "event_types": event_types,
        "rule_model": RULE_MODEL,
        "tag_glossary": [{"tag": t, "group": g, "note": n} for t, g, n in TAG_GLOSSARY],
        "state_glossary": [{"state": s, "note": n} for s, n in STATE_GLOSSARY],
        "relation_glossary": [{"relation": r, "note": n} for r, n in RELATION_GLOSSARY],
        "association": association,
        "risk_scoring": risk,
        "rules": rl,
        "nested_concealment": nested,
    }


# ---------------------------------------------------------------------------
# HTML 模板 (自包含; 无外部资源)
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>商店盗窃检测 · 规则总览仪表盘</title>
<style>
  :root{
    --bg:#0e1420; --panel:#151d2b; --panel2:#1a2436; --line:#263349;
    --text:#dbe4f0; --muted:#8ea0b8; --accent:#4da3ff; --accent2:#22c55e;
    --amber:#f5b83d; --red:#ff5f6d; --purple:#b18cff; --cyan:#38bdf8;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
    font-family:-apple-system,"Segoe UI","Microsoft YaHei",Roboto,sans-serif;line-height:1.6}
  header{position:sticky;top:0;z-index:10;background:rgba(14,20,32,.94);
    backdrop-filter:blur(6px);border-bottom:1px solid var(--line);padding:14px 28px;
    display:flex;align-items:center;gap:16px;flex-wrap:wrap}
  header h1{font-size:18px;margin:0;font-weight:650}
  header .sub{font-size:12px;color:var(--muted)}
  nav{display:flex;gap:8px;flex-wrap:wrap}
  nav a{color:var(--muted);text-decoration:none;font-size:13px;padding:5px 12px;
    border:1px solid var(--line);border-radius:20px;transition:.15s}
  nav a:hover{color:var(--text);border-color:var(--accent)}
  main{max-width:1200px;margin:0 auto;padding:24px 28px 80px}
  section{margin-bottom:40px}
  h2{font-size:20px;border-left:4px solid var(--accent);padding-left:12px;margin:0 0 6px}
  .lead{color:var(--muted);font-size:13px;margin:0 0 18px}
  .grid{display:grid;gap:16px}
  .g-cards{grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}
  .g-cols{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}
  .card h3{margin:0 0 10px;font-size:15px;display:flex;align-items:center;gap:8px}
  .badge{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:600;white-space:nowrap}
  .b-high{background:rgba(255,95,109,.15);color:var(--red)}
  .b-medium{background:rgba(245,184,61,.15);color:var(--amber)}
  .b-low{background:rgba(56,189,248,.15);color:var(--cyan)}
  .b-family{background:rgba(177,140,255,.15);color:var(--purple)}
  .chain{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-top:8px}
  .node{padding:6px 10px;border-radius:8px;font-size:12px;border:1px solid var(--line);
    max-width:230px;overflow-wrap:anywhere}
  .node .lbl{font-weight:600;word-break:break-all;overflow-wrap:anywhere}
  .node .nt{color:var(--muted);font-size:11px;display:block;overflow-wrap:anywhere}
  .n-relation{background:rgba(77,163,255,.12);border-color:rgba(77,163,255,.4)}
  .n-relation .lbl{color:var(--accent)}
  .n-tag{background:rgba(245,184,61,.12);border-color:rgba(245,184,61,.4)}
  .n-tag .lbl{color:var(--amber)}
  .n-state{background:rgba(34,197,94,.12);border-color:rgba(34,197,94,.4)}
  .n-state .lbl{color:var(--accent2)}
  .n-event{background:rgba(255,95,109,.12);border-color:rgba(255,95,109,.45)}
  .n-event .lbl{color:var(--red)}
  .arrow{color:var(--muted);font-weight:700}
  .rule-note{margin-top:10px;padding:8px 12px;border:1px dashed rgba(245,184,61,.5);
    background:rgba(245,184,61,.08);border-radius:8px;font-size:12px;color:var(--amber);
    overflow-wrap:anywhere}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top;word-break:break-word}
  th{background:var(--panel2);color:var(--muted);font-weight:600;white-space:nowrap}
  td.mono, .mono{font-family:"Cascadia Code",Consolas,monospace;font-size:12px;word-break:break-all;overflow-wrap:anywhere}
  .formula{background:var(--panel2);border:1px solid var(--line);border-radius:10px;
    padding:14px 16px;font-family:Consolas,monospace;font-size:13px;overflow-x:auto;white-space:pre-wrap}
  .pill{display:inline-block;font-size:11px;padding:1px 8px;border-radius:8px;margin:1px 2px;
    background:rgba(77,163,255,.12);border:1px solid rgba(77,163,255,.3);color:var(--accent)}
  .group-title{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;
    margin:16px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:13px}
  .kv dt{color:var(--muted);word-break:break-all;overflow-wrap:anywhere}
  .kv dd{margin:0;font-family:Consolas,monospace;font-size:12px;word-break:break-all}
  .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
  .legend span{display:flex;align-items:center;gap:5px}
  .dot{width:10px;height:10px;border-radius:3px;display:inline-block}
  footer{color:var(--muted);font-size:12px;text-align:center;padding:20px}
</style>
</head>
<body>
<header>
  <div>
    <h1>商店盗窃检测 · 规则总览仪表盘</h1>
    <div class="sub">规则触发链路(relation → tags → 状态机 → 事件) + 评分标准 · 生成于 {{GENERATED_AT}} · 来源 {{SOURCE_YAML}}</div>
  </div>
  <nav>
    <a href="#overview">事件总览</a>
    <a href="#chains">触发链路</a>
    <a href="#scoring">评分标准</a>
    <a href="#validator">校验降档</a>
    <a href="#params">参数表</a>
    <a href="#glossary">标签/状态释义</a>
  </nav>
</header>
<main>
  <section id="overview">
    <h2>一、事件类型总览</h2>
    <p class="lead">最新管线(rules.hi384_geometry_v1.yml)支持的事件类型;点击卡片可跳到对应触发链路。</p>
    <div class="grid g-cards" id="cards"></div>
  </section>

  <section id="chains">
    <h2>二、触发链路(完整 tags)</h2>
    <p class="lead">每条规则从底层关系(relation) → reason tags → 状态机(state) → 事件(event)的完整链路。</p>
    <div class="legend">
      <span><i class="dot" style="background:#4da3ff"></i>关系 relation</span>
      <span><i class="dot" style="background:#f5b83d"></i>标签 tag</span>
      <span><i class="dot" style="background:#22c55e"></i>状态机 state</span>
      <span><i class="dot" style="background:#ff5f6d"></i>事件 event</span>
    </div>
    <div id="chain-list"></div>
  </section>

  <section id="scoring">
    <h2>三、评分标准</h2>
    <p class="lead">风险分公式与各分量权重、得分上限(cap)与档位阈值。</p>
    <div class="formula" id="formula"></div>
    <div class="grid g-cols">
      <div class="card">
        <h3>事件类型基础权重 action_type_weights</h3>
        <div class="kv" id="action-weights"></div>
      </div>
      <div class="card">
        <h3>容器类别权重 container_type_weights</h3>
        <div class="kv" id="container-weights"></div>
      </div>
      <div class="card">
        <h3>分量权重与阈值</h3>
        <div class="kv" id="risk-params"></div>
      </div>
      <div class="card">
        <h3>得分上限与档位</h3>
        <div class="kv" id="caps"></div>
      </div>
    </div>
  </section>

  <section id="validator">
    <h2>四、校验 / 降档规则 (RiskRuleValidator)</h2>
    <p class="lead">事件产出后经校验器归一化;命中即打标并压低风险档与分数。</p>
    <div id="validator-table"></div>
  </section>

  <section id="params">
    <h2>五、完整参数表</h2>
    <p class="lead">association / risk_scoring / rules / nested_concealment 全量参数(直接读取 rules yml)。</p>
    <div id="params-table"></div>
  </section>

  <section id="glossary">
    <h2>六、标签 / 关系 / 状态 释义</h2>
    <div class="grid g-cols">
      <div class="card"><h3>reason_tags 全表</h3><div id="tag-table"></div></div>
      <div class="card"><h3>关系 relation 类型</h3><div id="relation-table"></div></div>
      <div class="card"><h3>状态机 state</h3><div id="state-table"></div></div>
    </div>
  </section>
</main>
<footer>由 scripts/generate_rules_dashboard.py 生成 · 自包含离线可用 · 参数动态读取 rules.hi384_geometry_v1.yml</footer>

<script>
const DATA = __DATA__;

const LV = {"high":"high","medium":"medium","low":"low","豁免":"low"};
function levelBadge(lv){
  const m = {"high":"b-high","medium":"b-medium","low":"b-low"};
  return '<span class="badge '+(m[lv]||'b-low')+'">'+lv+'</span>';
}

/* 一、事件总览卡片 */
function renderCards(){
  const el = document.getElementById('cards');
  el.innerHTML = DATA.rule_model.map(r=>`
    <div class="card">
      <h3>${r.name.split(' (')[0]}
        ${levelBadge(r.target_level)}
        <span class="badge b-family">${r.family}</span>
      </h3>
      <div style="font-size:13px;color:var(--muted);margin-bottom:8px">${r.zh}</div>
      <div class="mono" style="color:var(--accent)">${r.id}</div>
      <div style="margin-top:10px"><a href="#chain-${r.id}" style="color:var(--accent);font-size:12px">查看完整链路 →</a></div>
    </div>`).join('');
}

/* 二、触发链路 */
function renderChains(){
  const el = document.getElementById('chain-list');
  el.innerHTML = DATA.rule_model.map(r=>{
    const nodes = r.chain.map(s=>{
      const cls = "n-"+s.kind;
      return `<div class="node ${cls}"><span class="lbl">${s.label}</span>${s.note?`<span class="nt">${s.note}</span>`:''}</div>`;
    }).join('<span class="arrow">→</span>');
    return `<div class="card" id="chain-${r.id}" style="margin-bottom:14px">
      <h3>${r.name} ${levelBadge(r.target_level)} <span class="badge b-family">${r.family}</span></h3>
      <div style="font-size:13px;color:var(--muted)">${r.zh}</div>
      ${r.note?`<div class="rule-note">${r.note}</div>`:''}
      <div class="chain">${nodes}</div>
    </div>`;
  }).join('');
}

/* 三、评分标准 */
function renderScoring(){
  document.getElementById('formula').textContent =
`risk_score = action_weight
           + container_weight
           + continuous_component   (continuous_evidence_weight × min(1, evidence_frames/high_risk_min_evidence_frames))
           + diversity_component    (relation_diversity_weight × min(1, 关系类型数/3))
           + confidence_component   (model_confidence_weight × mean(证据score))
           + area_risk_component    (area_risk_weight × 区域风险因子 0/0.5/1)
           - normal_downgrade       (仅容器为 normal 时扣 normal_shopping_downgrade)

risk_level:  high ≥ high_threshold(0.75);  medium ≥ medium_threshold(0.45);  否则 low`;
  const aw = DATA.risk_scoring.action_type_weights||{};
  document.getElementById('action-weights').innerHTML = Object.entries(aw)
    .map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join('');
  const cw = DATA.risk_scoring.container_type_weights||{};
  document.getElementById('container-weights').innerHTML = Object.entries(cw)
    .map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join('');
  const rp = DATA.risk_scoring;
  const rpKeys = ['continuous_evidence_weight','relation_diversity_weight','model_confidence_weight',
    'area_risk_weight','normal_shopping_downgrade','high_risk_min_evidence_frames','bulk_item_count_threshold',
    'low_confidence_threshold'];
  document.getElementById('risk-params').innerHTML = rpKeys
    .map(k=>`<dt>${k}</dt><dd>${rp[k]}</dd>`).join('');
  const capKeys = ['low_visibility_score_cap','low_confidence_score_cap','single_frame_contact_score_cap',
    'medium_threshold','high_threshold'];
  document.getElementById('caps').innerHTML = capKeys
    .map(k=>`<dt>${k}</dt><dd>${rp[k]}</dd>`).join('');
}

/* 四、校验降档 */
function renderValidator(){
  const r = DATA.rules;
  const rows = [
    ["high 但 reason tags < "+r.high_risk_min_reason_tags+" 个","high_risk_requires_multiple_reason_tags","score 压到 < "+r.high_threshold+" (即 ≤0.74)"],
    ["high 但证据为单帧接触","single_frame_contact_must_not_be_high","降档 low, score ≤ "+r.single_frame_contact_score_cap],
    ["high 但容器为 normal(篮/车/结账袋)","normal_container_must_not_be_high","降档 medium, score ≤ "+r.normal_container_score_cap],
    ["high 但低可见度/严重遮挡","low_visibility_must_be_downgraded","降档 medium, score ≤ "+r.low_visibility_score_cap],
  ];
  document.getElementById('validator-table').innerHTML = `<table>
    <tr><th>触发条件</th><th>违规码</th><th>归一化动作</th></tr>
    ${rows.map(x=>`<tr><td>${x[0]}</td><td class="mono">${x[1]}</td><td>${x[2]}</td></tr>`).join('')}
  </table>`;
}

/* 五、参数表 */
function renderParams(){
  const secs = [
    ["association", DATA.association],
    ["risk_scoring", DATA.risk_scoring],
    ["rules", DATA.rules],
    ["nested_concealment", DATA.nested_concealment],
  ];
  document.getElementById('params-table').innerHTML = secs.map(([name,obj])=>{
    const rows = Object.entries(obj).map(([k,v])=>{
      let val = v;
      if(Array.isArray(v)) val = v.join(', ');
      else if(v && typeof v==='object') val = JSON.stringify(v);
      return `<tr><td class="mono">${k}</td><td class="mono">${val}</td></tr>`;
    }).join('');
    return `<div class="group-title">${name}</div><table><tr><th>参数</th><th>值</th></tr>${rows}</table>`;
  }).join('');
}

/* 六、释义 */
function renderGlossary(){
  const groups = {};
  DATA.tag_glossary.forEach(t=>{ (groups[t.group]=groups[t.group]||[]).push(t); });
  const order = ["接触","跟随","入口","消失","降级","批量","两段式","校验"];
  const tagHtml = order.filter(g=>groups[g]).map(g=>`
    <div class="group-title">${g}</div>
    <table><tr><th>tag</th><th>含义</th></tr>
    ${groups[g].map(t=>`<tr><td class="mono">${t.tag}</td><td>${t.note}</td></tr>`).join('')}
    </table>`).join('');
  document.getElementById('tag-table').innerHTML = tagHtml;
  document.getElementById('relation-table').innerHTML = `<table><tr><th>relation</th><th>含义</th></tr>
    ${DATA.relation_glossary.map(r=>`<tr><td class="mono">${r.relation}</td><td>${r.note}</td></tr>`).join('')}</table>`;
  document.getElementById('state-table').innerHTML = `<table><tr><th>state</th><th>含义</th></tr>
    ${DATA.state_glossary.map(s=>`<tr><td class="mono">${s.state}</td><td>${s.note}</td></tr>`).join('')}</table>`;
}

renderCards(); renderChains(); renderScoring(); renderValidator(); renderParams(); renderGlossary();
</script>
</body>
</html>
"""


def generate() -> None:
    rules = _load_rules()
    payload = _build_payload(rules)
    data_json = json.dumps(payload, ensure_ascii=False)
    html = (
        _HTML_TEMPLATE
        .replace("__DATA__", data_json)
        .replace("{{GENERATED_AT}}", payload["meta"]["generated_at"])
        .replace("{{SOURCE_YAML}}", payload["meta"]["source_yaml"])
    )
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"生成完成: {OUTPUT_HTML} ({len(html)} 字节)")


if __name__ == "__main__":
    generate()
