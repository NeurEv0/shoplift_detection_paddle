"""Live web dashboard for realtime_infer metrics + events (zero-dependency).

实时把 realtime_infer 产出的 metrics.csv / events.jsonl 画成浏览器折线图,
浏览器每 2s 轮询刷新。只依赖 Python 标准库(http.server),不装 Flask,前端用原生
canvas 手绘折线,不依赖外网 CDN。

用法(服务器, 自动指向最新一次 run):
  python -m shoplift.cli.monitor_dashboard --latest outputs/realtime --port 8080
  或显式指定某次 run:
  python -m shoplift.cli.monitor_dashboard \
      --metrics outputs/realtime/single_stream_20260911_184530/metrics.csv \
      --events  outputs/realtime/single_stream_20260911_184530/events.jsonl \
      --port 8080
然后浏览器打开 http://<服务器IP>:8080/。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


HTML = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Shoplift 实时监控</title>
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#111;color:#ddd;margin:16px}
  h1{font-size:18px;margin:0 0 6px}
  #status{font-size:13px;color:#9f9;margin-bottom:10px;font-family:ui-monospace,monospace}
  .chart{display:block;background:#181818;border:1px solid #333;margin:8px 0;width:100%}
  #events{font-size:12px;font-family:ui-monospace,monospace;background:#181818;border:1px solid #333;padding:8px;max-height:220px;overflow:auto}
  .ev{color:#fb3;white-space:nowrap}.ev.high{color:#f44;font-weight:bold}
  .muted{color:#777}
</style>
</head>
<body>
<h1>Shoplift 实时监控</h1>
<div id="status">等待数据...</div>
<canvas id="gpu" class="chart" width="1000" height="240"></canvas>
<canvas id="cpu" class="chart" width="1000" height="160"></canvas>
<canvas id="fps" class="chart" width="1000" height="180"></canvas>
<div id="events"><span class="muted">暂无事件</span></div>
<script>
function num(v){const n=Number(v);return Number.isFinite(n)?n:null;}
async function poll(){
  try{
    const r=await fetch('/metrics');
    const d=await r.json();
    render(d);
  }catch(e){}
}
function render(d){
  const s=d.samples||[];
  if(!s.length){return;}
  const last=s[s.length-1];
  document.getElementById('status').textContent =
    '样本 '+s.length+' | 最新: 显存 reserved='+fmt(last.gpu_reserved_mb)+
    'MB allocated='+fmt(last.gpu_allocated_mb)+
    'MB 进程='+fmt(last.gpu_process_used_mb)+
    'MB | 内存 RSS='+fmt(last.cpu_rss_mb)+
    'MB | fps='+fmt(last.fps)+' | 时延='+fmt(last.total_ms)+'ms';
  const t=s.map(x=>num(x.wall_s)||0);
  draw('gpu', t, [
    {name:'reserved', color:'#4af', data:s.map(x=>num(x.gpu_reserved_mb))},
    {name:'allocated', color:'#0d0', data:s.map(x=>num(x.gpu_allocated_mb))},
    {name:'process', color:'#f80', data:s.map(x=>num(x.gpu_process_used_mb))},
  ], 'GPU 显存 (MB)');
  draw('cpu', t, [{name:'RSS', color:'#fa0', data:s.map(x=>num(x.cpu_rss_mb))}], 'CPU 内存 (MB)');
  draw('fps', t, [
    {name:'FPS', color:'#4af', data:s.map(x=>num(x.fps))},
    {name:'latency_ms', color:'#f44', data:s.map(x=>num(x.total_ms))},
  ], '吞吐 / 时延');
  const ev=d.events||[];
  document.getElementById('events').innerHTML = ev.length
    ? ev.slice().reverse().map(e=>'<div class="ev '+(e.risk_level||'')+'">['+e.event_type+'] '+
        (e.person_track_id||'-')+' '+(e.risk_level||'')+' score='+fmt(e.risk_score)+'</div>').join('')
    : '<span class="muted">暂无事件</span>';
}
function fmt(v){return (v===undefined||v===null||v==='')?'-':Number(v).toFixed(0);}
function draw(id, t, series, title){
  const c=document.getElementById(id);
  const ctx=c.getContext('2d');
  const W=c.width,H=c.height;
  ctx.fillStyle='#181818';ctx.fillRect(0,0,W,H);
  const pad={l:48,r:12,t:20,b:24}, iw=W-pad.l-pad.r, ih=H-pad.t-pad.b;
  ctx.fillStyle='#aaa';ctx.font='12px ui-monospace,monospace';ctx.fillText(title,pad.l,14);
  if(!t.length)return;
  let all=[];series.forEach(s=>all=all.concat(s.data.filter(v=>v!==null)));
  if(!all.length)return;
  let mn=Math.min.apply(null,all), mx=Math.max.apply(null,all);
  if(mx-mn<1e-6){mx=mn+1;}
  const x0=t[0], x1=t[t.length-1]||1;
  const X=v=>pad.l+(v-x0)/(x1-x0)*iw;
  const Y=v=>pad.t+ih-(v-mn)/(mx-mn)*ih;
  ctx.strokeStyle='#333';ctx.fillStyle='#888';
  for(let i=0;i<=4;i++){
    const v=mn+(mx-mn)*i/4, y=Y(v);
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
    ctx.fillText(v.toFixed(0),4,y+4);
  }
  series.forEach(s=>{
    ctx.strokeStyle=s.color;ctx.lineWidth=1.5;ctx.beginPath();let started=false;
    for(let i=0;i<t.length;i++){
      const v=s.data[i];if(v===null)continue;
      const x=X(t[i]), y=Y(v);
      started?ctx.lineTo(x,y):(ctx.moveTo(x,y),started=true);
    }
    ctx.stroke();
  });
  let lx=pad.l;ctx.font='12px ui-monospace,monospace';
  series.forEach(s=>{
    ctx.fillStyle=s.color;ctx.fillRect(lx,H-17,10,10);
    ctx.fillStyle='#ddd';ctx.fillText(s.name,lx+13,H-8);
    lx+=13+ctx.measureText(s.name).width+22;
  });
}
setInterval(poll,2000);
poll();
</script>
</body>
</html>
"""


def _num(value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def read_csv_tail(path: str, n: int = 600) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: _num(v) for k, v in row.items()})
    return rows[-n:]


def read_events_tail(path: str, n: int = 80) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


_TIMESTAMP_DIR_RE = re.compile(r"_\d{8}_\d{6}(_\d+)?$")


def _find_latest_run(parent: Path) -> Path | None:
    """在 parent 下找最新的一次 run 输出目录(按时间戳字典序最大)。"""
    if not parent.is_dir():
        return None
    cands = [p for p in parent.iterdir() if p.is_dir() and _TIMESTAMP_DIR_RE.search(p.name)]
    if not cands:
        return None
    return max(cands, key=lambda p: p.name)


class _Handler(BaseHTTPRequestHandler):
    metrics_path: str = ""
    events_path: str | None = None

    def do_GET(self) -> None:
        if self.path.startswith("/metrics"):
            payload = {
                "samples": read_csv_tail(self.metrics_path),
                "events": read_events_tail(self.events_path or ""),
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, fmt: str, *args: Any) -> None:  # 静默,不刷屏
        return


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, default=None, help="realtime_infer 产出的 metrics.csv(不填则用 --latest 自动找最新 run)")
    p.add_argument("--events", type=Path, default=None, help="realtime_infer 产出的 events.jsonl(可选)")
    p.add_argument("--latest", type=Path, default=None, help="父目录(如 outputs/realtime),自动选最新的 *_YYYYMMDD_HHMMSS 子目录")
    p.add_argument("--host", default="0.0.0.0", help="监听地址,默认 0.0.0.0(局域网可访问)")
    p.add_argument("--port", type=int, default=8080, help="端口,默认 8080")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.metrics is None:
        parent = args.latest or Path("outputs/realtime")
        run_dir = _find_latest_run(parent)
        if run_dir is None:
            print(
                f"[dashboard] 未在 {parent} 下找到 *_YYYYMMDD_HHMMSS 输出目录,请用 --metrics 指定",
                file=sys.stderr,
            )
            return 2
        args.metrics = run_dir / "metrics.csv"
        if args.events is None and (run_dir / "events.jsonl").exists():
            args.events = run_dir / "events.jsonl"
        print(f"[dashboard] 自动选择最新 run: {run_dir}", flush=True)
    _Handler.metrics_path = str(args.metrics)
    _Handler.events_path = str(args.events) if args.events else None
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(
        f"[dashboard] http://{args.host}:{args.port}/  "
        f"(metrics={args.metrics}, events={args.events})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
