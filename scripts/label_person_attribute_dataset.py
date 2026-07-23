"""Lightweight local web annotator for person-attribute CSV datasets."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, quote, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shoplift.models.person_attribute.dataset import REQUIRED_COLUMNS  # noqa: E402
from shoplift.models.person_attribute.labels import HEAD_SPECS_BY_NAME  # noqa: E402


DEFAULT_DATASET = Path("datasets/person_attribute_yulong_store_crops")
DEFAULT_ANNOTATION = Path("full.csv")
LABEL_STATUS_COLUMN = "label_status"
LABEL_STATUS_REVIEWED = "reviewed"
LABEL_STATUS_SKIPPED = "skipped"
LABEL_STATUS_UNREVIEWED = "unreviewed"


@dataclass(frozen=True)
class DatasetPaths:
    dataset_dir: Path
    annotation_path: Path
    image_root: Path


class CsvAnnotationStore:
    def __init__(self, paths: DatasetPaths) -> None:
        self.paths = paths
        self._lock = threading.Lock()
        self.fieldnames: list[str] = []
        self.rows: list[dict[str, str]] = []
        self.reload()

    def reload(self) -> None:
        with self._lock:
            with self.paths.annotation_path.open("r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                self.fieldnames = list(reader.fieldnames or [])
                self.rows = [dict(row) for row in reader]
            missing = [name for name in REQUIRED_COLUMNS if name not in self.fieldnames]
            if missing:
                raise ValueError(f"annotation is missing required columns: {missing}")
            if LABEL_STATUS_COLUMN not in self.fieldnames:
                self.fieldnames.append(LABEL_STATUS_COLUMN)
                for row in self.rows:
                    row[LABEL_STATUS_COLUMN] = LABEL_STATUS_UNREVIEWED

    def state(self, *, status_filter: str, cursor: int) -> dict[str, Any]:
        with self._lock:
            indices = self._filtered_indices(status_filter)
            if indices:
                cursor = max(0, min(cursor, len(indices) - 1))
                row_index = indices[cursor]
                row = dict(self.rows[row_index])
                sample = self._sample_payload(row, row_index)
            else:
                cursor = 0
                sample = None
            return {
                "sample": sample,
                "cursor": cursor,
                "visible_count": len(indices),
                "total_count": len(self.rows),
                "stats": self._stats(),
                "labels": {
                    name: list(spec.labels)
                    for name, spec in HEAD_SPECS_BY_NAME.items()
                },
                "status_filter": status_filter,
            }

    def save_row(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        row_index = int(payload.get("row_index", -1))
        with self._lock:
            if row_index < 0 or row_index >= len(self.rows):
                raise ValueError(f"row_index out of range: {row_index}")
            row = self.rows[row_index]
            for name, spec in HEAD_SPECS_BY_NAME.items():
                value = str(payload.get(name, row.get(name, "")))
                if value not in spec.labels:
                    raise ValueError(f"invalid label for {name}: {value}")
                row[name] = value
            status = str(payload.get(LABEL_STATUS_COLUMN, LABEL_STATUS_REVIEWED))
            if status not in {
                LABEL_STATUS_REVIEWED,
                LABEL_STATUS_SKIPPED,
                LABEL_STATUS_UNREVIEWED,
            }:
                raise ValueError(f"invalid label_status: {status}")
            row[LABEL_STATUS_COLUMN] = status
            self._write_locked()
            return self._sample_payload(row, row_index)

    def image_path_for(self, raw_image_path: str) -> Path:
        path = Path(unquote(raw_image_path))
        if path.is_absolute():
            candidate = path
        else:
            candidate = self.paths.image_root / path
        resolved = candidate.resolve()
        allowed = self.paths.image_root.resolve()
        if not _is_relative_to(resolved, allowed):
            raise ValueError(f"image path is outside image root: {raw_image_path}")
        return resolved

    def _sample_payload(self, row: Mapping[str, str], row_index: int) -> dict[str, Any]:
        image_path = str(row["image_path"])
        return {
            "row_index": row_index,
            "image_path": image_path,
            "image_url": f"/image/{quote(image_path, safe='')}",
            "labels": {name: row.get(name, "") for name in HEAD_SPECS_BY_NAME},
            "label_status": row.get(LABEL_STATUS_COLUMN, LABEL_STATUS_UNREVIEWED),
            "metadata": {
                key: row.get(key, "")
                for key in (
                    "source_video",
                    "source_image",
                    "frame_id",
                    "timestamp_ms",
                    "person_track_id",
                    "bbox_x1",
                    "bbox_y1",
                    "bbox_x2",
                    "bbox_y2",
                    "crop_source",
                )
                if key in row
            },
        }

    def _filtered_indices(self, status_filter: str) -> list[int]:
        if status_filter == "all":
            return list(range(len(self.rows)))
        return [
            index
            for index, row in enumerate(self.rows)
            if row.get(LABEL_STATUS_COLUMN, LABEL_STATUS_UNREVIEWED) == status_filter
        ]

    def _stats(self) -> dict[str, int]:
        stats = {
            "reviewed": 0,
            "unreviewed": 0,
            "skipped": 0,
        }
        for row in self.rows:
            status = row.get(LABEL_STATUS_COLUMN, LABEL_STATUS_UNREVIEWED)
            stats[status] = stats.get(status, 0) + 1
        return stats

    def _write_locked(self) -> None:
        temp_path = self.paths.annotation_path.with_suffix(self.paths.annotation_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({name: row.get(name, "") for name in self.fieldnames})
        temp_path.replace(self.paths.annotation_path)


class AnnotationRequestHandler(BaseHTTPRequestHandler):
    store: CsvAnnotationStore

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_text(INDEX_HTML, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                status_filter = query.get("status", ["unreviewed"])[0]
                cursor = int(query.get("cursor", ["0"])[0])
                self._send_json(self.store.state(status_filter=status_filter, cursor=cursor))
                return
            if parsed.path.startswith("/image/"):
                image_path = self.store.image_path_for(parsed.path[len("/image/") :])
                self._send_file(image_path)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:  # pragma: no cover - HTTP boundary
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if urlparse(self.path).path != "/api/save":
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("payload must be a JSON object")
            self._send_json({"sample": self.store.save_row(payload), "ok": True})
        except Exception as exc:  # pragma: no cover - HTTP boundary
            self._send_json({"error": str(exc), "ok": False}, status=HTTPStatus.BAD_REQUEST)

    def _send_json(self, payload: Mapping[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "image not found")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Dataset directory.")
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION, help="CSV path or path relative to dataset.")
    parser.add_argument("--image-root", type=Path, default=None, help="Image root. Defaults to <dataset>/images.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolve_dataset_paths(args.dataset, args.annotation, args.image_root)
    store = CsvAnnotationStore(paths)
    handler = type(
        "BoundAnnotationRequestHandler",
        (AnnotationRequestHandler,),
        {"store": store},
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving {paths.annotation_path} at {url}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


def resolve_dataset_paths(dataset: Path, annotation: Path, image_root: Path | None) -> DatasetPaths:
    dataset_dir = _resolve_from_root(dataset)
    annotation_path = annotation if annotation.is_absolute() else dataset_dir / annotation
    image_root_path = image_root if image_root is not None else dataset_dir / "images"
    image_root_path = image_root_path if image_root_path.is_absolute() else ROOT / image_root_path
    if not annotation_path.exists():
        raise FileNotFoundError(f"annotation CSV does not exist: {annotation_path}")
    if not image_root_path.exists():
        raise FileNotFoundError(f"image root does not exist: {image_root_path}")
    return DatasetPaths(
        dataset_dir=dataset_dir,
        annotation_path=annotation_path,
        image_root=image_root_path,
    )


def _resolve_from_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Person Attribute Labeler</title>
  <style>
    :root {
      color-scheme: light;
      font-family: "Segoe UI", Arial, sans-serif;
      background: #f5f6f8;
      color: #1f2933;
    }
    * { box-sizing: border-box; }
    body { margin: 0; }
    main {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(320px, 1fr) 420px;
      gap: 0;
    }
    .viewer {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: #252a31;
      min-width: 0;
    }
    .viewer img {
      max-width: 100%;
      max-height: calc(100vh - 48px);
      object-fit: contain;
      background: #111820;
      border: 1px solid #3b424b;
    }
    .panel {
      display: flex;
      flex-direction: column;
      min-height: 100vh;
      padding: 18px;
      gap: 14px;
      background: #ffffff;
      border-left: 1px solid #d8dde5;
    }
    .topline, .nav, .actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .topline { justify-content: space-between; }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    .stats { font-size: 13px; color: #596574; }
    select, button {
      min-height: 36px;
      border: 1px solid #c7ced8;
      border-radius: 6px;
      background: #fff;
      color: #1f2933;
      font-size: 14px;
      padding: 0 10px;
    }
    button {
      cursor: pointer;
      background: #f9fafb;
      font-weight: 600;
    }
    button.primary {
      background: #1d4ed8;
      color: white;
      border-color: #1d4ed8;
    }
    button.warn {
      background: #fff7ed;
      color: #9a3412;
      border-color: #fed7aa;
    }
    .fields {
      display: grid;
      gap: 10px;
    }
    label {
      display: grid;
      gap: 5px;
      font-size: 13px;
      font-weight: 600;
      color: #344054;
    }
    label select { width: 100%; }
    .meta {
      border-top: 1px solid #e4e8ee;
      padding-top: 12px;
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: #596574;
      overflow-wrap: anywhere;
    }
    .status {
      min-height: 20px;
      font-size: 13px;
      color: #166534;
    }
    .empty {
      padding: 18px;
      border: 1px solid #e4e8ee;
      border-radius: 6px;
      color: #596574;
      background: #f9fafb;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      .panel { min-height: auto; border-left: 0; border-top: 1px solid #d8dde5; }
      .viewer { min-height: 55vh; }
      .viewer img { max-height: 50vh; }
    }
  </style>
</head>
<body>
  <main>
    <section class="viewer">
      <img id="image" alt="">
    </section>
    <section class="panel">
      <div class="topline">
        <h1>Person Attribute</h1>
        <span class="stats" id="stats"></span>
      </div>
      <div class="nav">
        <select id="filter">
          <option value="unreviewed">unreviewed</option>
          <option value="reviewed">reviewed</option>
          <option value="skipped">skipped</option>
          <option value="all">all</option>
        </select>
        <button id="prev">Prev</button>
        <button id="next">Next</button>
      </div>
      <div id="empty" class="empty" hidden>No samples in this filter.</div>
      <form id="form" class="fields"></form>
      <div class="actions">
        <button class="primary" id="saveNext">Save + Next</button>
        <button id="saveStay">Save</button>
        <button class="warn" id="skip">Skip</button>
      </div>
      <div class="status" id="message"></div>
      <div class="meta" id="meta"></div>
    </section>
  </main>
  <script>
    const fields = [
      "left_hand_state",
      "left_hand_visibility",
      "right_hand_state",
      "right_hand_visibility",
      "body_orientation",
      "occlusion_level",
    ];
    let state = null;
    let cursor = 0;
    let filter = "unreviewed";

    async function fetchState(nextCursor = cursor) {
      const res = await fetch(`/api/state?status=${encodeURIComponent(filter)}&cursor=${nextCursor}`);
      state = await res.json();
      if (state.error) throw new Error(state.error);
      cursor = state.cursor;
      render();
    }

    function render() {
      const sample = state.sample;
      document.getElementById("stats").textContent =
        `${state.visible_count}/${state.total_count} visible | reviewed ${state.stats.reviewed || 0} | skipped ${state.stats.skipped || 0}`;
      document.getElementById("empty").hidden = !!sample;
      document.getElementById("image").hidden = !sample;
      document.getElementById("form").hidden = !sample;
      if (!sample) return;
      document.getElementById("image").src = sample.image_url;
      const form = document.getElementById("form");
      form.innerHTML = "";
      for (const name of fields) {
        const label = document.createElement("label");
        label.textContent = name;
        const select = document.createElement("select");
        select.id = name;
        for (const optionValue of state.labels[name]) {
          const option = document.createElement("option");
          option.value = optionValue;
          option.textContent = optionValue;
          select.appendChild(option);
        }
        select.value = sample.labels[name];
        label.appendChild(select);
        form.appendChild(label);
      }
      const meta = document.getElementById("meta");
      meta.innerHTML = "";
      const rows = [
        ["row", sample.row_index],
        ["image", sample.image_path],
        ["status", sample.label_status],
        ...Object.entries(sample.metadata),
      ];
      for (const [key, value] of rows) {
        const div = document.createElement("div");
        div.textContent = `${key}: ${value}`;
        meta.appendChild(div);
      }
    }

    async function save(status = "reviewed", advance = true) {
      if (!state?.sample) return;
      const payload = { row_index: state.sample.row_index, label_status: status };
      for (const name of fields) payload[name] = document.getElementById(name).value;
      const res = await fetch("/api/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "save failed");
      document.getElementById("message").textContent = `Saved row ${payload.row_index}`;
      const nextCursor = advance && (filter === "all" || filter === status) ? cursor + 1 : cursor;
      await fetchState(nextCursor);
    }

    function step(delta) {
      if (!state) return;
      const maxCursor = Math.max(0, state.visible_count - 1);
      fetchState(Math.min(maxCursor, Math.max(0, cursor + delta)));
    }

    document.getElementById("filter").addEventListener("change", (event) => {
      filter = event.target.value;
      fetchState(0);
    });
    document.getElementById("prev").addEventListener("click", () => step(-1));
    document.getElementById("next").addEventListener("click", () => step(1));
    document.getElementById("saveNext").addEventListener("click", (event) => {
      event.preventDefault();
      save("reviewed", true);
    });
    document.getElementById("saveStay").addEventListener("click", (event) => {
      event.preventDefault();
      save("reviewed", false);
    });
    document.getElementById("skip").addEventListener("click", (event) => {
      event.preventDefault();
      save("skipped", true);
    });
    document.addEventListener("keydown", (event) => {
      if (event.target.tagName === "SELECT") return;
      if (event.key === "ArrowLeft") step(-1);
      if (event.key === "ArrowRight") step(1);
      if (event.key.toLowerCase() === "s") save("reviewed", true);
      if (event.key.toLowerCase() === "x") save("skipped", true);
    });
    fetchState(0).catch((error) => {
      document.getElementById("message").textContent = error.message;
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
