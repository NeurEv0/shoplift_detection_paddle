"""Environment self-check for the shoplift PaddleDetection workspace."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "shoplift" / "configs" / "env.example.yml"
MIN_PYTHON = (3, 10)
MAX_PYTHON_EXCLUSIVE = (3, 11)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _status(result: CheckResult) -> str:
    if result.ok:
        return "OK"
    return "FAIL" if result.required else "WARN"


def _import_module(name: str) -> tuple[bool, str, Any | None]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - exact import errors vary.
        return False, str(exc), None
    version = getattr(module, "__version__", "version unknown")
    return True, str(version), module


def _load_config(path: Path) -> tuple[dict[str, Any], list[CheckResult]]:
    if not path.exists():
        return {}, [CheckResult("config file", False, f"missing: {path}")]

    results = [CheckResult("config file", True, str(path))]
    ok, detail, yaml = _import_module("yaml")
    if not ok:
        return {}, [*results, CheckResult("PyYAML import", False, detail)]

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {}, [*results, CheckResult("config parse", False, str(exc))]

    return data, [*results, CheckResult("PyYAML import", True, detail)]


def _check_python() -> CheckResult:
    version = sys.version_info[:3]
    ok = MIN_PYTHON <= version < MAX_PYTHON_EXCLUSIVE
    wanted = (
        f">={'.'.join(map(str, MIN_PYTHON))},"
        f"<{'.'.join(map(str, MAX_PYTHON_EXCLUSIVE))}"
    )
    detail = f"{sys.version.split()[0]} expected {wanted}"
    return CheckResult("Python version", ok, detail)


def _check_import(name: str, label: str | None = None, required: bool = True) -> CheckResult:
    ok, detail, _ = _import_module(name)
    return CheckResult(label or f"{name} import", ok, detail, required=required)


def _check_paddle_gpu(paddle: Any | None) -> CheckResult:
    if paddle is None:
        return CheckResult("Paddle GPU", False, "paddle is not importable", required=False)

    try:
        compiled = bool(paddle.device.is_compiled_with_cuda())
        count = int(paddle.device.cuda.device_count()) if compiled else 0
    except Exception as exc:  # pragma: no cover - depends on hardware/runtime.
        return CheckResult("Paddle GPU", False, str(exc), required=False)

    if compiled and count > 0:
        return CheckResult("Paddle GPU", True, f"{count} CUDA device(s)", required=False)
    return CheckResult("Paddle GPU", False, "CUDA not visible; CPU mode is usable", required=False)


def _configured_path(config: dict[str, Any], *keys: str, default: str) -> Path:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return ROOT / default
        value = value[key]
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _check_path(name: str, path: Path, required: bool = True) -> CheckResult:
    return CheckResult(name, path.exists(), str(path), required=required)


def _check_ppdet(config: dict[str, Any]) -> list[CheckResult]:
    ppdet_root = _configured_path(
        config,
        "paths",
        "paddledetection_root",
        default="src/PaddleDetection-release-2.9",
    )
    results = [_check_path("PaddleDetection root", ppdet_root)]
    if ppdet_root.exists():
        sys.path.insert(0, str(ppdet_root))
    ok, detail, _ = _import_module("ppdet")
    results.append(CheckResult("ppdet import", ok, detail))
    return results


def run_checks(config_path: Path) -> list[CheckResult]:
    config, config_results = _load_config(config_path)
    ok, paddle_detail, paddle = _import_module("paddle")

    results = [
        _check_python(),
        *config_results,
        CheckResult("paddle import", ok, paddle_detail),
        _check_paddle_gpu(paddle),
        _check_import("cv2", "OpenCV import"),
        *_check_ppdet(config),
        _check_path("shoplift package", ROOT / "shoplift"),
        _check_path("pipeline config example", ROOT / "shoplift" / "configs" / "pipeline.example.yml"),
        _check_path("rules config example", ROOT / "shoplift" / "configs" / "rules.example.yml"),
        _check_path("risk event schema", ROOT / "shoplift" / "events" / "risk_event.schema.json"),
    ]
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to env.yml or env.example.yml.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat optional warnings, such as missing GPU, as failures.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable check results.",
    )
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    results = run_checks(config_path)

    if args.json:
        print(json.dumps([result.__dict__ for result in results], indent=2))
    else:
        for result in results:
            print(f"[{_status(result)}] {result.name}: {result.detail}")

    failures = [result for result in results if not result.ok and (result.required or args.strict)]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
