"""Load rules YAML into the P1 tunable config dataclasses.

The pipeline entry points (offline_analyze / video_infer_visualize) build the
event engine from ``rules.example.yml`` through :func:`load_rules_config`, so
tuning weights/thresholds only requires editing that YAML file — no code
changes. Section names mirror the dataclass fields 1:1:

- ``association``  -> :class:`shoplift.tracking.association.AssociationConfig`
- ``risk_scoring`` -> :class:`shoplift.rules.risk_score.RiskScoringConfig`
- ``rules``        -> :class:`shoplift.rules.validators.RiskRuleConfig`

Unknown keys inside these sections raise ``ValueError`` (a silently ignored
typo would otherwise invalidate a whole tuning pass).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from shoplift.events.nested_concealment import NestedConcealmentConfig
from shoplift.rules.risk_score import RiskScoringConfig
from shoplift.rules.validators import RiskRuleConfig
from shoplift.tracking.association import AssociationConfig

DEFAULT_RULES_CONFIG_PATH = Path(__file__).resolve().parent / "rules.example.yml"

_SECTIONS = {
    "association": AssociationConfig,
    "risk_scoring": RiskScoringConfig,
    "rules": RiskRuleConfig,
    "nested_concealment": NestedConcealmentConfig,
}


@dataclass(frozen=True)
class RulesConfigBundle:
    """All tunable P1 configs loaded from one rules YAML file."""

    association: AssociationConfig
    risk_scoring: RiskScoringConfig
    rules: RiskRuleConfig
    nested_concealment: NestedConcealmentConfig = NestedConcealmentConfig()


def load_rules_config(path: str | Path | None = None) -> RulesConfigBundle:
    """Load and validate the rules YAML file into a :class:`RulesConfigBundle`."""
    config_path = Path(path) if path is not None else DEFAULT_RULES_CONFIG_PATH
    if not config_path.exists():
        raise OSError(f"rules config does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - guarded by env checks
        raise RuntimeError("PyYAML is required to load the rules config") from exc
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"rules config must be a mapping: {config_path}")
    return from_mapping(data)


def from_mapping(mapping: Mapping[str, Any]) -> RulesConfigBundle:
    """Build a bundle from a parsed mapping, merging defaults for missing keys."""
    bundle: dict[str, Any] = {}
    for section, config_cls in _SECTIONS.items():
        kwargs = _subconfig_kwargs(config_cls, _mapping_at(mapping, section))
        bundle[section] = config_cls(**kwargs)
    return RulesConfigBundle(
        association=bundle["association"],
        risk_scoring=bundle["risk_scoring"],
        rules=bundle["rules"],
        nested_concealment=bundle["nested_concealment"],
    )


def _subconfig_kwargs(config_cls: type, section: Mapping[str, Any]) -> dict[str, Any]:
    field_names = {item.name for item in fields(config_cls)}
    kwargs: dict[str, Any] = {}
    for key, value in section.items():
        if key not in field_names:
            allowed = ", ".join(sorted(field_names))
            raise ValueError(
                f"unknown key {key!r} in rules section {config_cls.__name__}; "
                f"allowed keys: {allowed}"
            )
        kwargs[key] = _coerce(value)
    return kwargs


def _coerce(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _mapping_at(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key, {})
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "DEFAULT_RULES_CONFIG_PATH",
    "RulesConfigBundle",
    "from_mapping",
    "load_rules_config",
]
