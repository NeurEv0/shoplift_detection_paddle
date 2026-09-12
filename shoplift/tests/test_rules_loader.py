from __future__ import annotations

import unittest

from shoplift.configs.rules_loader import (
    DEFAULT_RULES_CONFIG_PATH,
    RulesConfigBundle,
    from_mapping,
    load_rules_config,
)
from shoplift.rules.risk_score import RiskScoringConfig
from shoplift.rules.validators import RiskRuleConfig
from shoplift.tracking.association import AssociationConfig


class RulesLoaderTest(unittest.TestCase):
    def test_load_default_config_matches_code_defaults(self) -> None:
        bundle = load_rules_config()
        self.assertIsInstance(bundle, RulesConfigBundle)
        self.assertIsInstance(bundle.association, AssociationConfig)
        self.assertIsInstance(bundle.risk_scoring, RiskScoringConfig)
        self.assertIsInstance(bundle.rules, RiskRuleConfig)
        self.assertEqual(bundle.association.min_contact_frames, 3)
        self.assertEqual(bundle.association.normal_container_categories, ("basket", "cart", "checkout_bag"))
        self.assertEqual(bundle.risk_scoring.action_type_weights["bag_concealment"], 0.32)
        self.assertEqual(bundle.risk_scoring.container_type_weights["normal"], -0.20)
        self.assertEqual(bundle.risk_scoring.medium_threshold, 0.45)
        self.assertEqual(bundle.risk_scoring.high_threshold, 0.75)
        self.assertEqual(bundle.rules.high_risk_min_reason_tags, 2)
        self.assertEqual(bundle.rules.single_frame_contact_score_cap, 0.44)

    def test_default_path_points_at_rules_example_yml(self) -> None:
        self.assertTrue(DEFAULT_RULES_CONFIG_PATH.name == "rules.example.yml")
        self.assertTrue(DEFAULT_RULES_CONFIG_PATH.exists())

    def test_from_mapping_overrides_and_preserves_defaults(self) -> None:
        bundle = from_mapping(
            {
                "association": {"min_entry_frames": 5},
                "risk_scoring": {
                    "high_threshold": 0.8,
                    "action_type_weights": {"bag_concealment": 0.5},
                },
                "rules": {"high_risk_min_reason_tags": 3},
            }
        )
        self.assertEqual(bundle.association.min_entry_frames, 5)
        self.assertEqual(bundle.association.min_contact_frames, 3)  # 未覆盖的字段保留默认
        self.assertEqual(bundle.risk_scoring.high_threshold, 0.8)
        self.assertEqual(bundle.risk_scoring.bulk_item_count_threshold, 3)
        self.assertEqual(bundle.risk_scoring.action_type_weights["bag_concealment"], 0.5)
        self.assertEqual(bundle.rules.high_risk_min_reason_tags, 3)
        self.assertEqual(bundle.rules.medium_threshold, 0.45)

    def test_unknown_key_in_section_raises(self) -> None:
        with self.assertRaises(ValueError):
            from_mapping({"association": {"min_contact_frames": 3, "typo_key": 1}})
        with self.assertRaises(ValueError):
            from_mapping({"risk_scoring": {"not_a_field": 0.5}})
        with self.assertRaises(ValueError):
            from_mapping({"rules": {"high_risk_min_reason_tags": 2, "wrong": 1}})

    def test_unknown_top_level_section_is_ignored(self) -> None:
        bundle = from_mapping({"event_types": {"supported": ["bag_concealment"]}})
        self.assertEqual(bundle.association.min_contact_frames, 3)

    def test_loader_rejects_missing_file(self) -> None:
        with self.assertRaises(OSError):
            load_rules_config("no_such_rules_file.yml")


if __name__ == "__main__":
    unittest.main()
