from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shoplift.backends import PaddleDetPPHumanBackend, PaddleDetPPHumanBackendConfig
from shoplift.cli.offline_analyze import BackendOptions, ModelFreeVisionBackend, create_vision_backend


class PaddleDetPPHumanBackendConfigTest(unittest.TestCase):
    def test_config_from_mapping_resolves_local_paths_without_importing_paddle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ppdet_root = root / "PaddleDetection"
            tracker_config = ppdet_root / "deploy" / "pipeline" / "config" / "tracker_config.yml"
            tracker_config.parent.mkdir(parents=True)
            tracker_config.write_text("type: JDETracker\nJDETracker: {}\n", encoding="utf-8")

            config = PaddleDetPPHumanBackendConfig.from_mapping(
                {
                    "paddledetection_root": str(ppdet_root),
                    "device": "cpu",
                    "mot": {
                        "enabled": True,
                        "model_dir": "models/person_mot",
                        "tracker_config": str(tracker_config),
                    },
                    "keypoint": {"enabled": False},
                    "item_container": {"enabled": False},
                },
                project_root=root,
            )

            self.assertEqual(config.device, "cpu")
            self.assertTrue(config.mot.enabled)
            self.assertEqual(config.mot.model_dir, root / "models/person_mot")
            self.assertEqual(config.mot.tracker_config, tracker_config)

    def test_backend_factory_supports_model_free_and_pphuman(self) -> None:
        self.assertIsInstance(create_vision_backend(BackendOptions()), ModelFreeVisionBackend)
        backend = create_vision_backend(
            BackendOptions(
                backend_type="paddledet_pphuman",
                options={
                    "mot": {"enabled": False},
                    "keypoint": {"enabled": False},
                    "item_container": {"enabled": False},
                },
            )
        )
        self.assertIsInstance(backend, PaddleDetPPHumanBackend)


if __name__ == "__main__":
    unittest.main()
