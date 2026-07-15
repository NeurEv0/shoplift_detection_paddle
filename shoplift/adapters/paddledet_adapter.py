"""PaddleDetection environment adapter scaffold.

The concrete PP-Human/MOT/keypoint output conversion belongs to P0 item 5.
This module only isolates path setup so the business package does not need to
live inside the PaddleDetection source tree.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaddleDetectionEnvironment:
    root: Path = Path("src/PaddleDetection-release-2.9")

    def resolve_root(self, project_root: Path | None = None) -> Path:
        root = self.root
        if root.is_absolute():
            return root
        return (project_root or Path.cwd()) / root

    def package_path(self, project_root: Path | None = None) -> Path:
        return self.resolve_root(project_root) / "ppdet"

    def is_available(self, project_root: Path | None = None) -> bool:
        package_path = self.package_path(project_root)
        return package_path.exists() and (package_path / "__init__.py").exists()


def ensure_ppdet_path(root: str | Path = "src/PaddleDetection-release-2.9") -> Path:
    resolved = Path(root).resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    return resolved
