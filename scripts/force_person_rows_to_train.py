"""Force designated person-attribute rows into the training split (D03 workflow).

After ``split_person_attribute_dataset.py`` has produced a new splits dir, some
rows of the protected person/track (e.g. the D03 person-472 relabels and the
cc015 person-71 window) may land in val/test.  This script moves every row of
``--any-track`` and every row of ``--window-track track:start:end`` into train
(rows + image files), rewrites the three CSVs in place (no ``.csv.csv`` copies),
then prints a per-split verification table.

Example (cc015 v5 round):
  python scripts/force_person_rows_to_train.py \
    --splits-dir datasets/person_attribute_v4_cc015/splits \
    --any-track person-472 \
    --window-track person-71:530:570
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _save_rows(path: Path, rows: list[dict[str, str]]) -> None:
    header = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in header})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("datasets/person_attribute_v4_cc015/splits"))
    parser.add_argument("--any-track", action="append", default=[], help="force every row of this track id")
    parser.add_argument(
        "--window-track",
        action="append",
        default=[],
        help="force rows of this track inside a frame window: track:start:end",
    )
    args = parser.parse_args()

    any_tracks = set(args.any_track)
    windows: list[tuple[str, int, int]] = []
    for spec in args.window_track:
        parts = spec.split(":")
        if len(parts) != 3:
            print(f"invalid --window-track (need track:start:end): {spec}")
            return 1
        windows.append((parts[0], int(parts[1]), int(parts[2])))

    def is_force(row: dict[str, str]) -> bool:
        track = str(row.get("person_track_id", ""))
        if track in any_tracks:
            return True
        try:
            fid = int(row.get("frame_id", "-1"))
        except ValueError:
            return False
        return any(track == t and lo <= fid <= hi for t, lo, hi in windows)

    splits = args.splits_dir
    if not (splits / "train.csv").is_file():
        print(f"train.csv not found under {splits} - run split_person_attribute_dataset.py first")
        return 1

    for sp in ("train", "val", "test"):
        csv_path = splits / f"{sp}.csv"
        if not csv_path.is_file():
            continue
        rows = _load_rows(csv_path)
        move = [r for r in rows if is_force(r)]
        keep = [r for r in rows if not is_force(r)]
        train = _load_rows(splits / "train.csv") if sp != "train" else rows
        have = {str(r["image_path"]).replace("\\", "/") for r in train}
        for r in move:
            ip = str(r["image_path"]).replace("\\", "/")
            key = ip if ip in have else f"train/{ip.rsplit('/', 1)[-1]}"
            if key not in have:
                src_img = splits / "images" / sp / ip.rsplit("/", 1)[-1]
                dst_img = splits / "images" / "train" / ip.rsplit("/", 1)[-1]
                if src_img.is_file() and not dst_img.exists():
                    shutil.move(str(src_img), str(dst_img))
                nr = dict(r)
                nr["image_path"] = f"train/{ip.rsplit('/', 1)[-1]}"
                train.append(nr)
                have.add(nr["image_path"])
        if sp != "train":
            _save_rows(csv_path, keep)
            _save_rows(splits / "train.csv", train)

    # verification
    tracks = sorted(any_tracks) + [t for t, _, _ in windows]
    print(f"verification (track counts per split) for: {tracks}")
    for sp in ("train", "val", "test"):
        rows = _load_rows(splits / f"{sp}.csv")
        counts: list[str] = []
        for t in tracks:
            n = sum(1 for r in rows if str(r.get("person_track_id", "")) == t)
            counts.append(f"{t}={n}")
        print(f"  {sp}: total={len(rows)} " + " ".join(counts))
    print(f"done. splits dir: {splits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
