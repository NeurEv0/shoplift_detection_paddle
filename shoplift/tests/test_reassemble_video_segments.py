from pathlib import Path

import pytest

from scripts.reassemble_video_segments import (
    build_default_output_path,
    extract_segment_index,
    validate_contiguous_indices,
)


def test_extract_segment_index_uses_numeric_suffix() -> None:
    assert extract_segment_index(Path("Shoplifting010_x264_12.mp4")) == 12


def test_build_default_output_path_for_directory_named_like_mp4() -> None:
    input_dir = Path("datasets/DCSASS_Shoplifting/Shoplifting/Shoplifting010_x264.mp4")
    assert build_default_output_path(input_dir) == Path(
        "datasets/DCSASS_Shoplifting/Shoplifting/Shoplifting010_x264_merged.mp4"
    )


def test_validate_contiguous_indices_rejects_gaps() -> None:
    with pytest.raises(ValueError):
        validate_contiguous_indices(
            [
                Path("Shoplifting010_x264_0.mp4"),
                Path("Shoplifting010_x264_1.mp4"),
                Path("Shoplifting010_x264_3.mp4"),
            ]
        )
