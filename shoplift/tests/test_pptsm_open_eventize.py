"""Unit tests for PP-TSM open-window eventization."""

from __future__ import annotations

import unittest

from shoplift.pptsm_open.eventize import eventize_open_windows
from shoplift.pptsm_open.types import OpenWindow


def _win(start: int, end: int, prob: float, frame_id: int | None = None) -> OpenWindow:
    return OpenWindow(start_timestamp_ms=start, end_timestamp_ms=end,
                      open_prob=prob, frame_id=frame_id)


class EventizeOpenWindowsTest(unittest.TestCase):
    def test_three_consecutive_fire(self) -> None:
        windows = [_win(0, 100, 0.7, 0), _win(100, 200, 0.8, 1), _win(200, 300, 0.9, 2)]
        events = eventize_open_windows(windows, camera_id="cam", person_track_id="p1")
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.camera_id, "cam")
        self.assertEqual(event.person_track_id, "p1")
        self.assertEqual(event.consecutive_windows, 3)
        self.assertEqual(event.start_timestamp_ms, 0)
        self.assertEqual(event.end_timestamp_ms, 300)
        self.assertEqual(event.peak_open_prob, 0.9)
        self.assertAlmostEqual(event.mean_open_prob, 0.8)

    def test_two_consecutive_no_fire(self) -> None:
        windows = [_win(0, 100, 0.7), _win(100, 200, 0.8)]
        self.assertEqual(eventize_open_windows(windows, camera_id="cam", person_track_id="p1"), [])

    def test_threshold_boundary_counts(self) -> None:
        windows = [_win(0, 100, 0.6), _win(100, 200, 0.6), _win(200, 300, 0.6)]
        self.assertEqual(len(eventize_open_windows(windows, camera_id="cam", person_track_id="p1")), 1)

    def test_below_threshold_ignored(self) -> None:
        windows = [_win(0, 100, 0.59), _win(100, 200, 0.59), _win(200, 300, 0.59)]
        self.assertEqual(eventize_open_windows(windows, camera_id="cam", person_track_id="p1"), [])

    def test_gap_splits_runs(self) -> None:
        windows = [
            _win(0, 100, 0.7), _win(100, 200, 0.8), _win(200, 300, 0.9),   # run of 3 -> event
            _win(300, 400, 0.1),                                            # break
            _win(400, 500, 0.7), _win(500, 600, 0.8),                       # run of 2 -> none
        ]
        events = eventize_open_windows(windows, camera_id="cam", person_track_id="p1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].consecutive_windows, 3)

    def test_unsorted_input_sorted_by_time(self) -> None:
        windows = [_win(200, 300, 0.9), _win(0, 100, 0.7), _win(100, 200, 0.8)]
        events = eventize_open_windows(windows, camera_id="cam", person_track_id="p1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].start_timestamp_ms, 0)
        self.assertEqual(events[0].end_timestamp_ms, 300)

    def test_empty_windows(self) -> None:
        self.assertEqual(eventize_open_windows([], camera_id="cam", person_track_id="p1"), [])

    def test_custom_thresholds(self) -> None:
        windows = [_win(0, 100, 0.5), _win(100, 200, 0.5)]
        events = eventize_open_windows(windows, camera_id="cam", person_track_id="p1",
                                       theta=0.5, min_consecutive_windows=2)
        self.assertEqual(len(events), 1)


class LogitsFromOutputTest(unittest.TestCase):
    """``_logits_from_output`` normalizes PaddleVideo test outputs to logits."""

    def _logits(self, output):

        from shoplift.pptsm_open.infer import _logits_from_output

        return _logits_from_output(output).tolist()

    def test_2d_array(self) -> None:
        import numpy as np

        self.assertEqual(self._logits(np.array([[0.2, 0.8]])), [0.2, 0.8])

    def test_flat_vector(self) -> None:
        import numpy as np

        self.assertEqual(self._logits(np.array([0.3, 0.7])), [0.3, 0.7])

    def test_list_wrapped(self) -> None:
        import numpy as np

        self.assertEqual(self._logits([np.array([[0.4, 0.6]])]), [0.4, 0.6])

    def test_dict_wrapped(self) -> None:
        import numpy as np

        self.assertEqual(self._logits({"output": np.array([[0.1, 0.9]])}), [0.1, 0.9])


if __name__ == "__main__":
    unittest.main()
