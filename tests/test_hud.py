"""Unit tests for Navigation HUD rendering."""

import unittest

from navigation.arrival_manager import NavigationState
from visualization.hud import draw_navigation_hud, get_state_color

try:
    import numpy as np
except ImportError:
    np = None


class TestNavigationHUD(unittest.TestCase):
    def test_state_colors(self):
        c_nav = get_state_color(NavigationState.NAVIGATING)
        c_app = get_state_color(NavigationState.APPROACHING)
        c_stop = get_state_color(NavigationState.STOPPING)
        c_arr = get_state_color(NavigationState.ARRIVED)

        self.assertEqual(len(c_nav), 3)
        self.assertEqual(len(c_app), 3)
        self.assertEqual(len(c_stop), 3)
        self.assertEqual(len(c_arr), 3)
        # Colors should be distinct
        self.assertNotEqual(c_nav, c_arr)

    def test_draw_hud_on_frame(self):
        if np is None:
            self.skipTest("numpy not installed in test environment")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rendered = draw_navigation_hud(
            frame=frame,
            state=NavigationState.APPROACHING,
            current_speed_kmh=18.5,
            target_speed_kmh=15.0,
            dist_to_destination_m=22.4,
            total_distance_m=180.0,
            progress_pct=87.5,
            is_arrived=False,
        )
        self.assertEqual(rendered.shape, (480, 640, 3))
        # Pixels in the HUD region should not be all zero
        self.assertGreater(rendered[60:120, 20:200].sum(), 0)

    def test_draw_hud_arrived_state(self):
        if np is None:
            self.skipTest("numpy not installed in test environment")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rendered = draw_navigation_hud(
            frame=frame,
            state=NavigationState.ARRIVED,
            current_speed_kmh=0.0,
            target_speed_kmh=0.0,
            dist_to_destination_m=0.8,
            total_distance_m=180.0,
            progress_pct=100.0,
            is_arrived=True,
        )
        self.assertEqual(rendered.shape, (480, 640, 3))
        self.assertGreater(rendered.sum(), 0)


if __name__ == "__main__":
    unittest.main()

