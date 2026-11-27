import unittest
from types import SimpleNamespace

from live_demo import (
    nearest_obstacle_distance,
    should_apply_emergency_brake,
    traffic_light_status,
)


class FakeActor:
    def __init__(self, actor_id, x, y=0.0, z=0.0, yaw=0.0, extent=(2.0, 1.0, 1.0)):
        self.id = actor_id
        self._transform = SimpleNamespace(
            location=SimpleNamespace(x=x, y=y, z=z),
            rotation=SimpleNamespace(yaw=yaw),
        )
        self.bounding_box = SimpleNamespace(
            extent=SimpleNamespace(x=extent[0], y=extent[1], z=extent[2])
        )
        self.traffic_light = None

    def get_transform(self):
        return self._transform

    def get_traffic_light(self):
        return self.traffic_light


class TestLiveDemoSafety(unittest.TestCase):
    def test_nearest_distance_uses_bounding_box_clearance_and_skips_ego(self):
        ego = FakeActor(1, 0.0)
        close_actor = FakeActor(2, 8.5)
        distant_actor = FakeActor(3, 20.0)

        distance = nearest_obstacle_distance(
            ego, [ego, distant_actor, close_actor]
        )

        self.assertAlmostEqual(distance, 4.5)

    def test_safety_zone_ignores_rear_and_back_actors_but_protects_front_and_front_sides(self):
        ego = FakeActor(1, 0.0)  # extent = (2.0, 1.0, 1.0)
        
        # 1. Back/rear actor directly behind vehicle (x = -6.0)
        rear_actor = FakeActor(2, -6.0)
        self.assertIsNone(nearest_obstacle_distance(ego, [ego, rear_actor]))
        
        # 2. Rear-side actor beside rear wheels (x = -1.5, y = 2.0)
        rear_side_actor = FakeActor(3, -1.5, y=2.0)
        self.assertIsNone(nearest_obstacle_distance(ego, [ego, rear_side_actor]))
        
        # 3. Front-side actor beside front fender (x = 1.0, y = 2.5, extent = (1.0, 1.0, 1.0))
        # lateral_gap = 2.5 - 1.0 (ego) - 1.0 (actor) = 0.5m
        front_side_actor = FakeActor(4, 1.0, y=2.5, extent=(1.0, 1.0, 1.0))
        d_front_side = nearest_obstacle_distance(ego, [ego, front_side_actor])
        self.assertIsNotNone(d_front_side)
        self.assertAlmostEqual(d_front_side, 0.5)

        # 4. Front actor ahead of front bumper (x = 7.0, extent = (2.0, 1.0, 1.0))
        # forward_gap = 7.0 - 2.0 (ego) - 2.0 (actor) = 3.0m
        front_actor = FakeActor(5, 7.0)
        d_front = nearest_obstacle_distance(ego, [ego, front_actor])
        self.assertIsNotNone(d_front)
        self.assertAlmostEqual(d_front, 3.0)

    def test_emergency_brake_activates_at_five_metres_and_releases_with_hysteresis(self):
        self.assertTrue(should_apply_emergency_brake(False, 5.0))
        self.assertFalse(should_apply_emergency_brake(False, 5.01))
        self.assertTrue(should_apply_emergency_brake(True, 5.8))
        self.assertFalse(should_apply_emergency_brake(True, 6.01))
        self.assertFalse(should_apply_emergency_brake(True, None))

    def test_traffic_light_status_is_normalized_for_hud(self):
        vehicle = FakeActor(1, 0.0)
        vehicle.traffic_light = SimpleNamespace(
            get_state=lambda: SimpleNamespace(name="Red")
        )

        self.assertEqual(traffic_light_status(vehicle), "RED")
        vehicle.traffic_light = None
        self.assertEqual(traffic_light_status(vehicle), "NONE")


if __name__ == "__main__":
    unittest.main()
