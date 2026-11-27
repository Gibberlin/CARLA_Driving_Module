"""Synchronous depth, semantic, LiDAR, and radar sensors for CARLA demos."""

import math
import queue

import numpy as np


def decode_depth(raw, height, width):
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    return (
        (bgra[..., 2].astype(np.float32)
         + bgra[..., 1].astype(np.float32) * 256.0
         + bgra[..., 0].astype(np.float32) * 65536.0)
        / 16777215.0 * 1000.0
    )


def decode_semantic(raw, height, width):
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)[..., 2].copy()


def decode_radar(measurement):
    points = []
    for detection in measurement:
        cos_altitude = math.cos(detection.altitude)
        points.append((
            detection.depth * cos_altitude * math.cos(detection.azimuth),
            detection.depth * cos_altitude * math.sin(detection.azimuth),
            detection.depth * math.sin(detection.altitude),
            detection.velocity,
        ))
    return np.asarray(points, dtype=np.float32).reshape(-1, 4)


class FusionSensorRig:
    def __init__(self, world, ego, carla, fixed_delta, width, height, fov=90.0):
        self.actors = []
        self.queues = {}
        blueprints = world.get_blueprint_library()
        transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        camera_attrs = {
            "image_size_x": width,
            "image_size_y": height,
            "fov": fov,
            "sensor_tick": fixed_delta,
        }
        specifications = (
            ("depth", "sensor.camera.depth", transform, camera_attrs),
            ("semantic", "sensor.camera.semantic_segmentation", transform, camera_attrs),
            ("lidar", "sensor.lidar.ray_cast",
             carla.Transform(carla.Location(z=2.2)), {
                 "channels": 32,
                 "points_per_second": 56000,
                 "rotation_frequency": 1.0 / fixed_delta,
                 "range": 50,
                 "sensor_tick": fixed_delta,
             }),
            ("radar", "sensor.other.radar", transform, {
                "horizontal_fov": 35,
                "vertical_fov": 20,
                "range": 80,
                "sensor_tick": fixed_delta,
            }),
        )
        try:
            for name, blueprint_id, sensor_transform, attributes in specifications:
                blueprint = blueprints.find(blueprint_id)
                for key, value in attributes.items():
                    blueprint.set_attribute(key, str(value))
                sensor = world.spawn_actor(
                    blueprint, sensor_transform, attach_to=ego
                )
                self.actors.append(sensor)
                self.queues[name] = queue.Queue()
                sensor.listen(self.queues[name].put)
        except Exception:
            self.stop()
            raise

    def read(self, frame_number, timeout=10.0):
        readings = {}
        for name, sensor_queue in self.queues.items():
            while True:
                try:
                    measurement = sensor_queue.get(timeout=timeout)
                except queue.Empty as exc:
                    raise TimeoutError(
                        f"{name} sensor timed out waiting for frame {frame_number}"
                    ) from exc
                if measurement.frame == frame_number:
                    readings[name] = measurement
                    break
                if measurement.frame > frame_number:
                    raise RuntimeError(
                        f"{name} sensor frame mismatch: expected {frame_number}, "
                        f"got {measurement.frame}"
                    )

        depth_measurement = readings["depth"]
        semantic_measurement = readings["semantic"]
        lidar_measurement = readings["lidar"]
        return {
            "depth": decode_depth(
                depth_measurement.raw_data,
                depth_measurement.height,
                depth_measurement.width,
            ),
            "semantic": decode_semantic(
                semantic_measurement.raw_data,
                semantic_measurement.height,
                semantic_measurement.width,
            ),
            "lidar": np.frombuffer(
                lidar_measurement.raw_data, dtype=np.float32
            ).reshape(-1, 4).copy(),
            "radar": decode_radar(readings["radar"]),
        }

    def stop(self):
        for actor in self.actors:
            if actor.is_alive:
                actor.stop()
        alive = [actor for actor in self.actors if actor.is_alive]
        for actor in alive:
            actor.destroy()
        self.actors.clear()
