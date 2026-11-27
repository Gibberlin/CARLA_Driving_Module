from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


BASE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE / "steer_model.pth"
DEFAULT_NORM_PATH = BASE / "steer_normalisation_params.csv"

INPUT_FEATURES = [
    "speed_kmh",
    "forward_speed_mps",
    "heading_deg",
    "yaw_rate_rad_s",
    "forward_acceleration_mps2",
    "lateral_acceleration_mps2",
    "curvature_m_inv",
    "persons_detected",
    "bicycles_detected",
    "cars_detected",
    "trucks_detected",
    "buses_detected",
    "traffic_lights_detected",
    "total_detections",
]


class AVControlModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        previous = len(INPUT_FEATURES)
        for hidden in (64, 32, 16):
            layers.extend((nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(0.05)))
            previous = hidden
        layers.append(nn.Linear(previous, 3))
        self.network = nn.Sequential(*layers)
        self.register_buffer("target_scales", torch.ones(3, dtype=torch.float32))

    def forward(self, sensor_inputs):
        return self.network(sensor_inputs) * self.target_scales


class KrittySteeringController:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, norm_path=DEFAULT_NORM_PATH, device="cpu"):
        model_path = Path(model_path)
        norm_path = Path(norm_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"KITTI steering model not found: {model_path}")
        if not norm_path.is_file():
            raise FileNotFoundError(f"KITTI normalisation parameters not found: {norm_path}")

        norm = pd.read_csv(norm_path).set_index("feature")
        self.mean = norm.loc[INPUT_FEATURES, "mean"].values.astype(np.float32)
        self.std = norm.loc[INPUT_FEATURES, "std"].values.astype(np.float32)
        self.device = torch.device(device)
        self.model = AVControlModel().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def predict_steering(self, sensor_values):
        raw = np.array(
            [float(sensor_values.get(feature, 0.0)) for feature in INPUT_FEATURES],
            dtype=np.float32,
        )
        normalised = (raw - self.mean) / self.std
        tensor = torch.from_numpy(normalised).unsqueeze(0).to(self.device)
        with torch.no_grad():
            steering_deg = float(self.model(tensor)[0, 0].item())
        # KITTI convention: positive steering is left. Convert to CARLA's -1..1
        # steering command with a bounded 30-degree mapping.
        return float(np.clip(-steering_deg / 30.0, -1.0, 1.0)), steering_deg


def detection_counts(boxes, classes):
    counts = {
        "persons_detected": 0,
        "bicycles_detected": 0,
        "cars_detected": 0,
        "trucks_detected": 0,
        "buses_detected": 0,
        "traffic_lights_detected": 0,
    }
    if boxes is None or classes is None:
        return counts
    for class_id in classes:
        class_id = int(class_id)
        if class_id == 0:
            counts["persons_detected"] += 1
        elif class_id == 1:
            counts["bicycles_detected"] += 1
        elif class_id in (2, 3):
            counts["cars_detected"] += 1
        elif class_id == 7:
            counts["trucks_detected"] += 1
        elif class_id == 5:
            counts["buses_detected"] += 1
        elif class_id in (9, 10, 11):
            counts["traffic_lights_detected"] += 1
    return counts


def build_carla_sensor_features(ego, boxes=None, classes=None):
    velocity = ego.get_velocity()
    acceleration = ego.get_acceleration()
    transform = ego.get_transform()
    forward = transform.get_forward_vector()
    right = transform.get_right_vector()
    forward_speed = forward.x * velocity.x + forward.y * velocity.y + forward.z * velocity.z
    forward_accel = forward.x * acceleration.x + forward.y * acceleration.y + forward.z * acceleration.z
    lateral_accel = right.x * acceleration.x + right.y * acceleration.y + right.z * acceleration.z
    speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5

    angular = ego.get_angular_velocity()
    yaw_rate = float(np.deg2rad(angular.z))
    curvature = yaw_rate / forward_speed if abs(forward_speed) > 0.5 else 0.0

    # Relative road heading error in [-15, 15] deg.
    # The training dataset has heading_deg with mean -0.25 deg and std 1.39 deg.
    # Using raw compass yaw [0, 360) causes an extreme 30-130 sigma out-of-distribution
    # input that saturates the model output to -1.0 steer, causing an infinite right-circling loop.
    heading = 0.0
    try:
        world = ego.get_world()
        carla_map = world.get_map()
        wp = carla_map.get_waypoint(transform.location, project_to_road=True)
        if wp is not None:
            road_yaw = float(wp.transform.rotation.yaw)
            ego_yaw = float(transform.rotation.yaw)
            rel_yaw = (ego_yaw - road_yaw + 180.0) % 360.0 - 180.0
            heading = float(np.clip(rel_yaw, -15.0, 15.0))
    except Exception:
        heading = 0.0

    counts = detection_counts(boxes, classes)

    values = {
        "speed_kmh": 3.6 * speed,
        "forward_speed_mps": forward_speed,
        "heading_deg": heading,
        "yaw_rate_rad_s": yaw_rate,
        "forward_acceleration_mps2": forward_accel,
        "lateral_acceleration_mps2": lateral_accel,
        "curvature_m_inv": curvature,
        **counts,
        "total_detections": len(boxes) if boxes is not None else 0,
    }
    return values
