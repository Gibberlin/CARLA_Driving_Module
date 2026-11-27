"""Multimodal CARLA control model and dataset loader."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


STATE_FEATURES = (
    "speed_kmh",
    "forward_speed_mps",
    "heading_deg",
    "yaw_rate_rad_s",
    "forward_acceleration_mps2",
    "lateral_acceleration_mps2",
    "traffic_light_state",
    "speed_limit_kmh",
)

IMAGE_HEIGHT = 90
IMAGE_WIDTH = 160
LIDAR_POINTS = 2048
RADAR_POINTS = 128


def episode_split_indices(episode_ids, validation_fraction=0.2):
    episode_ids = list(episode_ids)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    unique_episodes = sorted(set(episode_ids))
    if len(unique_episodes) > 1:
        validation_count = max(1, round(len(unique_episodes) * validation_fraction))
        validation_episodes = set(unique_episodes[-validation_count:])
        train_indices = [
            index for index, episode in enumerate(episode_ids)
            if episode not in validation_episodes
        ]
        validation_indices = [
            index for index, episode in enumerate(episode_ids)
            if episode in validation_episodes
        ]
    else:
        split = int(len(episode_ids) * (1.0 - validation_fraction))
        train_indices = list(range(split))
        validation_indices = list(range(split, len(episode_ids)))
    return train_indices, validation_indices


class PointCloudEncoder(nn.Module):
    def __init__(self, output_size=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.ReLU(),
        )

    def forward(self, points):
        encoded = self.layers(points)
        valid = points.abs().sum(dim=-1, keepdim=True) > 0
        encoded = encoded.masked_fill(~valid, -1e4)
        pooled = encoded.max(dim=1).values
        return torch.where(valid.any(dim=1), pooled, torch.zeros_like(pooled))


class FusedControlModel(nn.Module):
    def __init__(self, state_size=len(STATE_FEATURES)):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(5, 24, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 5)),
            nn.Flatten(),
            nn.Linear(64 * 3 * 5, 128),
            nn.ReLU(),
        )
        self.lidar_encoder = PointCloudEncoder()
        self.radar_encoder = PointCloudEncoder()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.control_head = nn.Sequential(
            nn.Linear(128 + 64 + 64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
        )
        self.register_buffer("target_scales", torch.ones(3))

    def forward(self, images, lidar, radar, state):
        features = torch.cat(
            (
                self.image_encoder(images),
                self.lidar_encoder(lidar),
                self.radar_encoder(radar),
                self.state_encoder(state),
            ),
            dim=1,
        )
        return self.control_head(features) * self.target_scales


def _read_point_cloud(path, count, rng):
    if path:
        points = np.load(path).astype(np.float32, copy=False)
    else:
        points = np.empty((0, 4), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"Expected Nx4 point cloud in {path}, got {points.shape}")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > count:
        points = points[rng.choice(len(points), count, replace=False)]
    output = np.zeros((count, 4), dtype=np.float32)
    output[:len(points)] = points
    return output


def _normalise_points(points, distance_scale, velocity_scale=None):
    points = points.copy()
    points[:, :3] /= distance_scale
    if velocity_scale is not None:
        points[:, 3] /= velocity_scale
    return points


class FusedControlDataset(Dataset):
    def __init__(self, manifest, state_mean, state_std, seed=0):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        with self.manifest.open(encoding="utf-8") as file:
            self.rows = [json.loads(line) for line in file if line.strip()]
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        rng = np.random.default_rng(self.seed + index)
        rgb = cv2.imread(str(self.root / row["rgb"]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"Could not read RGB frame: {row['rgb']}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float32) / 255.0

        depth = np.load(self.root / row["depth"]).astype(np.float32, copy=False)
        semantic = np.load(self.root / row["semantic"]).astype(np.float32, copy=False)
        depth = cv2.resize(depth, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        semantic = cv2.resize(
            semantic, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST
        )
        depth = np.clip(depth / 1000.0, 0.0, 1.0)
        semantic = np.clip(semantic / 22.0, 0.0, 1.0)
        image = np.concatenate(
            (rgb, depth[..., None], semantic[..., None]), axis=2
        ).transpose(2, 0, 1)

        lidar = _read_point_cloud(
            self.root / row["lidar"] if row.get("lidar") else None,
            LIDAR_POINTS,
            rng,
        )
        lidar = _normalise_points(lidar, 50.0)
        radar = _read_point_cloud(
            self.root / row["radar"] if row.get("radar") else None,
            RADAR_POINTS,
            rng,
        )
        radar = _normalise_points(radar, 80.0, velocity_scale=30.0)
        state = np.array([row[key] for key in STATE_FEATURES], dtype=np.float32)
        state = (state - self.state_mean) / self.state_std
        target = np.array(
            [row["steer"], row["throttle"], row["brake"]], dtype=np.float32
        )
        return (
            torch.from_numpy(image.copy()),
            torch.from_numpy(lidar),
            torch.from_numpy(radar),
            torch.from_numpy(state),
            torch.from_numpy(target),
        )


class FusedSteeringController:
    def __init__(self, checkpoint, device="cpu"):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Fused steering checkpoint not found: {checkpoint}")
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved.get("state_features") != list(STATE_FEATURES):
            raise ValueError(f"Unsupported state feature schema in {checkpoint}")
        self.device = torch.device(device)
        self.model = FusedControlModel().to(self.device)
        self.model.load_state_dict(saved["model_state"])
        self.model.eval()
        self.state_mean = saved["state_mean"].cpu().numpy()
        self.state_std = saved["state_std"].cpu().numpy()

    @torch.no_grad()
    def predict(self, rgb, depth, semantic, lidar, radar, state):
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float32) / 255.0
        depth = cv2.resize(
            depth.astype(np.float32), (IMAGE_WIDTH, IMAGE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        semantic = cv2.resize(
            semantic.astype(np.float32), (IMAGE_WIDTH, IMAGE_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )
        image = np.concatenate(
            (
                rgb,
                np.clip(depth / 1000.0, 0.0, 1.0)[..., None],
                np.clip(semantic / 22.0, 0.0, 1.0)[..., None],
            ),
            axis=2,
        ).transpose(2, 0, 1).copy()

        lidar = _normalise_points(
            _read_point_cloud_array(lidar, LIDAR_POINTS), 50.0
        )
        radar = _normalise_points(
            _read_point_cloud_array(radar, RADAR_POINTS), 80.0, 30.0
        )
        state_values = np.array(
            [state[name] for name in STATE_FEATURES], dtype=np.float32
        )
        state_values = (state_values - self.state_mean) / self.state_std
        output = self.model(
            torch.from_numpy(image).unsqueeze(0).to(self.device),
            torch.from_numpy(lidar).unsqueeze(0).to(self.device),
            torch.from_numpy(radar).unsqueeze(0).to(self.device),
            torch.from_numpy(state_values).unsqueeze(0).to(self.device),
        )[0].cpu().numpy()
        throttle = float(np.clip(output[1], 0.0, 1.0))
        brake = float(np.clip(output[2], 0.0, 1.0))
        if brake > 0.15:
            throttle = 0.0
        return {
            "steer": float(np.clip(output[0], -1.0, 1.0)),
            "throttle": throttle,
            "brake": brake,
        }


def build_runtime_state(ego):
    velocity = ego.get_velocity()
    acceleration = ego.get_acceleration()
    transform = ego.get_transform()
    forward = transform.get_forward_vector()
    right = transform.get_right_vector()
    forward_speed = (
        forward.x * velocity.x + forward.y * velocity.y + forward.z * velocity.z
    )
    return {
        "speed_kmh": 3.6 * np.sqrt(
            velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
        ),
        "forward_speed_mps": forward_speed,
        "heading_deg": float(transform.rotation.yaw) % 360.0,
        "yaw_rate_rad_s": float(np.deg2rad(ego.get_angular_velocity().z)),
        "forward_acceleration_mps2": (
            forward.x * acceleration.x + forward.y * acceleration.y
            + forward.z * acceleration.z
        ),
        "lateral_acceleration_mps2": (
            right.x * acceleration.x + right.y * acceleration.y
            + right.z * acceleration.z
        ),
        "traffic_light_state": float(ego.get_traffic_light_state()),
        "speed_limit_kmh": float(ego.get_speed_limit()),
    }


def _read_point_cloud_array(points, count):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"Expected Nx4 point cloud, got {points.shape}")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > count:
        indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
        points = points[indices]
    output = np.zeros((count, 4), dtype=np.float32)
    output[:len(points)] = points
    return output
