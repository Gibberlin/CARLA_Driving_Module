"""GPS and Vehicle Location Tracker.

Tracks vehicle world position, GPS coordinates, distance to destination,
odometry (distance travelled), heading, speed, navigation state, and route progress.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


class GPSTracker:
    """Provides unified location, GPS, odometry, and navigation progress tracking."""

    def __init__(
        self,
        carla_map: Optional[Any] = None,
        destination_location: Optional[Any] = None,
        planned_route_distance: float = 0.0,
    ):
        """Initializes the GPSTracker.

        Args:
            carla_map: Optional CARLA map instance (for transform_to_geolocation)
            destination_location: Target destination (tuple (x, y) or Location object)
            planned_route_distance: Total planned route distance in meters
        """
        self.carla_map = carla_map
        self.planned_route_distance = max(0.0, float(planned_route_distance))
        self.dest_x, self.dest_y = self._extract_coords(destination_location) if destination_location else (None, None)

        # State attributes
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0
        self.latitude: float = 0.0
        self.longitude: float = 0.0
        self.altitude: float = 0.0
        self.speed_kmh: float = 0.0
        self.speed_ms: float = 0.0
        self.heading_yaw: float = 0.0
        self.distance_to_destination: float = 0.0
        self.distance_travelled: float = 0.0
        self.route_progress_pct: float = 0.0
        self.navigation_state: str = "NAVIGATING"
        self.previous_location: Optional[Tuple[float, float, float]] = None

    @staticmethod
    def _extract_coords(location: Any) -> Tuple[float, float]:
        if hasattr(location, "x") and hasattr(location, "y"):
            return float(location.x), float(location.y)
        if isinstance(location, (tuple, list)) and len(location) >= 2:
            return float(location[0]), float(location[1])
        raise ValueError(f"Invalid location object: {location}")

    def set_destination(self, destination_location: Any) -> None:
        """Updates the destination coordinate."""
        self.dest_x, self.dest_y = self._extract_coords(destination_location)

    def set_planned_distance(self, distance_m: float) -> None:
        """Sets the planned total route distance."""
        self.planned_route_distance = max(0.0, float(distance_m))

    def update(
        self,
        vehicle_transform: Any,
        speed_kmh: float,
        navigation_state: str = "NAVIGATING",
        remaining_waypoints_count: Optional[int] = None,
        total_waypoints_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Updates the vehicle position and computed tracking parameters.

        Args:
            vehicle_transform: CARLA Transform object or object with .location and .rotation
            speed_kmh: Current vehicle speed in km/h
            navigation_state: String or NavigationState representing current FSM state
            remaining_waypoints_count: Optional remaining waypoint count for progress calculation
            total_waypoints_count: Optional total waypoint count

        Returns:
            Dictionary with current tracking snapshot.
        """
        loc = getattr(vehicle_transform, "location", None)
        rot = getattr(vehicle_transform, "rotation", None)

        if loc is not None:
            self.x = float(loc.x)
            self.y = float(loc.y)
            self.z = float(loc.z)
        elif isinstance(vehicle_transform, (tuple, list)):
            self.x = float(vehicle_transform[0])
            self.y = float(vehicle_transform[1])
            self.z = float(vehicle_transform[2]) if len(vehicle_transform) > 2 else 0.0

        if rot is not None:
            self.heading_yaw = float(rot.yaw)
        elif hasattr(vehicle_transform, "yaw"):
            self.heading_yaw = float(vehicle_transform.yaw)

        self.speed_kmh = max(0.0, float(speed_kmh))
        self.speed_ms = self.speed_kmh / 3.6
        self.navigation_state = str(getattr(navigation_state, "value", navigation_state))

        # 1. Update distance travelled (odometry)
        if self.previous_location is not None:
            prev_x, prev_y, prev_z = self.previous_location
            step = math.sqrt((self.x - prev_x) ** 2 + (self.y - prev_y) ** 2 + (self.z - prev_z) ** 2)
            if step > 0.001:
                self.distance_travelled += step
        self.previous_location = (self.x, self.y, self.z)

        # 2. Geolocation (GPS)
        if self.carla_map is not None and loc is not None:
            try:
                geo = self.carla_map.transform_to_geolocation(loc)
                self.latitude = float(geo.latitude)
                self.longitude = float(geo.longitude)
                self.altitude = float(geo.altitude)
            except Exception:
                self._approximate_gps()
        else:
            self._approximate_gps()

        # 3. Distance to Destination
        if self.dest_x is not None and self.dest_y is not None:
            self.distance_to_destination = math.hypot(self.x - self.dest_x, self.y - self.dest_y)
        else:
            self.distance_to_destination = 0.0

        # 4. Route Progress Percentage
        if remaining_waypoints_count is not None and total_waypoints_count and total_waypoints_count > 0:
            completed_ratio = max(0.0, min(1.0, 1.0 - (remaining_waypoints_count / total_waypoints_count)))
            self.route_progress_pct = completed_ratio * 100.0
        elif self.planned_route_distance > 0.0:
            if self.navigation_state == "ARRIVED":
                self.route_progress_pct = 100.0
            else:
                progress = max(0.0, self.planned_route_distance - self.distance_to_destination)
                self.route_progress_pct = min(100.0, (progress / self.planned_route_distance) * 100.0)
        else:
            self.route_progress_pct = 100.0 if self.navigation_state == "ARRIVED" else 0.0

        return self.get_telemetry_dict()

    def _approximate_gps(self) -> None:
        """Approximates GPS coordinates for Town03 using standard WGS84 projection."""
        lat = self.y / 111319.49
        lon = self.x / (111319.49 * max(0.1, math.cos(math.radians(lat))))
        self.latitude = lat
        self.longitude = lon
        self.altitude = self.z

    def get_telemetry_dict(self) -> Dict[str, Any]:
        """Returns structured dictionary of current vehicle location and navigation state."""
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "z": round(self.z, 3),
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "altitude": round(self.altitude, 2),
            "speed_kmh": round(self.speed_kmh, 2),
            "speed_ms": round(self.speed_ms, 2),
            "heading_yaw": round(self.heading_yaw, 2),
            "distance_to_destination": round(self.distance_to_destination, 2),
            "distance_travelled": round(self.distance_travelled, 2),
            "route_progress_pct": round(self.route_progress_pct, 1),
            "navigation_state": self.navigation_state,
        }

