"""Destination Arrival Manager and Finite State Machine.

Manages destination approach, speed tapering, controlled stopping, and arrival
confirmation for autonomous vehicle navigation.
"""

from __future__ import annotations

import enum
import math
from typing import Any, Dict, List, Optional, Tuple


class NavigationState(str, enum.Enum):
    """Navigation states for route following and destination arrival."""
    NAVIGATING = "NAVIGATING"
    APPROACHING = "APPROACHING"
    STOPPING = "STOPPING"
    ARRIVED = "ARRIVED"


class ArrivalManager:
    """Finite State Machine for autonomous destination approach and stopping.

    State transitions:
        NAVIGATING -> APPROACHING (when distance <= approach_distance)
        APPROACHING -> STOPPING   (when distance <= stopping_distance)
        STOPPING -> ARRIVED       (when distance <= arrival_radius AND speed <= stop_speed_threshold for consecutive frames)
        ARRIVED -> locked in ARRIVED (full stop, no acceleration permitted)
    """

    def __init__(
        self,
        destination_location: Any,
        cruise_speed_kmh: float = 25.0,
        approach_distance: float = 20.0,
        stopping_distance: float = 2.5,
        arrival_radius: float = 2.0,
        deceleration: float = 2.0,
        stop_speed_threshold: float = 0.5,
        arrival_confirm_frames: int = 10,
        min_approach_speed_kmh: float = 4.0,
    ):
        """Initializes the ArrivalManager with validation.

        Args:
            destination_location: Tuple (x, y) or Location object with .x, .y
            cruise_speed_kmh: Normal target cruising speed in km/h (> 0).
            approach_distance: Distance in meters to start decelerating (> 0).
            stopping_distance: Distance in meters to enter stopping zone (> 0).
            arrival_radius: Target radius in meters around destination to confirm arrival (> 0).
            deceleration: Target deceleration profile in m/s^2 (> 0).
            stop_speed_threshold: Maximum speed in km/h to consider vehicle stopped (>= 0).
            arrival_confirm_frames: Number of consecutive stopped frames to confirm arrival (>= 1).
            min_approach_speed_kmh: Minimum crawling speed while approaching (> 0).
        """
        if cruise_speed_kmh <= 0:
            raise ValueError(f"cruise_speed_kmh must be positive, got {cruise_speed_kmh}")
        if approach_distance <= 0:
            raise ValueError(f"approach_distance must be positive, got {approach_distance}")
        if stopping_distance <= 0:
            raise ValueError(f"stopping_distance must be positive, got {stopping_distance}")
        if arrival_radius <= 0:
            raise ValueError(f"arrival_radius must be positive, got {arrival_radius}")
        if deceleration <= 0:
            raise ValueError(f"deceleration must be positive, got {deceleration}")
        if stop_speed_threshold < 0:
            raise ValueError(f"stop_speed_threshold cannot be negative, got {stop_speed_threshold}")
        if arrival_confirm_frames < 1:
            raise ValueError(f"arrival_confirm_frames must be >= 1, got {arrival_confirm_frames}")
        if approach_distance < stopping_distance:
            raise ValueError(
                f"approach_distance ({approach_distance}) must be >= stopping_distance ({stopping_distance})"
            )

        self.destination_x, self.destination_y = self._extract_coords(destination_location)
        self.cruise_speed_kmh = float(cruise_speed_kmh)
        self.approach_distance = float(approach_distance)
        self.stopping_distance = float(stopping_distance)
        self.arrival_radius = float(arrival_radius)
        self.deceleration = float(deceleration)
        self.stop_speed_threshold = float(stop_speed_threshold)
        self.arrival_confirm_frames = int(arrival_confirm_frames)
        self.min_approach_speed_kmh = float(min_approach_speed_kmh)

        # FSM state
        self.state = NavigationState.NAVIGATING
        self.target_speed_kmh = self.cruise_speed_kmh
        self.current_distance: float = float("inf")
        self.current_speed_kmh: float = 0.0
        self.consecutive_stopped_frames: int = 0
        self.state_history: List[Tuple[float, NavigationState, str]] = []

    @staticmethod
    def _extract_coords(location: Any) -> Tuple[float, float]:
        """Extracts (x, y) coordinates from tuple, list, or CARLA Location object."""
        if hasattr(location, "x") and hasattr(location, "y"):
            return float(location.x), float(location.y)
        if isinstance(location, (tuple, list)) and len(location) >= 2:
            return float(location[0]), float(location[1])
        raise ValueError(f"Invalid destination location: {location}")

    def calculate_distance(self, location: Any) -> float:
        """Calculates 2D horizontal Euclidean distance to destination."""
        loc_x, loc_y = self._extract_coords(location)
        return math.hypot(loc_x - self.destination_x, loc_y - self.destination_y)

    def calculate_deceleration_target_speed(self, distance_m: float) -> float:
        """Calculates smooth target speed based on deceleration profile: v = min(v_cruise, sqrt(2 * a * d))."""
        if distance_m <= 0.0:
            return 0.0

        v_cruise_ms = self.cruise_speed_kmh / 3.6
        v_profile_ms = math.sqrt(2.0 * self.deceleration * distance_m)
        v_target_ms = min(v_cruise_ms, v_profile_ms)
        v_target_kmh = v_target_ms * 3.6

        # Avoid vehicle stalling prematurely before reaching the actual stopping zone
        if distance_m > self.stopping_distance:
            v_target_kmh = max(self.min_approach_speed_kmh, v_target_kmh)

        return float(min(self.cruise_speed_kmh, max(0.0, v_target_kmh)))

    def update(
        self,
        current_location: Any,
        current_speed_kmh: float,
        sim_time_s: float = 0.0,
    ) -> NavigationState:
        """Updates the Arrival FSM state and calculates target speed.

        Args:
            current_location: Current vehicle location (Location object or (x, y))
            current_speed_kmh: Current vehicle speed in km/h
            sim_time_s: Current simulation timestamp in seconds

        Returns:
            Current NavigationState
        """
        self.current_distance = self.calculate_distance(current_location)
        self.current_speed_kmh = max(0.0, float(current_speed_kmh))
        previous_state = self.state

        if self.state == NavigationState.NAVIGATING:
            self.target_speed_kmh = self.cruise_speed_kmh
            if self.current_distance <= self.approach_distance:
                self.state = NavigationState.APPROACHING
                self._log_transition(
                    sim_time_s,
                    previous_state,
                    self.state,
                    f"Remaining distance {self.current_distance:.1f} m <= approach distance {self.approach_distance:.1f} m",
                )
                print(
                    f"[NAVIGATION] Approaching destination "
                    f"({self.current_distance:.1f} m remaining, reducing target speed)"
                )

        elif self.state == NavigationState.APPROACHING:
            self.target_speed_kmh = self.calculate_deceleration_target_speed(self.current_distance)
            if self.current_distance <= self.stopping_distance:
                self.state = NavigationState.STOPPING
                self.target_speed_kmh = 0.0
                self._log_transition(
                    sim_time_s,
                    previous_state,
                    self.state,
                    f"Reached stopping zone ({self.current_distance:.1f} m <= {self.stopping_distance:.1f} m)",
                )
                print(
                    f"[NAVIGATION] Stopping vehicle at destination zone "
                    f"({self.current_distance:.1f} m, speed {self.current_speed_kmh:.1f} km/h)"
                )

        elif self.state == NavigationState.STOPPING:
            self.target_speed_kmh = 0.0
            # Check arrival confirmation criteria:
            # 1. Distance within arrival radius
            # 2. Speed below stopped threshold
            if (
                self.current_distance <= self.arrival_radius
                and self.current_speed_kmh <= self.stop_speed_threshold
            ):
                self.consecutive_stopped_frames += 1
                if self.consecutive_stopped_frames >= self.arrival_confirm_frames:
                    self.state = NavigationState.ARRIVED
                    self._log_transition(
                        sim_time_s,
                        previous_state,
                        self.state,
                        f"Stationary for {self.consecutive_stopped_frames} frames within {self.arrival_radius} m",
                    )
                    print(
                        f"[NAVIGATION] Destination reached! Vehicle is stationary at destination "
                        f"(offset: {self.current_distance:.2f} m, speed: {self.current_speed_kmh:.2f} km/h)"
                    )
            else:
                # If speed rises above threshold (e.g. still rolling), reset consecutive count
                if self.current_speed_kmh > self.stop_speed_threshold:
                    self.consecutive_stopped_frames = 0

        elif self.state == NavigationState.ARRIVED:
            # Locked state: target speed is strictly zero and counter increments
            self.target_speed_kmh = 0.0
            self.consecutive_stopped_frames += 1

        return self.state

    def get_control_override(self, agent_control: Any = None) -> Optional[Any]:
        """Returns control overrides to guarantee vehicle stopping and holding.

        Returns:
            Dict or vehicle control kwargs when an override is active, or None.
        """
        if self.state == NavigationState.ARRIVED:
            # Absolute lockout: full brake, zero throttle, neutral steering
            return {
                "throttle": 0.0,
                "steer": 0.0,
                "brake": 1.0,
                "hand_brake": True,
            }
        elif self.state == NavigationState.STOPPING:
            # Active deceleration override: zero throttle, controlled heavy braking
            # Apply progressive brake proportional to current speed
            brake_intensity = min(1.0, max(0.8, 0.5 + 0.5 * (self.current_speed_kmh / 10.0)))
            steer = getattr(agent_control, "steer", 0.0) if agent_control else 0.0
            return {
                "throttle": 0.0,
                "steer": float(steer),
                "brake": float(brake_intensity),
                "hand_brake": self.current_speed_kmh <= self.stop_speed_threshold,
            }
        return None

    def _log_transition(
        self, sim_time_s: float, from_state: NavigationState, to_state: NavigationState, reason: str
    ) -> None:
        self.state_history.append((sim_time_s, to_state, reason))

    @property
    def is_arrived(self) -> bool:
        return self.state == NavigationState.ARRIVED

    @property
    def is_stopping(self) -> bool:
        return self.state in (NavigationState.STOPPING, NavigationState.ARRIVED)

    @property
    def is_approaching(self) -> bool:
        return self.state == NavigationState.APPROACHING

    @property
    def is_navigating(self) -> bool:
        return self.state == NavigationState.NAVIGATING

