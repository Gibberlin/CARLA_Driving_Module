"""Edge Navigation Client for Node 1 (Hybrid Edge Streaming with Local Failsafe).

Connects Node 1 (Vehicle Onboard Driver) to Node 2 (Edge Navigation Server)
using a persistent HTTP keep-alive session with strict per-tick timeouts and an
automatic local RouteFollower failsafe if Node 2 experiences latency spikes or
disconnects mid-drive.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from navigation.route_follower import RouteFollower
from navigation.router import RoutePlan

try:
    import httpx
except ImportError:
    httpx = None


@dataclass
class _CachedLocation:
    x: float
    y: float
    z: float = 0.5


@dataclass
class _CachedRotation:
    yaw: float
    pitch: float = 0.0
    roll: float = 0.0


class _CachedTransform:
    __slots__ = ("location", "rotation")

    def __init__(self, x: float, y: float, yaw: float):
        self.location = _CachedLocation(x=x, y=y)
        self.rotation = _CachedRotation(yaw=yaw)

    def get_forward_vector(self) -> _CachedLocation:
        rad = math.radians(self.rotation.yaw)
        return _CachedLocation(x=math.cos(rad), y=math.sin(rad), z=0.0)

    def get_right_vector(self) -> _CachedLocation:
        rad = math.radians(self.rotation.yaw + 90.0)
        return _CachedLocation(x=math.cos(rad), y=math.sin(rad), z=0.0)


class _CachedWaypoint:
    __slots__ = ("transform",)

    def __init__(self, x: float, y: float, yaw: float):
        self.transform = _CachedTransform(x=x, y=y, yaw=yaw)


@dataclass
class EdgeRoutePlan:
    route_id: str
    map_name: str
    origin: Tuple[float, float]
    destination: Tuple[float, float]
    total_distance_m: float
    polyline_2d: List[Tuple[float, float]]
    maneuvers: List[str]


@dataclass
class EdgeStepInstruction:
    nav_steer: float
    throttle_hint: float
    brake_hint: float
    target_speed_kmh: float
    maneuver: str
    nav_state: str
    remaining_distance_m: float
    route_progress_pct: float
    needs_replan: bool
    new_polyline_2d: Optional[List[Tuple[float, float]]] = None
    latency_ms: float = 0.0
    using_fallback: bool = False


def _build_local_route_plan(
    waypoints_raw: List[List[float]],
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    total_distance_m: float,
    maneuvers: Optional[List[str]] = None,
) -> RoutePlan:
    wps = [_CachedWaypoint(float(w[0]), float(w[1]), float(w[2])) for w in waypoints_raw]
    poly = [(float(w[0]), float(w[1])) for w in waypoints_raw]
    opts = maneuvers if maneuvers else ["LANEFOLLOW"] * len(wps)
    return RoutePlan(
        success=bool(wps),
        total_distance_m=float(total_distance_m),
        waypoints=wps,
        road_options=opts,
        polyline_2d=poly,
        start_location=origin,
        destination_location=destination,
    )


class EdgeNavClient:
    """Hybrid Edge client with keep-alive pooling and automatic local fallback."""

    def __init__(
        self,
        base_url: str,
        plan_timeout_s: float = 5.0,
        step_timeout_s: float = 0.06,
        step_stride: int = 1,
        target_speed_kmh: float = 25.0,
        arrival_radius_m: float = 2.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.plan_timeout_s = max(0.5, float(plan_timeout_s))
        self.step_timeout_s = max(0.01, float(step_timeout_s))
        self.step_stride = max(1, int(step_stride))
        self.target_speed_kmh = float(target_speed_kmh)
        self.arrival_radius_m = float(arrival_radius_m)

        self._http = (
            httpx.Client(base_url=self.base_url, timeout=self.plan_timeout_s)
            if httpx is not None
            else None
        )

        self.route_id: Optional[str] = None
        self.active_plan: Optional[EdgeRoutePlan] = None
        self._fallback_follower = RouteFollower(
            target_speed_kmh=self.target_speed_kmh,
            arrival_radius_m=self.arrival_radius_m,
        )
        self._tick_count: int = 0
        self._last_instruction: Optional[EdgeStepInstruction] = None
        self.edge_connected: bool = False
        self.consecutive_failures: int = 0

    def _post_json(self, path: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
        if self._http is not None:
            resp = self._http.post(path, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()

        req = urllib.request.Request(
            url=f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_json(self, path: str, timeout_s: float) -> Dict[str, Any]:
        if self._http is not None:
            resp = self._http.get(path, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()

        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def check_health(self) -> Optional[Dict[str, Any]]:
        try:
            data = self._get_json("/health", timeout_s=min(2.0, self.plan_timeout_s))
            self.edge_connected = True
            return data
        except Exception:
            self.edge_connected = False
            return None

    def plan_route(
        self,
        map_name: str,
        origin: Tuple[float, float, float] | Tuple[float, float],
        destination: Optional[Tuple[float, float]] = None,
    ) -> Optional[EdgeRoutePlan]:
        """Requests a route from Node 2 using the vehicle's current standing `origin`."""
        payload: Dict[str, Any] = {
            "map_name": map_name,
            "origin": [float(v) for v in origin],
            "target_speed_kmh": self.target_speed_kmh,
            "arrival_radius_m": self.arrival_radius_m,
        }
        if destination is not None:
            payload["destination"] = [float(destination[0]), float(destination[1])]

        data = self._post_json("/plan_route", payload, timeout_s=self.plan_timeout_s)
        if data.get("status") != "SUCCESS":
            return None

        waypoints_raw = data["waypoints"]
        origin_2d = (float(data["origin"][0]), float(data["origin"][1]))
        dest_2d = (float(data["destination"][0]), float(data["destination"][1]))
        total_dist = float(data["total_distance_m"])
        maneuvers = list(data.get("maneuvers", []))

        local_plan = _build_local_route_plan(
            waypoints_raw=waypoints_raw,
            origin=origin_2d,
            destination=dest_2d,
            total_distance_m=total_dist,
            maneuvers=maneuvers,
        )
        self._fallback_follower.set_route(local_plan)

        self.route_id = data["route_id"]
        self.active_plan = EdgeRoutePlan(
            route_id=self.route_id,
            map_name=map_name,
            origin=origin_2d,
            destination=dest_2d,
            total_distance_m=total_dist,
            polyline_2d=local_plan.polyline_2d,
            maneuvers=maneuvers,
        )
        self._tick_count = 0
        self._last_instruction = None
        self.edge_connected = True
        self.consecutive_failures = 0
        return self.active_plan

    def step_navigation(
        self,
        x: float,
        y: float,
        yaw_deg: float,
        speed_kmh: float,
        delta_s: float = 0.05,
        failsafe_blocked: bool = False,
    ) -> EdgeStepInstruction:
        """Gets navigation instruction from Node 2, falling back locally on timeout/error."""
        if self.route_id is None or self.active_plan is None:
            raise RuntimeError("No active route; call plan_route() first.")

        self._tick_count += 1

        # Always keep the local fallback follower synchronized with current vehicle pose
        fb_steer, fb_throttle, fb_brake, fb_replan = self._fallback_follower.step(
            current_x=x,
            current_y=y,
            current_yaw_deg=yaw_deg,
            speed_kmh=speed_kmh,
            delta_s=delta_s,
            failsafe_blocked=failsafe_blocked,
        )

        # Optional stride throttling to reduce network/CPU overhead when desired
        if (
            self.step_stride > 1
            and (self._tick_count % self.step_stride) != 1
            and self._last_instruction is not None
            and not failsafe_blocked
        ):
            return self._last_instruction

        t0 = time.perf_counter()
        try:
            resp = self._post_json(
                "/navigation/step",
                {
                    "route_id": self.route_id,
                    "x": float(x),
                    "y": float(y),
                    "yaw_deg": float(yaw_deg),
                    "speed_kmh": float(speed_kmh),
                    "delta_s": float(delta_s),
                    "failsafe_blocked": bool(failsafe_blocked),
                },
                timeout_s=self.step_timeout_s,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.edge_connected = True
            self.consecutive_failures = 0

            new_poly = None
            if resp.get("new_waypoints"):
                new_wps = resp["new_waypoints"]
                new_plan = _build_local_route_plan(
                    waypoints_raw=new_wps,
                    origin=(float(x), float(y)),
                    destination=self.active_plan.destination,
                    total_distance_m=float(resp.get("remaining_distance_m", 0.0)),
                )
                self._fallback_follower.set_route(new_plan)
                self.active_plan.polyline_2d = new_plan.polyline_2d
                new_poly = new_plan.polyline_2d

            instruction = EdgeStepInstruction(
                nav_steer=float(resp["nav_steer"]),
                throttle_hint=float(resp["throttle_hint"]),
                brake_hint=float(resp["brake_hint"]),
                target_speed_kmh=float(resp["target_speed_kmh"]),
                maneuver=str(resp["maneuver"]),
                nav_state=str(resp["nav_state"]),
                remaining_distance_m=float(resp["remaining_distance_m"]),
                route_progress_pct=float(resp["route_progress_pct"]),
                needs_replan=bool(resp["needs_replan"]),
                new_polyline_2d=new_poly,
                latency_ms=round(latency_ms, 2),
                using_fallback=False,
            )
            self._last_instruction = instruction
            return instruction

        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.edge_connected = False
            self.consecutive_failures += 1

            instruction = EdgeStepInstruction(
                nav_steer=float(fb_steer),
                throttle_hint=float(fb_throttle),
                brake_hint=float(fb_brake),
                target_speed_kmh=self.target_speed_kmh,
                maneuver="FALLBACK",
                nav_state=self._fallback_follower.state.value,
                remaining_distance_m=float(self._fallback_follower.remaining_distance_m),
                route_progress_pct=0.0,
                needs_replan=bool(fb_replan),
                new_polyline_2d=None,
                latency_ms=round(latency_ms, 2),
                using_fallback=True,
            )
            self._last_instruction = instruction
            return instruction

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None
