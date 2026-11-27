import argparse
import csv
import json
import math
import os
import queue
import random
import threading
import time
import traceback
from datetime import datetime

try:
    import numpy as np
except ImportError:
    np = None

from navigation.fixed_route import (
    MODEL_STEER_WEIGHT,
    ROUTE_CHECKPOINT_IDS,
    ROUTE_START_SPAWN_INDEX,
    ROUTE_TOWN,
    blend_model_and_navigation_steering,
    build_navigation_agent,
    build_fixed_route,
)


CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
CAMERA_FOV = 90.0
CAMERA_HEIGHT_M = 2.0
LANE_HALF_WIDTH_M = 1.8
DETECTION_CLASSES = [0, 1, 2, 3, 5, 7]
LEAD_CLASSES = {2, 3, 5, 7}
OBSTACLE_STOP_DISTANCE_M = 5.0
OBSTACLE_CLEAR_DISTANCE_M = 6.0


class YoloWorker:
    def __init__(self, weights, device, half):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.half = half
        self.input = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.result = None
        self.error = None
        self.thread = threading.Thread(target=self._run, name="yolo-inference", daemon=True)
        self.thread.start()

    def submit(self, frame_number, frame):
        try:
            self.input.put_nowait((frame_number, frame.copy()))
        except queue.Full:
            try:
                self.input.get_nowait()
            except queue.Empty:
                pass
            self.input.put_nowait((frame_number, frame.copy()))

    def latest(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError("YOLO inference failed") from self.error
            return self.result

    def _run(self):
        while not self.stop_event.is_set():
            try:
                frame_number, frame = self.input.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                started = time.perf_counter()
                inference_args = {
                    "device": self.device,
                    "verbose": False,
                    "classes": DETECTION_CLASSES,
                    "imgsz": 320,
                }
                if self.half:
                    inference_args["quantize"] = 16
                result = self.model(
                    frame,
                    **inference_args,
                )[0]
                elapsed_ms = (time.perf_counter() - started) * 1000
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                item = (frame_number, elapsed_ms, boxes, classes)
                with self.lock:
                    self.result = item
            except Exception as exc:
                with self.lock:
                    self.error = exc
                self.stop_event.set()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            print("[cleanup] YOLO worker did not stop within 5 seconds")


def parse_args():
    parser = argparse.ArgumentParser(description="Live CARLA KITTI controller / autopilot demo")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--town", default=None, help="optionally load a map such as Town03")
    parser.add_argument("--weights", default=r"steer_model.pth", help="Kritty steering state-dict path")
    parser.add_argument("--norm", default=r"steer_normalisation_params.csv", help="Kritty steering normalisation parameters")
    parser.add_argument("--fusion-weights", default=None,
                        help="optional multimodal RGB/depth/semantic/LiDAR/radar checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--yolo-weights", default="yolov8n.pt")
    parser.add_argument("--yolo-device", default=None, help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--yolo-half", action="store_true")
    parser.add_argument("--no-yolo", action="store_true", help="disable object detection")
    parser.add_argument("--vehicles", type=int, default=20, help="number of NPC vehicles")
    parser.add_argument("--target-speed", type=float, default=25.0, help="model-mode speed in km/h")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="stop after this simulation time")
    parser.add_argument("--fixed-delta", type=float, default=0.05, help="simulation step in seconds")
    parser.add_argument("--video", default=None, help="optional output video path")
    parser.add_argument("--no-display", action="store_true", help="run without an OpenCV window")
    parser.add_argument("--start-model", dest="start_autopilot", action="store_false",
                        help="start with the KITTI controller instead of CARLA autopilot")
    parser.set_defaults(start_autopilot=True)
    parser.add_argument(
        "--navigation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use the fixed Town03 route (maximum 3000 m) with navigation-assisted model steering (default: disabled)",
    )
    parser.add_argument(
        "--minimap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable road minimap overlay and route planning (default: enabled)",
    )
    parser.add_argument("--minimap-position", choices=("top-right", "top-left"), default="top-right",
                        help="minimap overlay position on display (default: top-right)")
    parser.add_argument("--minimap-size", type=int, default=260,
                        help="minimap pixel dimensions (width and height, default: 260)")
    parser.add_argument("--minimap-opacity", type=float, default=0.85,
                        help="minimap overlay opacity between 0.0 and 1.0 (default: 0.85)")
    parser.add_argument("--checkpoints-config", default=None,
                        help="optional path to custom checkpoints JSON file")
    parser.add_argument("--refresh-map-cache", action="store_true",
                        help="force re-extracting map road topology from live CARLA world")
    parser.add_argument("--steer-bias", type=float, default=0.0,
                        help="manual steering bias offset trim to subtract from model output")
    parser.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True,
                        help="enable anti-bias symmetric inference 0.5 * (f(x) - f(flip(x))) (default: enabled)")
    parser.add_argument("--destination", nargs=2, type=float, default=None, metavar=("X", "Y"),
                        help="optional explicit destination world coordinates X Y (default: snaps to minimap destination)")
    parser.add_argument("--edge-nav-url", default=None,
                        help="Node 2 Edge Navigation Server URL (e.g. http://192.168.1.50:8000 or http://127.0.0.1:8000)")
    parser.add_argument("--edge-steer-mode", choices=("blend", "nav_only", "model_only"), default="blend",
                        help="how Node 1 combines PilotNet steering with Node 2 nav_steer (default: blend)")
    parser.add_argument("--edge-timeout", type=float, default=0.06,
                        help="per-tick timeout in seconds before falling back to local cached route (default: 0.06)")
    parser.add_argument("--edge-step-stride", type=int, default=1,
                        help="query Node 2 every N ticks (default: 1 = every tick)")
    return parser.parse_args()


def safe_destroy(client, carla, actors):
    alive_actors = [actor for actor in actors if actor is not None]
    if not alive_actors:
        return
    try:
        responses = client.apply_batch(
            [carla.command.DestroyActor(actor.id) for actor in alive_actors]
        )
        if responses is not None:
            for response in responses:
                if response.error:
                    print(f"[cleanup] Actor destruction failed: {response.error}")
    except Exception as exc:
        print(f"[cleanup] Could not destroy all CARLA actors: {exc}")


def speed_kmh(vehicle):
    velocity = vehicle.get_velocity()
    return 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5


def nearest_obstacle_distance(ego, actors):
    ego_transform = ego.get_transform()
    ego_box = ego.bounding_box
    ego_yaw = math.radians(ego_transform.rotation.yaw)
    cos_yaw, sin_yaw = math.cos(ego_yaw), math.sin(ego_yaw)
    nearest = None

    for actor in actors:
        if actor.id == ego.id:
            continue
        actor_transform = actor.get_transform()
        actor_box = actor.bounding_box
        dx = actor_transform.location.x - ego_transform.location.x
        dy = actor_transform.location.y - ego_transform.location.y
        dz = actor_transform.location.z - ego_transform.location.z
        forward = dx * cos_yaw + dy * sin_yaw
        lateral = -dx * sin_yaw + dy * cos_yaw

        relative_yaw = math.radians(
            actor_transform.rotation.yaw - ego_transform.rotation.yaw
        )
        cos_relative = abs(math.cos(relative_yaw))
        sin_relative = abs(math.sin(relative_yaw))
        actor_forward_extent = (
            cos_relative * actor_box.extent.x + sin_relative * actor_box.extent.y
        )
        actor_lateral_extent = (
            sin_relative * actor_box.extent.x + cos_relative * actor_box.extent.y
        )
        # Safety spot is only for the front and front-sides; back/rear are not needed for failsafe
        actor_front_x = forward + actor_forward_extent
        if actor_front_x <= 0.0:
            # Entire obstacle is behind the vehicle center (rear / back)
            continue
        if forward < 0.0 and actor_front_x < ego_box.extent.x:
            # Obstacle is centered in the rear and does not reach the front bumper (rear-side)
            continue

        if forward >= ego_box.extent.x + actor_forward_extent:
            forward_gap = forward - ego_box.extent.x - actor_forward_extent
        else:
            # Front-side overlap (alongside front bumper or front portion of ego)
            forward_gap = 0.0

        lateral_gap = max(
            0.0, abs(lateral) - ego_box.extent.y - actor_lateral_extent
        )
        vertical_gap = max(
            0.0, abs(dz) - ego_box.extent.z - actor_box.extent.z
        )
        distance = math.sqrt(
            forward_gap**2 + lateral_gap**2 + vertical_gap**2
        )
        if nearest is None or distance < nearest:
            nearest = distance

    return nearest


def should_apply_emergency_brake(active, obstacle_distance):
    if obstacle_distance is None:
        return False
    limit = OBSTACLE_CLEAR_DISTANCE_M if active else OBSTACLE_STOP_DISTANCE_M
    return obstacle_distance <= limit


def traffic_light_status(vehicle):
    light = vehicle.get_traffic_light()
    if light is None:
        return "NONE"
    state = light.get_state()
    return str(getattr(state, "name", state)).rsplit(".", 1)[-1].upper()


def find_lead(boxes, classes, width, height):
    cx, cy = width / 2.0, height / 2.0
    focal = width / (2.0 * np.tan(np.radians(CAMERA_FOV / 2.0)))
    lead = None
    for box, class_id in zip(boxes, classes):
        x1, _, x2, y2 = box
        if class_id not in LEAD_CLASSES or y2 <= cy + 3:
            continue
        distance = focal * CAMERA_HEIGHT_M / (y2 - cy)
        lateral = ((x1 + x2) / 2.0 - cx) * distance / focal
        if abs(lateral) < LANE_HALF_WIDTH_M and (lead is None or distance < lead[0]):
            lead = (distance, x1, y2, x2)
    return lead


def draw_distance_guide(frame, cv2, lead_distance=None):
    height, width = frame.shape[:2]
    center_x, horizon_y = width / 2.0, height / 2.0
    focal = width / (2.0 * np.tan(np.radians(CAMERA_FOV / 2.0)))

    def ground(distance, lateral):
        x = center_x + focal * lateral / distance
        y = horizon_y + focal * CAMERA_HEIGHT_M / distance
        return int(round(x)), int(round(y))

    def lead_color(distance):
        if distance < 8:
            return (0, 0, 255)
        if distance < 15:
            return (0, 255, 255)
        return (0, 255, 0)

    far = min(max(3.0, lead_distance), 30.0) if lead_distance is not None else 30.0
    shade_color = lead_color(lead_distance) if lead_distance is not None else (255, 200, 0)
    polygon = np.array([
        ground(3.0, -LANE_HALF_WIDTH_M),
        ground(3.0, LANE_HALF_WIDTH_M),
        ground(far, LANE_HALF_WIDTH_M),
        ground(far, -LANE_HALF_WIDTH_M),
    ], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], shade_color)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0.0, frame)

    for distance in (5, 10, 20, 30):
        left = ground(distance, -LANE_HALF_WIDTH_M)
        right = ground(distance, LANE_HALF_WIDTH_M)
        cv2.line(frame, left, right, (255, 255, 255), 1)
        cv2.putText(
            frame,
            f"{distance}m",
            (left[0] - 38, left[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
    return frame


def main():
    args = parse_args()

    try:
        import cv2
        import torch
    except ImportError as exc:
        raise RuntimeError("This demo requires OpenCV and PyTorch to be installed") from exc

    try:
        from kritty_motion_controller import (
            KrittySteeringController,
            build_carla_sensor_features,
        )
        from fused_control_model import (
            FusedSteeringController,
            build_runtime_state,
        )
        from fused_carla_sensors import FusionSensorRig
    except ImportError as exc:
        raise RuntimeError("Steering controller modules must be available in the project directory") from exc

    try:
        import carla
    except ImportError as exc:
        raise RuntimeError("CARLA Python API is unavailable; run with CARLA's Python environment") from exc

    if args.fixed_delta <= 0:
        raise ValueError("--fixed-delta must be greater than zero")
    if args.vehicles < 0:
        raise ValueError("--vehicles cannot be negative")
    if args.target_speed < 0:
        raise ValueError("--target-speed cannot be negative")
    if args.fusion_weights and not os.path.isfile(args.fusion_weights):
        raise FileNotFoundError(f"Fused controller weights not found: {args.fusion_weights}")
    if not args.fusion_weights and not os.path.isfile(args.weights):
        raise FileNotFoundError(f"KITTI controller weights not found: {args.weights}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    fusion_model = (
        FusedSteeringController(args.fusion_weights, device)
        if args.fusion_weights else None
    )
    model = (
        None if fusion_model is not None
        else KrittySteeringController(args.weights, args.norm, device)
    )

    yolo_device = args.yolo_device or ("0" if device == "cuda" else "cpu")
    yolo = None if args.no_yolo else YoloWorker(args.yolo_weights, yolo_device, args.yolo_half)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", f"live_{stamp}.csv")

    client = world = tm = ego = None
    original_settings = None
    sensors, actors = [], []
    camera_queue = queue.Queue(maxsize=1)
    writer = telemetry = None
    navigation_agent = navigation_route = route_planner = None
    planned_route_distance = route_progress_m = 0.0
    navigation_steer = None
    status, error_text = "ok", ""
    sim_time_run = model_time_run = distance = model_distance_run = 0.0
    frames_run = yolo_runs_run = 0
    pipeline_sum_run = yolo_sum_run = 0.0
    previous_location = None
    steer_previous = 0.0
    state = {"auto": args.start_autopilot}
    failsafe_active = False
    last_traffic_light_state = None
    safety_stop_frames = 0
    stats = {"collisions": 0, "invasions": 0, "collisions_model": 0, "invasions_model": 0}
    stats_lock = threading.Lock()
    started_at = time.perf_counter()
    yolo_frame_number = -1
    yolo_ms = 0.0
    yolo_boxes = np.empty((0, 4), dtype=np.float32)
    yolo_classes = np.empty((0,), dtype=int)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        if args.navigation:
            print("[navigation] Route navigation is disabled (limited to Town03; deferred until general multi-town navigation is ready). Running in direct driving mode.")
        world = client.load_world(args.town) if args.town else client.get_world()
        spawn_points = world.get_map().get_spawn_points()
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta
        world.apply_settings(settings)

        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)
        blueprints = world.get_blueprint_library()


        if not spawn_points:
            raise RuntimeError("The current CARLA map has no vehicle spawn points")
        if args.navigation:
            ego_spawn = spawn_points[ROUTE_START_SPAWN_INDEX]
            traffic_spawn_points = [
                spawn for index, spawn in enumerate(spawn_points)
                if index != ROUTE_START_SPAWN_INDEX
            ]
            random.Random(2026).shuffle(traffic_spawn_points)
        else:
            random.shuffle(spawn_points)
            ego_spawn = spawn_points.pop()
            traffic_spawn_points = spawn_points

        ego_bp = blueprints.find("vehicle.tesla.model3")
        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego is None:
            if args.navigation:
                raise RuntimeError(
                    f"Could not spawn at fixed Town03 spawn point "
                    f"{ROUTE_START_SPAWN_INDEX}; ensure it is not occupied"
                )
            raise RuntimeError("Could not spawn the ego vehicle at the selected spawn point")
        actors.append(ego)
        ego.set_autopilot(state["auto"], tm.get_port())

        if args.navigation:
            navigation_agent = build_navigation_agent(
                ego,
                world.get_map(),
                route_planner,
                navigation_route,
                args.target_speed,
            )

        vehicle_bps = [
            blueprint for blueprint in blueprints.filter("vehicle.*")
            if int(blueprint.get_attribute("number_of_wheels")) == 4
        ]
        for spawn in traffic_spawn_points[:args.vehicles]:
            npc = world.try_spawn_actor(random.choice(vehicle_bps), spawn)
            if npc is not None:
                npc.set_autopilot(True, tm.get_port())
                actors.append(npc)

        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=CAMERA_HEIGHT_M)),
            attach_to=ego,
        )
        sensors.append(camera)
        camera.listen(camera_queue.put)
        fusion_rig = None
        if fusion_model is not None:
            fusion_rig = FusionSensorRig(
                world, ego, carla, args.fixed_delta,
                CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV,
            )
            sensors.extend(fusion_rig.actors)

        collision_bp = blueprints.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=ego
        )
        sensors.append(collision_sensor)

        lane_bp = blueprints.find("sensor.other.lane_invasion")
        lane_sensor = world.spawn_actor(lane_bp, carla.Transform(), attach_to=ego)
        sensors.append(lane_sensor)


        def record_event(_event, stat_key, model_key):
            _ = _event
            with stats_lock:
                stats[stat_key] += 1
                if not state["auto"]:
                    stats[model_key] += 1

        collision_sensor.listen(
            lambda event: record_event(event, "collisions", "collisions_model")
        )
        lane_sensor.listen(
            lambda event: record_event(event, "invasions", "invasions_model")
        )

        telemetry = open(csv_path, "w", newline="", encoding="utf-8")
        telemetry_writer = csv.writer(telemetry)
        telemetry_writer.writerow([
            "sim_time_s", "world_frame", "camera_frame", "sensor_lag_frames",
            "mode", "speed_kmh", "autopilot_steer", "steer_raw",
            "navigation_steer", "steer_command", "throttle", "brake",
            "route_progress_m", "steer_ms", "yolo_ms",
            "yolo_age_frames", "pipeline_ms", "detections", "lead_distance_m",
            "collisions", "lane_invasions", "distance_m",
            "obstacle_distance_m", "failsafe_active", "traffic_light_state",
        ])

        minimap = None
        if args.minimap:
            from visualization.minimap import (
                Minimap,
                road_cache_path_for_map,
            )

            map_name = world.get_map().name
            road_cache_path = road_cache_path_for_map(map_name)
            carla_map_arg = (
                world.get_map()
                if args.refresh_map_cache or not os.path.isfile(road_cache_path)
                else None
            )
            minimap = Minimap(
                size=args.minimap_size,
                position=args.minimap_position,
                opacity=args.minimap_opacity,
                carla_map=carla_map_arg,
                road_cache_path=road_cache_path,
                map_name=map_name,
                preferred_start=(
                    ego_spawn.location.x,
                    ego_spawn.location.y,
                ),
            )
            print(
                f"[minimap] Enabled for {map_name} "
                f"({args.minimap_position}, size={args.minimap_size}px, "
                f"opacity={args.minimap_opacity}, "
                f"start={minimap.start_location}, "
                f"destination={minimap.destination_location}, "
                f"cache={road_cache_path})"
            )

        print("[ready] CARLA live demo")
        print("[keys] m: follow route / model control | a: CARLA autopilot | d: pick new destination | q / Esc: quit")

        router = None
        route_follower = None
        NavigationState = None
        route_active = False
        edge_client = None
        edge_inst = None

        try:
            # Determine destination: CLI argument or minimap destination
            dest_coords = None
            if args.destination:
                dest_coords = (float(args.destination[0]), float(args.destination[1]))
                if minimap is not None:
                    minimap.set_destination(*dest_coords)
            elif minimap is not None:
                dest_coords = minimap.destination_location

            if args.edge_nav_url:
                from navigation.edge_client import EdgeNavClient

                edge_client = EdgeNavClient(
                    base_url=args.edge_nav_url,
                    step_timeout_s=args.edge_timeout,
                    step_stride=args.edge_step_stride,
                    target_speed_kmh=args.target_speed,
                    arrival_radius_m=2.5,
                )
                edge_plan = edge_client.plan_route(
                    map_name=world.get_map().name,
                    origin=(ego_spawn.location.x, ego_spawn.location.y, ego_spawn.rotation.yaw),
                    destination=dest_coords,
                )
                if edge_plan is not None:
                    if minimap is not None:
                        minimap.set_destination(*edge_plan.destination)
                        minimap.set_route(edge_plan.polyline_2d, edge_plan.total_distance_m)
                    route_active = True
                    print(
                        f"[edge-node2] Connected to {args.edge_nav_url} | "
                        f"route_id={edge_plan.route_id}, {len(edge_plan.polyline_2d)} waypoints, "
                        f"{edge_plan.total_distance_m:.1f} m to ({edge_plan.destination[0]:.1f}, {edge_plan.destination[1]:.1f})"
                    )
                else:
                    print(f"[edge-node2] Warning: Edge server returned no route; falling back to local router")
                    edge_client.close()
                    edge_client = None

            if edge_client is None:
                from navigation.router import RoadRouter
                from navigation.route_follower import RouteFollower, NavigationState

                router = RoadRouter(world.get_map(), resolution_m=2.0)
                route_follower = RouteFollower(target_speed_kmh=args.target_speed, arrival_radius_m=2.5)

                if dest_coords is not None:
                    plan = router.plan_route(
                        (ego_spawn.location.x, ego_spawn.location.y),
                        dest_coords,
                        origin_heading=ego_spawn.rotation.yaw,
                    )
                    if plan.success:
                        route_follower.set_route(plan)
                        if minimap is not None:
                            minimap.set_route(plan.polyline_2d, plan.total_distance_m)
                        route_active = True
                        print(
                            f"[route] Planned valid road route: {len(plan.polyline_2d)} waypoints, "
                            f"{plan.total_distance_m:.1f} m to destination ({dest_coords[0]:.1f}, {dest_coords[1]:.1f})"
                        )
                    else:
                        print(f"[route] Warning: Could not plan initial route: {plan.error_message}")
        except Exception as exc:
            print(f"[route] Route navigation initialization failed: {exc}")

        while True:
            world_frame = world.tick()
            image = camera_queue.get(timeout=10.0)
            camera_frame = int(image.frame)
            sensor_lag = max(0, int(world_frame) - camera_frame)
            frame = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                image.height, image.width, 4
            )[:, :, :3]
            frame = np.ascontiguousarray(frame)
            if writer is None and args.video:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    args.video, cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / args.fixed_delta,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video output: {args.video}")

            pipeline_start = time.perf_counter()
            if yolo is not None:
                yolo.submit(camera_frame, frame)
                latest = yolo.latest()
                if latest is not None:
                    yolo_frame_number, yolo_ms, yolo_boxes, yolo_classes = latest
            else:
                yolo_frame_number, yolo_ms = camera_frame, 0.0
                yolo_boxes = np.empty((0, 4), dtype=np.float32)
                yolo_classes = np.empty((0,), dtype=int)

            lead = find_lead(yolo_boxes, yolo_classes, frame.shape[1], frame.shape[0])
            world_actors = world.get_actors()
            obstacle_actors = (
                list(world_actors.filter("vehicle.*"))
                + list(world_actors.filter("walker.pedestrian.*"))
                + list(world_actors.filter("static.prop.*"))
            )
            obstacle_distance = nearest_obstacle_distance(ego, obstacle_actors)
            was_failsafe_active = failsafe_active
            failsafe_active = should_apply_emergency_brake(
                failsafe_active, obstacle_distance
            )
            if failsafe_active and not was_failsafe_active:
                if state["auto"]:
                    ego.set_autopilot(False)
                distance_text = (
                    f"{obstacle_distance:.1f} m"
                    if obstacle_distance is not None
                    else "unknown"
                )
                print(
                    f"[safety] Emergency stop: nearby actor at {distance_text}"
                )
            elif was_failsafe_active and not failsafe_active:
                if state["auto"]:
                    ego.set_autopilot(True, tm.get_port())
                print("[safety] Obstacle clear; resuming normal control")

            traffic_light_state = traffic_light_status(ego)
            if traffic_light_state != last_traffic_light_state:
                print(f"[traffic-light] {traffic_light_state}")
                last_traffic_light_state = traffic_light_state

            speed = speed_kmh(ego)
            location = ego.get_location()
            if previous_location is not None:
                step_distance = location.distance(previous_location)
                distance += step_distance
                if not state["auto"]:
                    model_distance_run += step_distance
            previous_location = location

            steer_start = time.perf_counter()
            fusion_control = None
            if fusion_model is not None:
                fusion_inputs = fusion_rig.read(camera_frame)
                fusion_control = fusion_model.predict(
                    frame,
                    fusion_inputs["depth"],
                    fusion_inputs["semantic"],
                    fusion_inputs["lidar"],
                    fusion_inputs["radar"],
                    build_runtime_state(ego),
                )
                steer_raw = fusion_control["steer"]
            else:
                sensor_values = build_carla_sensor_features(
                    ego, yolo_boxes, yolo_classes
                )
                steer_raw, steer_deg = model.predict_steering(sensor_values)
            if args.steer_bias != 0.0:
                steer_raw -= args.steer_bias
            if device == "cuda":
                torch.cuda.synchronize()
            steer_ms = (time.perf_counter() - steer_start) * 1000
            steer_command = float(np.clip(0.75 * steer_previous + 0.25 * steer_raw, -1.0, 1.0))
            steer_previous = steer_command

            navigation_control = None
            if navigation_agent is not None and not state["auto"]:
                navigation_control = navigation_agent.run_step()
                navigation_steer = float(navigation_control.steer)
                remaining_waypoints = len(
                    navigation_agent.get_local_planner().get_plan()
                )
                route_progress_m = planned_route_distance * (
                    1.0 - remaining_waypoints / max(len(navigation_route), 1)
                )
                route_progress_m = max(
                    0.0, min(planned_route_distance, route_progress_m)
                )
            else:
                navigation_steer = None

            # Autonomous road route following and progress tracking
            route_steer = None
            route_throttle = None
            route_brake = None
            edge_inst = None
            if edge_client is not None and edge_client.active_plan is not None:
                edge_inst = edge_client.step_navigation(
                    x=location.x,
                    y=location.y,
                    yaw_deg=ego.get_transform().rotation.yaw,
                    speed_kmh=speed,
                    delta_s=args.fixed_delta,
                    failsafe_blocked=failsafe_active,
                )
                route_steer = edge_inst.nav_steer
                route_throttle = edge_inst.throttle_hint
                route_brake = edge_inst.brake_hint
                if minimap is not None:
                    if edge_inst.new_polyline_2d:
                        minimap.set_route(edge_inst.new_polyline_2d, edge_inst.remaining_distance_m)
                    minimap.update_progress(
                        location.x,
                        location.y,
                        remaining_distance_m=edge_inst.remaining_distance_m,
                        status=edge_inst.nav_state,
                    )
            elif route_follower is not None and NavigationState is not None and route_follower.state != NavigationState.IDLE:
                route_steer, route_throttle, route_brake, needs_replan = route_follower.step(
                    location.x,
                    location.y,
                    ego.get_transform().rotation.yaw,
                    speed,
                    delta_s=args.fixed_delta,
                    failsafe_blocked=failsafe_active,
                )
                if minimap is not None:
                    minimap.update_progress(
                        location.x,
                        location.y,
                        remaining_distance_m=route_follower.remaining_distance_m,
                        status=route_follower.state.value,
                    )
                if needs_replan and router is not None and route_follower.destination is not None:
                    replan_result = router.plan_route(
                        (location.x, location.y),
                        route_follower.destination,
                        origin_heading=ego.get_transform().rotation.yaw,
                    )
                    if replan_result.success:
                        route_follower.set_route(replan_result)
                        if minimap is not None:
                            minimap.set_route(replan_result.polyline_2d, replan_result.total_distance_m)
                        print(f"[route] Replanned route: {replan_result.total_distance_m:.1f} m remaining")

            if state["auto"]:
                control = ego.get_control()
                steer_autopilot = float(control.steer)
                throttle, brake = float(control.throttle), float(control.brake)
            else:
                steer_autopilot = None
                if navigation_control is not None:
                    steer_command = blend_model_and_navigation_steering(
                        steer_command, navigation_steer
                    )
                    throttle = float(navigation_control.throttle)
                    brake = float(navigation_control.brake)
                elif route_active and route_steer is not None:
                    if edge_inst is not None:
                        if args.edge_steer_mode == "blend":
                            steer_command = blend_model_and_navigation_steering(
                                steer_command, route_steer
                            )
                        elif args.edge_steer_mode == "nav_only":
                            steer_command = route_steer
                        # "model_only": keep PilotNet steer_command untouched
                        throttle = route_throttle
                        brake = route_brake
                        if edge_inst.nav_state == "ARRIVED":
                            throttle, brake, steer_command = 0.0, 1.0, 0.0
                    else:
                        steer_command = route_steer
                        throttle = route_throttle
                        brake = route_brake
                        if route_follower is not None and NavigationState is not None and route_follower.state == NavigationState.ARRIVED:
                            throttle, brake, steer_command = 0.0, 1.0, 0.0
                else:
                    if fusion_control is not None:
                        throttle = fusion_control["throttle"]
                        brake = fusion_control["brake"]
                    else:
                        speed_error = args.target_speed - speed
                        throttle = float(np.clip(speed_error * 0.04, 0.0, 0.5))
                        brake = float(np.clip(-speed_error * 0.04, 0.0, 1.0))
                ego.apply_control(carla.VehicleControl(
                    throttle=throttle,
                    steer=steer_command,
                    brake=brake,
                ))
                model_time_run += args.fixed_delta
            if failsafe_active:
                safety_stop_frames += 1
                throttle, brake = 0.0, 1.0
                safety_steer = (
                    steer_autopilot
                    if steer_autopilot is not None
                    else steer_command
                )
                ego.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    steer=safety_steer,
                    brake=1.0,
                ))

            pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
            sim_time_run += args.fixed_delta
            frames_run += 1
            pipeline_sum_run += pipeline_ms
            if yolo is not None and yolo_frame_number >= 0:
                yolo_runs_run += 1
                yolo_sum_run += yolo_ms
            yolo_age_frames = max(0, camera_frame - yolo_frame_number) if yolo_frame_number >= 0 else 0

            with stats_lock:
                collision_count = stats["collisions"]
                invasion_count = stats["invasions"]
            mode_name = (
                "autopilot"
                if state["auto"]
                else "model+navigation" if navigation_agent is not None else "model"
            )
            if failsafe_active:
                mode_name = f"failsafe+{mode_name}"
            telemetry_writer.writerow([
                round(sim_time_run, 2),
                int(world_frame),
                int(camera_frame),
                int(sensor_lag),
                mode_name,
                round(speed, 3),
                "" if steer_autopilot is None else round(steer_autopilot, 4),
                round(steer_raw, 4),
                "" if navigation_steer is None else round(navigation_steer, 4),
                round(steer_command, 4),
                round(throttle, 4),
                round(brake, 4),
                round(route_progress_m, 2),
                round(steer_ms, 2),
                round(yolo_ms, 2),
                int(yolo_age_frames),
                round(pipeline_ms, 2),
                len(yolo_boxes),
                "" if lead is None else round(lead[0], 2),
                collision_count,
                invasion_count,
                round(distance, 2),
                "" if obstacle_distance is None else round(obstacle_distance, 2),
                int(failsafe_active),
                traffic_light_state,
            ])
            if frames_run % 20 == 0:
                telemetry.flush()

            display = frame.copy()
            draw_distance_guide(
                display, cv2, None if lead is None else lead[0]
            )
            for box, class_id in zip(yolo_boxes, yolo_classes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 0), 1)
                cv2.putText(display, str(class_id), (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
            if lead is not None:
                lead_distance, x1, y2, x2 = lead
                cv2.line(display, (int(x1), int(y2)), (int(x2), int(y2)), (0, 0, 255), 3)
                cv2.putText(display, f"{lead_distance:.1f} m",
                            (int(x1), min(int(y2) + 18, display.shape[0] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(
                display,
                f"{mode_name} | {speed:.1f} km/h | steer {steer_command:+.2f} | "
                f"YOLO age {yolo_age_frames}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            traffic_light_color = {
                "RED": (0, 0, 255),
                "YELLOW": (0, 255, 255),
                "GREEN": (0, 255, 0),
            }.get(traffic_light_state, (200, 200, 200))
            cv2.putText(
                display,
                f"Traffic light: {traffic_light_state}",
                (8, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                traffic_light_color,
                1,
            )
            if failsafe_active and obstacle_distance is not None:
                safety_text = f"EMERGENCY BRAKE | obstacle {obstacle_distance:.1f} m"
            elif failsafe_active:
                safety_text = "EMERGENCY BRAKE"
            elif obstacle_distance is not None:
                safety_text = f"Safety clear | nearest actor {obstacle_distance:.1f} m"
            else:
                safety_text = "Safety clear | no nearby actors"
            cv2.putText(
                display,
                safety_text,
                (8, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255) if failsafe_active else (0, 220, 0),
                1,
            )
            if edge_inst is not None:
                link_tag = "FALLBACK" if edge_inst.using_fallback else f"EDGE {edge_inst.latency_ms:.0f}ms"
                nav_msg = (
                    f"{link_tag} | {edge_inst.nav_state} ({edge_inst.maneuver}) "
                    f"| Rem: {edge_inst.remaining_distance_m:.0f}m"
                )
                cv2.putText(
                    display,
                    nav_msg,
                    (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 165, 255) if edge_inst.using_fallback else (0, 255, 200),
                    1,
                )
            elif navigation_agent is not None:
                cv2.putText(
                    display,
                    f"Route {route_progress_m:.0f}/{planned_route_distance:.0f} m "
                    f"| nav {navigation_steer if navigation_steer is not None else 0.0:+.2f}",
                    (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                )
            elif route_follower is not None and NavigationState is not None and route_follower.state != NavigationState.IDLE:
                nav_msg = f"Route: {route_follower.state.value}"
                if route_follower.remaining_distance_m is not None:
                    nav_msg += f" | Rem: {route_follower.remaining_distance_m:.0f}m"
                cv2.putText(
                    display,
                    nav_msg,
                    (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 235, 255),
                    1,
                )
            if minimap is not None:
                ego_tf = ego.get_transform()
                display = minimap.overlay_on_frame(
                    display,
                    vehicle_x=ego_tf.location.x,
                    vehicle_y=ego_tf.location.y,
                    vehicle_yaw=ego_tf.rotation.yaw,
                )
            if writer is not None:
                writer.write(display)
            if not args.no_display:
                cv2.imshow("CARLA live demo", display)

            key = cv2.waitKey(1) & 0xFF if not args.no_display else 255
            if key in (27, ord("q")) or (args.max_seconds and sim_time_run >= args.max_seconds):
                break
            if key == ord("d"):
                if router is not None and minimap is not None and route_follower is not None:
                    try:
                        all_road_pts = [pt for seg in minimap.roads for pt in seg]
                        new_dest = random.choice(all_road_pts)
                        minimap.set_destination(new_dest[0], new_dest[1])
                        new_plan = router.plan_route(
                            (location.x, location.y),
                            new_dest,
                            origin_heading=ego.get_transform().rotation.yaw,
                        )
                        if new_plan.success:
                            minimap.set_route(new_plan.polyline_2d, new_plan.total_distance_m)
                            route_follower.set_route(new_plan)
                            route_active = True
                            state["auto"] = False
                            ego.set_autopilot(False)
                            print(f"[destination] Selected new destination ({new_dest[0]:.1f}, {new_dest[1]:.1f}): {new_plan.total_distance_m:.1f} m")
                    except Exception as exc:
                        print(f"[destination] Could not update destination: {exc}")
            elif key == ord("m"):
                if state["auto"]:
                    ego.set_autopilot(False)
                state["auto"] = False
                if route_follower is not None and route_follower.route_plan is not None:
                    route_active = True
                    print("[mode] Route follower active towards destination")
                else:
                    route_active = False
                    print("[mode] Model driving active")
                steer_previous = 0.0
            elif key == ord("a") and not state["auto"]:
                state["auto"] = True
                route_active = False
                if failsafe_active:
                    print(
                        "[mode] CARLA autopilot will resume after the safety stop"
                    )
                else:
                    ego.apply_control(carla.VehicleControl(
                        throttle=0.0, steer=0.0, brake=0.0
                    ))
                    ego.set_autopilot(True, tm.get_port())
                    print("[mode] CARLA autopilot now has control")
            if route_active and route_follower is not None and NavigationState is not None and route_follower.state == NavigationState.ARRIVED:
                if not state["auto"]:
                    ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            if navigation_agent is not None and not state["auto"] and navigation_agent.done():
                print("[finish] Fixed checkpoint route reached")
                break

    except KeyboardInterrupt:
        status = "interrupted"
        print("\n[exit] Interrupted by user")
    except Exception as exc:
        status = "error"
        error_text = f"{type(exc).__name__}: {exc}"
        print(f"\n[error] {error_text}")
        traceback.print_exc()
    finally:
        if yolo is not None:
            yolo.stop()
        for sensor in sensors:
            try:
                sensor.stop()
            except Exception as exc:
                print(f"[cleanup] Could not stop sensor: {exc}")
        if writer is not None:
            try:
                writer.release()
            except Exception as exc:
                print(f"[cleanup] Could not release video writer: {exc}")
        if telemetry is not None:
            try:
                telemetry.flush()
                telemetry.close()
            except Exception as exc:
                print(f"[cleanup] Could not close telemetry CSV: {exc}")
        try:
            cv2.destroyAllWindows()
        except Exception as exc:
            print(f"[cleanup] Could not close OpenCV windows: {exc}")

        if client is not None:
            safe_destroy(client, carla, sensors + actors)
        if tm is not None:
            try:
                tm.set_synchronous_mode(False)
            except Exception as exc:
                print(f"[cleanup] Could not disable Traffic Manager sync: {exc}")
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception as exc:
                print(f"[cleanup] Could not restore CARLA settings: {exc}")

        with stats_lock:
            if edge_client is not None:
                edge_client.close()
            final_stats = dict(stats)
        summary = {
            "time": stamp,
            "status": status,
            "error": error_text,
            "weights": args.weights,
            "sim_seconds": round(sim_time_run, 2),
            "dist_m": round(distance, 2),
            "model_driving_s": round(model_time_run, 2),
            "model_driving_m": round(model_distance_run, 2),
            "collisions": final_stats["collisions"],
            "lane_invasions": final_stats["invasions"],
            "collisions_while_model_drove": final_stats["collisions_model"],
            "invasions_while_model_drove": final_stats["invasions_model"],
            "safety_stop_frames": safety_stop_frames,
            "mean_pipeline_ms": round(pipeline_sum_run / max(frames_run, 1), 2),
            "mean_yolo_ms": round(yolo_sum_run / max(yolo_runs_run, 1), 2),
            "frames": frames_run,
            "yolo_runs": yolo_runs_run,
            "planned_route_m": round(planned_route_distance, 2),
            "route_progress_m": round(route_progress_m, 2),
            "csv": csv_path,
            "video": args.video or None,
            "params": vars(args),
            "wall_seconds": round(time.perf_counter() - started_at, 2),
        }
        try:
            with open(os.path.join("results", "live_runs.jsonl"), "a", encoding="utf-8") as summary_file:
                summary_file.write(json.dumps(summary) + "\n")
            print("\n[summary]")
            print(json.dumps(summary, indent=2))
        except Exception as exc:
            print(f"[cleanup] Could not write run summary: {exc}")

    return 1 if status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
