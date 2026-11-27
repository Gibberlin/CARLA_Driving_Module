import argparse
import csv
import os
import queue
import random
import threading
import traceback
from datetime import datetime

from navigation.arrival_manager import ArrivalManager, NavigationState
from navigation.fixed_route import (
    MODEL_STEER_WEIGHT,
    ROUTE_CHECKPOINT_IDS,
    ROUTE_START_SPAWN_INDEX,
    ROUTE_TOWN,
    blend_model_and_navigation_steering,
    build_navigation_agent,
    build_fixed_route,
    build_route_to_destination,
    parse_coords,
)
from navigation.gps_tracker import GPSTracker
from visualization.hud import draw_navigation_hud

CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
CAMERA_FOV = 90.0
CAMERA_HEIGHT_M = 2.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drive a fixed Town03 checkpoint route using the KITTI controller and navigation"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--weights", default=r"steer_model.pth")
    parser.add_argument("--norm", default=r"steer_normalisation_params.csv")
    parser.add_argument("--fusion-weights", default=None,
                        help="optional multimodal RGB/depth/semantic/LiDAR/radar checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--vehicles", type=int, default=25, help="number of NPC vehicles")
    parser.add_argument("--target-speed", type=float, default=25.0, help="speed in km/h")
    parser.add_argument("--fixed-delta", type=float, default=0.05, help="simulation step in seconds")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="optional test-run time limit")
    parser.add_argument("--video", default=None, help="optional output video path")
    parser.add_argument("--yolo-weights", default="yolov8n.pt")
    parser.add_argument("--yolo-device", default=None, help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--yolo-half", action="store_true")
    parser.add_argument("--no-yolo", action="store_true", help="disable object detection")
    parser.add_argument("--plan-only", action="store_true",
                        help="plan and report the fixed route without driving")
    parser.add_argument("--arrival-radius", type=float, default=3.0,
                        help="destination arrival confirmation radius in meters (default: 3.0)")
    parser.add_argument("--stop-speed-threshold", type=float, default=0.5,
                        help="speed threshold in km/h for full stop (default: 0.5)")
    parser.add_argument("--approach-distance", type=float, default=35.0,
                        help="distance to destination in meters to start decelerating (default: 35.0)")
    parser.add_argument("--deceleration", type=float, default=1.5,
                        help="deceleration rate in m/s^2 for smooth stopping (default: 1.5)")
    parser.add_argument("--arrival-confirm-frames", type=int, default=5,
                        help="consecutive frames stopped within arrival radius to confirm arrival (default: 5)")
    parser.add_argument("--start-coords", default=None,
                        help="optional Start coordinates override as 'x,y' or 'x,y,z'")
    parser.add_argument("--dest-coords", default=None,
                        help="optional Destination coordinates override as 'x,y' or 'x,y,z'")
    parser.add_argument("--checkpoints-config", default=None,
                        help="optional path to custom checkpoints or route JSON file")
    parser.add_argument("--steer-bias", type=float, default=0.0,
                        help="manual steering bias offset trim to subtract from model output")
    parser.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True,
                        help="enable anti-bias symmetric inference 0.5 * (f(x) - f(flip(x))) (default: enabled)")
    return parser.parse_args()


def enqueue_latest(frame_queue, image):
    try:
        frame_queue.put_nowait(image)
    except queue.Full:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put_nowait(image)


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


def main():
    args = parse_args()
    if args.vehicles < 0:
        raise ValueError("--vehicles cannot be negative")
    if args.target_speed < 0:
        raise ValueError("--target-speed cannot be negative")
    if args.fixed_delta <= 0:
        raise ValueError("--fixed-delta must be greater than zero")
    if args.max_seconds < 0:
        raise ValueError("--max-seconds cannot be negative")

    try:
        import carla
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API is unavailable; run with CARLA's Python environment"
        ) from exc

    client = world = traffic_manager = ego = None
    original_settings = None
    sensors, actors = [], []
    writer = telemetry = None
    cv2 = np = torch = None
    yolo = None
    status, error_text = "ok", ""
    sim_time = driven_distance = model_time = 0.0
    planned_distance = 0.0
    frames = 0
    collision_count = lane_invasion_count = 0
    stats_lock = threading.Lock()
    previous_location = None
    model_steer_previous = 0.0
    mode = {"name": "model+navigation"}
    yolo_frame_number = -1
    yolo_ms = 0.0
    yolo_boxes = None
    yolo_classes = None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", f"route_{stamp}.csv")

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        world = client.load_world(ROUTE_TOWN)
        original_settings = world.get_settings()

        start_override = parse_coords(args.start_coords)
        dest_override = parse_coords(args.dest_coords)
        route, planned_distance, global_planner, start_wp, dest_wp = (
            build_route_to_destination(
                world.get_map(),
                start_coords=start_override,
                dest_coords=dest_override,
                resolution=ROUTE_RESOLUTION_M,
                config_path=args.checkpoints_config,
            )
        )
        spawn_points = world.get_map().get_spawn_points()
        destination_coords = (
            dest_wp.transform.location.x,
            dest_wp.transform.location.y,
            dest_wp.transform.location.z,
        )
        if not route:
            raise RuntimeError("The route planner returned an empty route")
        print(
            f"[route] {ROUTE_TOWN}, Start ({start_wp.transform.location.x:.1f}, {start_wp.transform.location.y:.1f}) -> "
            f"Destination ({dest_wp.transform.location.x:.1f}, {dest_wp.transform.location.y:.1f}), "
            f"{planned_distance:.0f} m, {len(route)} waypoints"
        )
        if args.plan_only:
            return 0

        try:
            import cv2
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError("This demo requires OpenCV and PyTorch") from exc
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
            from live_demo import YoloWorker, draw_distance_guide, find_lead
        except ImportError as exc:
            raise RuntimeError(
                "kritty_motion_controller.py, live_demo.py, and CARLA's navigation dependencies must be available"
            ) from exc

        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if args.fusion_weights and not os.path.isfile(args.fusion_weights):
            raise FileNotFoundError(
                f"Fused controller weights not found: {args.fusion_weights}"
            )
        if not args.fusion_weights and not os.path.isfile(args.weights):
            raise FileNotFoundError(f"KITTI controller weights not found: {args.weights}")

        fusion_model = (
            FusedSteeringController(args.fusion_weights, device)
            if args.fusion_weights else None
        )
        model = (
            None if fusion_model is not None
            else KrittySteeringController(args.weights, args.norm, device)
        )

        yolo_device = args.yolo_device or ("0" if device == "cuda" else "cpu")
        if not args.no_yolo:
            yolo = YoloWorker(args.yolo_weights, yolo_device, args.yolo_half)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta
        world.apply_settings(settings)
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_synchronous_mode(True)

        blueprints = world.get_blueprint_library()
        ego_blueprint = blueprints.find("vehicle.tesla.model3")
        ego_spawn = start_wp.transform
        ego_spawn.location.z += 0.5
        ego = world.try_spawn_actor(ego_blueprint, ego_spawn)
        if ego is None:
            raise RuntimeError(
                f"Could not spawn at Start waypoint ({ego_spawn.location.x:.1f}, {ego_spawn.location.y:.1f}); "
                f"ensure the lane is not blocked"
            )
        actors.append(ego)

        vehicle_blueprints = [
            blueprint
            for blueprint in blueprints.filter("vehicle.*")
            if int(blueprint.get_attribute("number_of_wheels")) == 4
        ]
        other_spawns = [
            spawn for spawn in spawn_points
            if spawn.location.distance(ego_spawn.location) > 10.0
        ]
        random.Random(2026).shuffle(other_spawns)
        for spawn in other_spawns[:args.vehicles]:
            npc = world.try_spawn_actor(random.choice(vehicle_blueprints), spawn)
            if npc is not None:
                npc.set_autopilot(True, traffic_manager.get_port())
                actors.append(npc)
        print(f"[traffic] spawned {len(actors) - 1} of {args.vehicles} requested NPC vehicles")

<<<<<<< HEAD
        arrival_manager = ArrivalManager(
            destination=destination_coords,
            arrival_radius=args.arrival_radius,
            stop_speed_threshold_kmh=args.stop_speed_threshold,
            approach_distance=args.approach_distance,
            deceleration=args.deceleration,
            cruise_speed_kmh=args.target_speed,
            confirm_frames=args.arrival_confirm_frames,
        )
        gps_tracker = GPSTracker(
            destination=destination_coords,
            planned_distance_m=planned_distance,
            carla_map=world.get_map(),
        )
        destination_reported = False

        agent = BasicAgent(
=======
        agent = build_navigation_agent(
>>>>>>> 6d07b6d959328e44076af3de8cce1f13b31a4b21
            ego,
            world.get_map(),
            global_planner,
            route,
            args.target_speed,
        )

        camera_queue = queue.Queue(maxsize=1)
        camera_blueprint = blueprints.find("sensor.camera.rgb")
        camera_blueprint.set_attribute("image_size_x", str(CAMERA_WIDTH))
        camera_blueprint.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        camera_blueprint.set_attribute("fov", str(CAMERA_FOV))
        camera = world.spawn_actor(
            camera_blueprint,
            carla.Transform(carla.Location(x=1.5, z=CAMERA_HEIGHT_M)),
            attach_to=ego,
        )
        sensors.append(camera)
        camera.listen(lambda image: enqueue_latest(camera_queue, image))
        fusion_rig = None
        if fusion_model is not None:
            fusion_rig = FusionSensorRig(
                world, ego, carla, args.fixed_delta,
                CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV,
            )
            sensors.extend(fusion_rig.actors)

        collision_blueprint = blueprints.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(
            collision_blueprint, carla.Transform(), attach_to=ego
        )
        sensors.append(collision_sensor)
        lane_blueprint = blueprints.find("sensor.other.lane_invasion")
        lane_sensor = world.spawn_actor(
            lane_blueprint, carla.Transform(), attach_to=ego
        )
        sensors.append(lane_sensor)

        def increment_stat(_event, key):
            _ = _event
            nonlocal collision_count, lane_invasion_count
            with stats_lock:
                if key == "collision":
                    collision_count += 1
                else:
                    lane_invasion_count += 1

        collision_sensor.listen(lambda event: increment_stat(event, "collision"))
        lane_sensor.listen(lambda event: increment_stat(event, "lane_invasion"))

        telemetry = open(csv_path, "w", newline="", encoding="utf-8")
        telemetry_writer = csv.writer(telemetry)
        telemetry_writer.writerow([
            "sim_time_s", "world_frame", "mode", "speed_kmh", "model_steer",
            "route_steer", "command_steer", "throttle", "brake", "distance_m",
            "route_progress_m", "yolo_ms", "yolo_age_frames", "detections", "lead_distance_m",
            "collisions", "lane_invasions", "nav_state", "dist_to_dest_m", "target_speed_kmh",
            "destination_reached",
        ])

<<<<<<< HEAD
        print("[ready] PilotNet + autonomous navigation demo with destination arrival")
        print("[keys] m: PilotNet + navigation | r: navigation baseline | q / Esc: quit")
        print("[mode] PilotNet + navigation steering active")
        print(
            f"[control] Navigation supplies {1.0 - MODEL_STEER_WEIGHT:.0%} of steering "
            f"plus route speed and arrival braking; PilotNet supplies "
=======
        print("[ready] KITTI controller + navigation demo: fixed Town03 checkpoint route")
        print("[keys] m: KITTI controller + navigation | r: navigation baseline | q / Esc: quit")
        print("[mode] KITTI controller + navigation steering active")
        print(
            f"[control] Navigation supplies {1.0 - MODEL_STEER_WEIGHT:.0%} of steering "
            f"plus route speed and hazard braking; the KITTI controller supplies "
>>>>>>> 6d07b6d959328e44076af3de8cce1f13b31a4b21
            f"{MODEL_STEER_WEIGHT:.0%} of steering"
        )
        if yolo is None:
            print("[yolo] disabled")
        else:
            print(f"[yolo] recognizing objects with {args.yolo_weights} on {yolo_device}")

        while True:
            world_frame = world.tick()
            image = camera_queue.get(timeout=10.0)
            frame = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                image.height, image.width, 4
            )[:, :, :3].copy()

            if yolo is not None:
                camera_frame = int(image.frame)
                yolo.submit(camera_frame, frame)
                latest = yolo.latest()
                if latest is not None:
                    yolo_frame_number, yolo_ms, yolo_boxes, yolo_classes = latest

<<<<<<< HEAD
            speed = speed_kmh(ego)
            location = ego.get_location()
            ego_tf = ego.get_transform()
            if previous_location is not None:
                driven_distance += location.distance(previous_location)
            previous_location = location

            # Navigation FSM and GPS updates
            nav_state, dynamic_target_speed = arrival_manager.update(
                current_position=(location.x, location.y, location.z),
                current_speed_kmh=speed,
            )
            gps_tracker.update(
                current_location=(location.x, location.y, location.z),
                current_speed_kmh=speed,
                heading_yaw_deg=ego_tf.rotation.yaw,
                sim_time_s=sim_time,
            )
            gps_tracker.set_navigation_state(nav_state)
            agent.set_target_speed(dynamic_target_speed)

            if arrival_manager.is_arrived() and not destination_reported:
                destination_reported = True
                print(
                    f"\n[arrival] *** DESTINATION REACHED ***\n"
                    f"          Location: ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})\n"
                    f"          Remaining Distance: {arrival_manager.distance_to_destination:.2f} m\n"
                    f"          Vehicle safely stopped. Press 'q' or 'Esc' to exit."
                )

            model_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
            model_input = preprocess(model_frame)
            model_tensor = (
                torch.from_numpy(model_input).unsqueeze(0).to(device).float() / 255.0
            )
            with torch.no_grad():
                if args.symmetric:
                    flipped_tensor = torch.flip(model_tensor, dims=[-1])
                    batch_in = torch.cat([model_tensor, flipped_tensor], dim=0)
                    preds = model(batch_in)
                    model_steer_raw = float((0.5 * (preds[0] - preds[1])).item())
                else:
                    model_steer_raw = float(model(model_tensor).item())
                if args.steer_bias != 0.0:
                    model_steer_raw -= args.steer_bias
=======
            if fusion_model is not None:
                fusion_inputs = fusion_rig.read(int(image.frame))
                fusion_control = fusion_model.predict(
                    frame,
                    fusion_inputs["depth"],
                    fusion_inputs["semantic"],
                    fusion_inputs["lidar"],
                    fusion_inputs["radar"],
                    build_runtime_state(ego),
                )
                model_steer_raw = fusion_control["steer"]
            else:
                sensor_values = build_carla_sensor_features(
                    ego, yolo_boxes, yolo_classes
                )
                model_steer_raw, model_steer_deg = model.predict_steering(
                    sensor_values
                )
            if args.steer_bias != 0.0:
                model_steer_raw -= args.steer_bias
>>>>>>> 6d07b6d959328e44076af3de8cce1f13b31a4b21
            if device == "cuda":
                torch.cuda.synchronize()
            model_steer = float(
                max(-1.0, min(1.0, 0.75 * model_steer_previous + 0.25 * model_steer_raw))
            )
            model_steer_previous = model_steer

            route_steer = 0.0
            throttle = 0.0
            brake = 0.0
            if not agent.done():
                route_control = agent.run_step()
                route_steer = float(route_control.steer)
                throttle = float(route_control.throttle)
                brake = float(route_control.brake)

            if mode["name"] == "model+navigation":
                command_steer = blend_model_and_navigation_steering(
                    model_steer, route_steer
                )
            else:
                command_steer = route_steer

            # Control resolution according to priority hierarchy
            if arrival_manager.is_stopped():
                override = arrival_manager.get_control_override()
                throttle = float(override["throttle"])
                brake = float(override["brake"])
                hand_brake = bool(override["hand_brake"])
                if arrival_manager.state == NavigationState.ARRIVED:
                    command_steer = 0.0
                control = carla.VehicleControl(
                    throttle=throttle,
                    steer=command_steer,
                    brake=brake,
                    hand_brake=hand_brake,
                )
            else:
                control = carla.VehicleControl(
                    throttle=throttle,
                    steer=command_steer,
                    brake=brake,
                )
            if mode["name"] != "route":
                model_time += args.fixed_delta
            ego.apply_control(control)

            route_progress = gps_tracker.driven_distance_m
            lead = (
                find_lead(yolo_boxes, yolo_classes, frame.shape[1], frame.shape[0])
                if yolo is not None and yolo_boxes is not None
                else None
            )
            yolo_age_frames = (
                max(0, int(image.frame) - yolo_frame_number)
                if yolo_frame_number >= 0
                else 0
            )
            sim_time += args.fixed_delta
            frames += 1

            with stats_lock:
                current_collisions = collision_count
                current_invasions = lane_invasion_count
            telemetry_writer.writerow([
                round(sim_time, 2),
                int(world_frame),
                mode["name"],
                round(speed, 3),
                round(model_steer, 4),
                round(route_steer, 4),
                round(command_steer, 4),
                round(throttle, 4),
                round(brake, 4),
                round(driven_distance, 2),
                round(route_progress, 2),
                round(yolo_ms, 2),
                yolo_age_frames,
                0 if yolo_boxes is None else len(yolo_boxes),
                "" if lead is None else round(lead[0], 2),
                current_collisions,
                current_invasions,
                nav_state.value,
                round(arrival_manager.distance_to_destination, 2),
                round(dynamic_target_speed, 1),
                int(arrival_manager.is_arrived()),
            ])
            if frames % 20 == 0:
                telemetry.flush()

            display = frame.copy()
            draw_distance_guide(
                display, cv2, None if lead is None else lead[0]
            )
            if yolo_boxes is not None and yolo_classes is not None:
                class_names = {
                    0: "person", 1: "bicycle", 2: "car",
                    3: "motorcycle", 5: "bus", 7: "truck",
                }
                for box, class_id in zip(yolo_boxes, yolo_classes):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 0), 1)
                    label = class_names.get(int(class_id), str(int(class_id)))
                    cv2.putText(
                        display, label, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1,
                    )
            if lead is not None:
                cv2.putText(
                    display, f"Lead {lead[0]:.1f} m", (8, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1,
                )
            cv2.putText(
                display,
                f"{mode['name']} | {speed:.1f} km/h | model {model_steer:+.2f} "
                f"| steer {command_steer:+.2f}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            display = draw_navigation_hud(
                display,
                state=arrival_manager.state,
                current_speed_kmh=speed,
                target_speed_kmh=dynamic_target_speed,
                dist_to_destination_m=arrival_manager.distance_to_destination,
                total_distance_m=planned_distance,
                progress_pct=gps_tracker.route_progress_pct,
                is_arrived=arrival_manager.is_arrived(),
                top_left=(8, 40),
            )
            if writer is None and args.video:
                height, width = display.shape[:2]
                writer = cv2.VideoWriter(
                    args.video,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    1.0 / args.fixed_delta,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video output: {args.video}")
            if writer is not None:
                writer.write(display)
            cv2.imshow("Our Model demo", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r") and mode["name"] != "route":
                mode["name"] = "route"
                print("[mode] Navigation steering active")
            elif key == ord("m") and mode["name"] != "model+navigation":
                mode["name"] = "model+navigation"
                print("[mode] KITTI controller + navigation steering active")

            if args.max_seconds and sim_time >= args.max_seconds:
                print("[exit] Reached --max-seconds")
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
                print(f"[cleanup] Could not close route CSV: {exc}")
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception as exc:
                print(f"[cleanup] Could not close OpenCV window: {exc}")
        if client is not None:
            safe_destroy(client, carla, sensors + actors)
        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception as exc:
                print(f"[cleanup] Could not disable Traffic Manager sync: {exc}")
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception as exc:
                print(f"[cleanup] Could not restore CARLA settings: {exc}")

        csv_summary = csv_path if telemetry is not None else "not created"
        print(
            f"[summary] status={status} route={ROUTE_TOWN} "
            f"planned_m={planned_distance:.1f} "
            f"driven_m={driven_distance:.1f} model_mode_s={model_time:.1f} "
            f"frames={frames} csv={csv_summary}"
        )
        if error_text:
            print(f"[summary] error={error_text}")

    return 1 if status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
