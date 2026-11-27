import carla, random, os, time

os.makedirs("frames", exist_ok=True)

client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

vehicle = world.spawn_actor(
    bp_lib.find("vehicle.tesla.model3"),
    random.choice(world.get_map().get_spawn_points()))
vehicle.set_autopilot(True)

cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "640")
cam_bp.set_attribute("image_size_y", "480")
cam = world.spawn_actor(
    cam_bp, carla.Transform(carla.Location(x=1.5, z=2.0)),
    attach_to=vehicle)
cam.listen(lambda img: img.save_to_disk(f"frames/{img.frame:06d}.png"))

try:
    for _ in range(200):
        world.tick()
finally:
    cam.stop()
    time.sleep(2)  # let pending frames finish writing to disk
    cam.destroy()
    vehicle.destroy()
    settings.synchronous_mode = False
    world.apply_settings(settings)