# vehicle.py
import os
import time
import json
import numpy as np
import carla

# ===============================
#   Shared Frame Store
# ===============================
class SharedFrameStore:
    def __init__(self, path, width, height):
        self.base = os.path.expanduser(path)
        os.makedirs(self.base, exist_ok=True)
        self.width = width
        self.height = height
        self.channels = 3
        self.frame_path = os.path.join(self.base, "frame.dat")
        self.meta_path = os.path.join(self.base, "meta.json")
        self.mm = np.memmap(self.frame_path, dtype=np.uint8, mode='w+',
                            shape=(self.height, self.width, self.channels))
        self.frame_id = 0
        self._write_meta()

    def _write_meta(self):
        meta = {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": "uint8",
            "format": "BGR",
            "frame_id": self.frame_id,
            "timestamp": time.time()
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def update(self, frame_bgr: np.ndarray):
        if frame_bgr.shape != (self.height, self.width, self.channels):
            return
        self.mm[:] = frame_bgr
        self.mm.flush()
        self.frame_id += 1
        self._write_meta()


# ===============================
#   Decision Input
# ===============================
class DecisionInput:
    def __init__(self, path):
        self.path = os.path.expanduser(path)
        self.last_a = 0.0
        self.last_mtime = 0.0

    def read(self) -> float:
        try:
            if not os.path.exists(self.path):
                self.last_a = 0.0
                return self.last_a
            mtime = os.path.getmtime(self.path)
            if mtime == self.last_mtime:
                return self.last_a
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            a = float(np.clip(data.get("steer", 0.0), -1.0, 1.0))
            self.last_a, self.last_mtime = a, mtime
            return a
        except Exception:
            self.last_a = 0.0
            return 0.0


# ===============================
#   Carla Vehicle Class
# ===============================
class CarlaVehicle:
    def __init__(self, world, spawn_index, blueprint_filter):
        self.world = world
        self.vehicle = None
        self.sensor = None
        self.spawn_vehicle(spawn_index, blueprint_filter)

    def spawn_vehicle(self, index, bp_filter):
        bp_lib = self.world.get_blueprint_library()
        bp = bp_lib.filter(bp_filter)[0]
        spawns = self.world.get_map().get_spawn_points()
        index = index if 0 <= index < len(spawns) else 0
        self.vehicle = self.world.spawn_actor(bp, spawns[index])
        print(f"🚗 Vehicle spawned at index {index} ({bp_filter}).")

    def attach_sensor(self, sensor_actor):
        self.sensor = sensor_actor

    def apply_control(self, throttle, steer):
        ctrl = carla.VehicleControl(
            throttle=float(np.clip(throttle, 0, 1)),
            steer=float(np.clip(steer, -1, 1)),
            brake=0.0
        )
        self.vehicle.apply_control(ctrl)

    def get_speed_kmh(self):
        v = self.vehicle.get_velocity()
        return 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)

    def destroy(self):
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        print("🛑 Vehicle destroyed.")


# ===============================
#   Camera Sensor
# ===============================
class CameraSensor:
    def __init__(self, world, vehicle, callback, width, height, fov, sensor_tick):
        self.sensor = None
        self._setup(world, vehicle, callback, width, height, fov, sensor_tick)

    def _setup(self, world, vehicle, callback, width, height, fov, sensor_tick):
        bp_lib = world.get_blueprint_library()
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(width))
        cam_bp.set_attribute('image_size_y', str(height))
        cam_bp.set_attribute('fov', str(fov))
        cam_bp.set_attribute('sensor_tick', str(sensor_tick))
        cam_tf = carla.Transform(carla.Location(x=1.2, z=1.4))  # carla.Location(x=0.7, z=0.8), carla.Rotation(pitch=-10.0)
        self.sensor = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        self.sensor.listen(callback)

    def stop(self):
        if self.sensor: self.sensor.stop()


# ===============================
#   Controller
# ===============================
class Controller:
    def __init__(self, car, store, dec, cfg):
        self.car = car
        self.store = store
        self.dec = dec
        self.cfg = cfg
        self.last_log = 0.0

    def throttle_from_steer(self, abs_s):
        c = self.cfg["control_policy"]
        if abs_s <= c["steer_slow_thresh"]:
            return c["throttle_base"]
        t = (abs_s - c["steer_slow_thresh"]) / max(1e-6, (c["steer_max"] - c["steer_slow_thresh"]))
        t = float(np.clip(t, 0, 1))
        return c["throttle_base"] - (c["throttle_base"] - c["throttle_min"]) * t

    def camera_callback(self, image):
        raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        frame_bgr = raw[:, :, :3]
        self.store.update(frame_bgr)

        a = self.dec.read()
        steer = float(np.clip(a, -1, 1))
        throttle = self.throttle_from_steer(abs(steer))
        self.car.apply_control(throttle, steer)

        now = time.time()
        if now - self.last_log >= self.cfg["print_period"]:
            print(f"Speed: {self.car.get_speed_kmh():5.1f} km/h | Steer: {steer:+.2f} | Throttle: {throttle:.2f}")
            self.last_log = now
