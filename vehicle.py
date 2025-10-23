# carla_vehicle.py
import carla
import numpy as np
import time
import os
import json

# ===============================
#   설정
# ===============================
FRAME_W, FRAME_H = 640, 480
SENSOR_TICK = 0.05             # 20 FPS
PRINT_PERIOD = 0.1
SHARED_PATH = "~/ADS/data"      # 프레임/메타 저장 경로
CONTROL_PATH = "~/ADS/data/control.json"  # decision.py가 쓰는 조향 입력 파일

# 속도 정책(조향에 따른 감속)
THROTTLE_BASE = 0.45            # 직진 시 기본 스로틀
THROTTLE_MIN = 0.18             # 급코너 시 최저 스로틀
STEER_SLOW_THRESH = 0.15        # |steer|가 이 값 넘으면 감속 시작
STEER_MAX = 0.70                # 감속 최대치로 가정하는 |steer| 상한 (클램프용)

# 스폰 설정
SPAWN_INDEX = 10                # 10번 인덱스에 스폰
VEHICLE_BP_FILTER = 'model3'    # 동일 차량


# ===============================
#   SharedFrameStore (공유 메모리)
# ===============================
class SharedFrameStore:
    def __init__(self, path="~/ADS/data", width=640, height=480):
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
        if (frame_bgr.shape[0] != self.height or
            frame_bgr.shape[1] != self.width or
            frame_bgr.shape[2] != self.channels):
            return
        self.mm[:] = frame_bgr
        self.mm.flush()
        self.frame_id += 1
        self._write_meta()


# ===============================
#   조향 입력 리더 (decision.py → a)
# ===============================
class DecisionInput:
    """
    ~/ADS/data/control.json 에서 a(steer offset) 읽기
    형식: {"steer": float, "timestamp": <epoch>}
    파일이 없거나 파싱 실패 시 a=0.0
    """
    def __init__(self, path="~/ADS/data/control.json"):
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
                return self.last_a  # 변경 없음
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            a = float(data.get("steer", 0.0))
            # 조향 범위 클램프(-1.0 ~ 1.0)
            a = float(np.clip(a, -1.0, 1.0))
            self.last_a = a
            self.last_mtime = mtime
            return self.last_a
        except Exception:
            # 어떤 이유든 안전하게 직진
            self.last_a = 0.0
            return self.last_a


# ===============================
#   CarlaVehicle (생성/제어)
# ===============================
class CarlaVehicle:
    def __init__(self, client, start_index=SPAWN_INDEX):
        self.client = client
        self.world = client.get_world()
        self.vehicle = None
        self.sensor = None
        self._last_print = 0.0
        self.spawn_vehicle(start_index)

    def spawn_vehicle(self, index):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter(VEHICLE_BP_FILTER)[0]
        spawns = self.world.get_map().get_spawn_points()
        if not spawns:
            raise RuntimeError("No spawn points found.")
        if index >= len(spawns) or index < 0:
            index = 0
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawns[index])

    def attach_sensor(self, sensor_actor):
        self.sensor = sensor_actor

    def apply_control(self, throttle: float, steer: float):
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            steer=float(np.clip(steer, -1.0, 1.0)),
            brake=0.0
        ))

    def get_speed_kmh(self) -> float:
        v = self.vehicle.get_velocity()
        return 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)

    def get_steer(self) -> float:
        ctrl = self.vehicle.get_control()
        return float(ctrl.steer)

    def destroy(self):
        try:
            if self.sensor is not None:
                self.sensor.stop()
                self.sensor.destroy()
        except Exception:
            pass
        try:
            if self.vehicle is not None:
                self.vehicle.destroy()
        except Exception:
            pass


# ===============================
#   CameraSensor (카메라 관리)
# ===============================
class CameraSensor:
    def __init__(self, world, vehicle, callback, width=640, height=480, fov=90, sensor_tick=0.05):
        self.sensor = None
        self._setup(world, vehicle, callback, width, height, fov, sensor_tick)

    def _setup(self, world, vehicle, callback, width, height, fov, sensor_tick):
        bp_lib = world.get_blueprint_library()
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(width))
        cam_bp.set_attribute('image_size_y', str(height))
        cam_bp.set_attribute('fov', str(fov))
        cam_bp.set_attribute('sensor_tick', str(sensor_tick))

        # PiRacer 비율 스케일 반영 카메라 위치 (요구사항: 기존과 동일)
        cam_tf = carla.Transform(
            carla.Location(x=0.7, z=0.8),
            carla.Rotation(pitch=-10.0)
        )
        self.sensor = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        self.sensor.listen(callback)

    def stop(self):
        if self.sensor is not None:
            self.sensor.stop()


# ===============================
#   컨트롤러 (a를 받아 조향/스로틀 적용)
# ===============================
class Controller:
    """
    - 항상 공유 메모리에 프레임 저장
    - decision 입력 a(steer offset)를 읽어 steer = 0 + a 로 적용
    - |steer|가 커질수록 throttle 선형 감속
    - 신호/속도 로그 주기 출력
    """
    def __init__(self, car: CarlaVehicle, shared_store: SharedFrameStore, dec_input: DecisionInput):
        self.car = car
        self.store = shared_store
        self.dec = dec_input
        self._last_print = 0.0

    @staticmethod
    def throttle_from_steer(abs_s: float) -> float:
        # 선형 맵핑: |steer| ∈ [STEER_SLOW_THRESH, STEER_MAX] → throttle ∈ [THROTTLE_BASE, THROTTLE_MIN]
        if abs_s <= STEER_SLOW_THRESH:
            return THROTTLE_BASE
        t = (abs_s - STEER_SLOW_THRESH) / max(1e-6, (STEER_MAX - STEER_SLOW_THRESH))
        t = float(np.clip(t, 0.0, 1.0))
        return THROTTLE_BASE - (THROTTLE_BASE - THROTTLE_MIN) * t

    def camera_callback(self, image):
        # 1) 프레임 공유 저장
        raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        frame_bgr = raw[:, :, :3]
        self.store.update(frame_bgr)

        # 2) a 읽고 제어 적용
        a = self.dec.read()              # steer offset (기본 0)
        steer = float(np.clip(a, -1.0, 1.0))
        throttle = self.throttle_from_steer(abs(steer))
        self.car.apply_control(throttle=throttle, steer=steer)

        # 3) 주기 로그
        now = time.time()
        if now - self._last_print >= PRINT_PERIOD:
            speed = self.car.get_speed_kmh()
            print(f"Speed: {speed:5.1f} km/h | Steering(a): {steer:+.3f} | Throttle: {throttle:.2f}")
            self._last_print = now


# ===============================
#   유틸: 월드 클린업 & 신호등 고정
# ===============================
def cleanup_world(world: carla.World):
    actors = world.get_actors()

    # 1) 보행자(walker) 제거
    walkers = actors.filter('*walker.pedestrian*')
    for w in walkers:
        try:
            w.destroy()
        except Exception:
            pass

    # 2) 다른 차량 제거
    vehicles = actors.filter('*vehicle*')
    for v in vehicles:
        try:
            v.destroy()
        except Exception:
            pass

def set_all_traffic_lights_green(world: carla.World):
    for light in world.get_actors().filter('*traffic_light*'):
        try:
            light.set_state(carla.TrafficLightState.Green)
            light.freeze(True)  # 상태 고정
        except Exception:
            pass


# ===============================
#   main()
# ===============================
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # === 1) 월드 동기화 설정 ===
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = SENSOR_TICK
    world.apply_settings(settings)

    # === 2) 월드 클린업 (보행자/다른 차량 제거) ===
    cleanup_world(world)

    # === 3) 차량 생성 (10번 인덱스) ===
    car = CarlaVehicle(client, start_index=SPAWN_INDEX)

    # === 4) 신호등 전부 녹색 고정 ===
    set_all_traffic_lights_green(world)

    # === 5) 공유 메모리 생성 ===
    shared = SharedFrameStore(path=SHARED_PATH, width=FRAME_W, height=FRAME_H)

    # === 6) decision 입력 리더 준비 ===
    dec_input = DecisionInput(path=CONTROL_PATH)

    # === 7) 카메라 부착 및 콜백 등록 ===
    ctrl = Controller(car, shared, dec_input)
    cam = CameraSensor(world, car.vehicle, callback=ctrl.camera_callback,
                       width=FRAME_W, height=FRAME_H, fov=90, sensor_tick=SENSOR_TICK)
    car.attach_sensor(cam.sensor)

    try:
        while True:
            world.tick()   # 동기화 step
    except KeyboardInterrupt:
        pass
    finally:
        car.destroy()
        print("🛑 Simulation stopped.")


if __name__ == "__main__":
    main()
