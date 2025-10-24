import time
import yaml
from client import CarlaClient
from vehicle import SharedFrameStore, DecisionInput, CarlaVehicle, CameraSensor, Controller
import os

# === Config 경로를 main.py 기준으로 설정 ===
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

def main():
    # === 1) Config 로드 ===
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # === 2) 클라이언트 초기화 ===
    client = CarlaClient(config_path)
    world = client.world

    # === 3) 월드 정리 및 신호등 설정 ===
    client.cleanup_world()
    client.set_all_traffic_lights_green()

    # === 4) 차량 생성 ===
    v_cfg = cfg["vehicle"]
    car = CarlaVehicle(world, v_cfg["spawn_index"], v_cfg["blueprint_filter"])

    # === 5) 공유 메모리 & Decision 입력 준비 ===
    shared = SharedFrameStore(cfg["paths"]["shared_path"],
                              cfg["camera"]["width"],
                              cfg["camera"]["height"])
    dec_input = DecisionInput(cfg["paths"]["control_path"])

    # === 6) 컨트롤러 및 카메라 ===
    ctrl = Controller(car, shared, dec_input, cfg)
    cam = CameraSensor(world, car.vehicle, ctrl.camera_callback,
                       cfg["camera"]["width"], cfg["camera"]["height"],
                       cfg["camera"]["fov"], cfg["camera"]["sensor_tick"])
    car.attach_sensor(cam.sensor)

    print("🚘 Simulation started. Press Ctrl+C to stop.")
    try:
        while True:
            world.tick()
    except KeyboardInterrupt:
        print("\n🛑 Stopping simulation...")
    finally:
        car.destroy()


if __name__ == "__main__":
    main()
