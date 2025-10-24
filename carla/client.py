# client.py
import carla
import yaml
import os

class CarlaClient:
    """Carla 연결 및 월드 설정 담당"""
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.client = self._connect()
        self.world = self.client.get_world()
        self._setup_world()

    def _load_config(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _connect(self):
        host = self.config["carla"]["host"]
        port = self.config["carla"]["port"]
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        print(f"✅ Connected to CARLA at {host}:{port}")
        return client

    def _setup_world(self):
        world = self.world
        cfg = self.config["carla"]
        settings = world.get_settings()
        settings.synchronous_mode = cfg["synchronous_mode"]
        settings.fixed_delta_seconds = cfg["fixed_delta_seconds"]
        world.apply_settings(settings)
        print(f"🌍 World set to sync={cfg['synchronous_mode']} | Δt={cfg['fixed_delta_seconds']}")

    # -----------------------------
    # 유틸리티 함수들
    # -----------------------------
    def cleanup_world(self):
        """보행자 및 다른 차량 제거"""
        actors = self.world.get_actors()
        walkers = actors.filter('*walker.pedestrian*')
        for w in walkers:
            try: w.destroy()
            except: pass
        vehicles = actors.filter('*vehicle*')
        for v in vehicles:
            try: v.destroy()
            except: pass
        print("🧹 World cleaned (removed walkers & vehicles).")

    def set_all_traffic_lights_green(self):
        """모든 신호등을 녹색으로 고정"""
        for light in self.world.get_actors().filter('*traffic_light*'):
            try:
                light.set_state(carla.TrafficLightState.Green)
                light.freeze(True)
            except: pass
        print("🚦 All traffic lights set to GREEN.")
