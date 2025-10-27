# decision.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import json
import numpy as np
import cv2
from lane_detection.lane_detection import detect_lane  # 사용자 구현 모듈 (결과: offset or steer 제안)



class SharedFrameReader:
    def __init__(self, frame_path, meta_path, width=640, height=480, channels=3):
        self.frame_path = frame_path
        self.meta_path = meta_path
        self.width = width
        self.height = height
        self.channels = channels
        self.last_frame_id = -1
        self.mm = None
        self._load_meta()

    def _load_meta(self):
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.width = meta.get("width", self.width)
            self.height = meta.get("height", self.height)
            self.channels = meta.get("channels", self.channels)
            self.last_frame_id = meta.get("frame_id", -1)
            self.mm = np.memmap(
                self.frame_path,
                dtype=np.uint8,
                mode='r',
                shape=(self.height, self.width, self.channels)
            )
        except Exception as e:
            print(f"[WARN] meta load failed: {e}")

    def read(self):
        """새 프레임 읽기"""
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            fid = meta.get("frame_id", -1)
            if fid == self.last_frame_id:
                return None
            self.last_frame_id = fid
            frame = np.copy(self.mm)
            return frame
        except Exception:
            return None


class DecisionAgent:
    def __init__(self, config: dict):
        data_path = os.path.expanduser(config.get("data_path", "~/ADS_Autonomous-Lane-Detection/data"))
        frame_file = config.get("frame_file", "frame.dat")
        meta_file = config.get("meta_file", "meta.json")
        control_file = config.get("control_file", "control.json")
        self.frame_path = os.path.join(data_path, frame_file)
        self.meta_path = os.path.join(data_path, meta_file)
        self.control_path = os.path.join(data_path, control_file)

        self.width = config.get("frame_w", 640)
        self.height = config.get("frame_h", 480)
        self.channels = config.get("channels", 3)
        self.fps = config.get("fps", 20)
        self.period = 1.0 / self.fps

        self.reader = SharedFrameReader(self.frame_path, self.meta_path, self.width, self.height, self.channels)

    def write_control(self, steer: float):
        data = {
            "steer": float(np.clip(steer, -1.0, 1.0)),
            "timestamp": time.time()
        }
        with open(self.control_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def run(self):
        print("✅ decision.py started.")
        last_time = time.time()
        cv2.namedWindow("Carla View", cv2.WINDOW_NORMAL)

        while True:
            start = time.time()
            frame = self.reader.read()
            if frame is None:
                time.sleep(0.005)
                continue

            steer = 0.0
            try:
                steer = detect_lane(frame)
            except Exception as e:
                print(f"[WARN] lane detection failed: {e}")
                steer = 0.0

            self.write_control(steer)

            try:
                vis = frame.copy()
                h, w, _ = vis.shape
                cv2.circle(vis, (w // 2, h - 30), 4, (0, 0, 255), -1)
                lane_center_x = int(w / 2 - steer * (w / 2))
                cv2.circle(vis, (lane_center_x, h - 30), 4, (0, 255, 0), -1)
                cv2.imshow("Carla View", vis)
                key = cv2.waitKey(1)
                if key == 27:
                    break
            except Exception as e:
                print(f"[WARN] display failed: {e}")

            elapsed = time.time() - start
            sleep_time = self.period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if time.time() - last_time > 0.5:
                print(f"[decision] steer={steer:+.3f}")
                last_time = time.time()

        cv2.destroyAllWindows()