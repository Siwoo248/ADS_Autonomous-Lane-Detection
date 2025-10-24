# decision.py
import os
import time
import json
import numpy as np
import cv2
from lane_detection import detect_lane  # 사용자 구현 모듈 (결과: offset or steer 제안)

# ===========================
# 설정
# ===========================
DATA_PATH = os.path.expanduser("~/ADS/data")
FRAME_PATH = os.path.join(DATA_PATH, "frame.dat")
META_PATH = os.path.join(DATA_PATH, "meta.json")
CONTROL_PATH = os.path.join(DATA_PATH, "control.json")

FRAME_W, FRAME_H = 640, 480
FPS = 20                # carla_vehicle.py와 동일
PERIOD = 1.0 / FPS


# ===========================
# SharedFrameReader
# ===========================
class SharedFrameReader:
    def __init__(self, frame_path=FRAME_PATH, meta_path=META_PATH):
        self.frame_path = frame_path
        self.meta_path = meta_path
        self.width = FRAME_W
        self.height = FRAME_H
        self.channels = 3
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
            # 메모리 매핑 초기화
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
                return None  # 새 프레임 없음
            self.last_frame_id = fid
            frame = np.copy(self.mm)
            return frame
        except Exception:
            return None


# ===========================
# ControlWriter
# ===========================
def write_control(steer: float):
    data = {
        "steer": float(np.clip(steer, -1.0, 1.0)),
        "timestamp": time.time()
    }
    with open(CONTROL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ===========================
# 메인 루프
# ===========================
def main():
    reader = SharedFrameReader()
    print("✅ decision.py started.")
    last_time = time.time()

    # === 카메라 시점 창 초기화 ===
    cv2.namedWindow("Carla View", cv2.WINDOW_NORMAL)

    while True:
        start = time.time()
        frame = reader.read()
        if frame is None:
            time.sleep(0.005)
            continue

        # === 1) 차선 감지 ===
        steer = 0.0
        try:
            # lane_detection.py 내부에서 offset 계산 후 -1~1 범위로 반환하도록 구현
            steer = detect_lane(frame)
        except Exception as e:
            print(f"[WARN] lane detection failed: {e}")
            steer = 0.0

        # === 2) 제어값 저장 ===
        write_control(steer)

        # === 카메라 시점 표시 ===
        try:
            vis = frame.copy()
            h, w, _ = vis.shape
            # 차량 중심 (빨간 점)
            cv2.circle(vis, (w // 2, h - 30), 4, (0, 0, 255), -1)
            # 차선 중심선 추정 (초록 점)
            lane_center_x = int(w / 2 - steer * (w / 2))
            cv2.circle(vis, (lane_center_x, h - 30), 4, (0, 255, 0), -1)
            cv2.imshow("Carla View", vis)
            key = cv2.waitKey(1)
            if key == 27:  # ESC 키
                break
        except Exception as e:
            print(f"[WARN] display failed: {e}")

        # === 3) FPS 동기화 ===
        elapsed = time.time() - start
        sleep_time = PERIOD - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        # === 4) 로그 ===
        if time.time() - last_time > 0.5:
            print(f"[decision] steer={steer:+.3f}")
            last_time = time.time()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("🛑 decision stopped.")
        cv2.destroyAllWindows()
