# lane_detection.py
import cv2
import numpy as np

def detect_lane(frame_bgr: np.ndarray) -> float:
    """
    입력: BGR 프레임 (640x480)
    출력: steer (-1.0 ~ +1.0)
    """
    # ROI 설정
    h, w, _ = frame_bgr.shape
    roi = frame_bgr[int(h*0.6):, :]   # 하단 40%

    # 흑백 변환 및 엣지
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 120)

    # 허프 변환으로 차선 추출
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=30, maxLineGap=150)
    if lines is None:
        return 0.0

    # 좌/우 차선 구분
    left_x, right_x = [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if abs(slope) < 0.3:  # 수평선 무시
            continue
        if slope < 0:
            left_x.append(x1)
            left_x.append(x2)
        else:
            right_x.append(x1)
            right_x.append(x2

    if not left_x or not right_x:
        return 0.0

    left_mean = np.mean(left_x)
    right_mean = np.mean(right_x)
    lane_center = (left_mean + right_mean) / 2
    car_center = w / 2
    offset = (car_center - lane_center) / (w / 2)  # -1 ~ +1

    # 양의 값 = 오른쪽 차선쪽으로 쏠림 → 왼쪽 조향 필요
    steer = np.clip(-offset, -1.0, 1.0)
    return steer
