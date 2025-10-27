# main.py
import os
import sys
import yaml
import cv2
from decision import DecisionAgent

if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] failed to load config: {e}")
        sys.exit(1)

    agent = DecisionAgent(cfg)
    try:
        agent.run()
    except KeyboardInterrupt:
        print("🛑 decision stopped.")
        cv2.destroyAllWindows()
