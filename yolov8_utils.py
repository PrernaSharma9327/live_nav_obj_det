import cv2
import numpy as np
from ultralytics import YOLO

# Load model once
model = YOLO("yolov8n.pt")

# Constants
FOCAL_LENGTH = 600
KNOWN_HEIGHTS = {'person': 1.7, 'chair': 0.5, 'car': 1.5}

def detect_objects(frame):
    height, width = frame.shape[:2]
    zones = {'left': 0, 'center': 0, 'right': 0}
    detections = []

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_name = results.names[int(box.cls[0].cpu().numpy())]

        obj_height = y2 - y1
        real_height = KNOWN_HEIGHTS.get(cls_name, 1.0)
        distance = (real_height * FOCAL_LENGTH) / obj_height if obj_height > 0 else 0
        distance = max(0.1, min(10.0, distance))

        obj_center = (x1 + x2) // 2
        zone = 'left' if obj_center < width/3 else 'right' if obj_center > 2*width/3 else 'center'

        detections.append({'object': cls_name, 'distance': round(distance, 2), 'zone': zone})

        if distance <= 2:
            zones[zone] += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{cls_name} {distance:.1f}m", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    safe_zones = [k for k, v in zones.items() if v == 0]
    obstacle_near = any(zones.values())

    instruction = "Path clear" if not obstacle_near else \
                  f"Safe: {', '.join(safe_zones)}" if safe_zones else "Caution: All paths blocked"

    return {
        "navigation_instruction": instruction,
        "objects_within_2m": [d for d in detections if d['distance'] <= 2],
        "all_detections": detections
    }
