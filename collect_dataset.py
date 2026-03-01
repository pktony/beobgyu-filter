import os
import sys
import cv2
import numpy as np
from hand_tracker import HandTracker
from gestures import GESTURE_MAP

# 관절 연결 인덱스 (부모 → 자식)
PARENT = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
CHILD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# 각도를 구할 벡터 쌍 인덱스
ANGLE_V1 = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
ANGLE_V2 = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

CSV_PATH = "data/gesture_train.csv"

gesture_names = list(GESTURE_MAP.keys())

if len(sys.argv) < 2 or sys.argv[1] not in GESTURE_MAP:
    print(f"Usage: python collect_dataset.py <gesture_name>")
    print(f"  가능한 제스처: {', '.join(gesture_names)}")
    sys.exit(1)

GESTURE_NAME = sys.argv[1]
LABEL = GESTURE_MAP[GESTURE_NAME]
print(f"수집 제스처: {GESTURE_NAME} (label={LABEL})")

IMG_DIR = f"data/images/{GESTURE_NAME}"
os.makedirs(IMG_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
dataset = []

with HandTracker(num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as tracker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = tracker.detect(frame, fps)

        angles = None

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                joint = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

                v = joint[CHILD] - joint[PARENT]
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                dot = np.einsum("nt,nt->n", v[ANGLE_V1], v[ANGLE_V2])
                dot = np.clip(dot, -1.0, 1.0)
                angles = np.degrees(np.arccos(dot))

                tracker.draw_landmarks(frame, hand_landmarks)

        cv2.putText(img=frame, text=f"{GESTURE_NAME}({LABEL}) | Collected: {len(dataset)}", org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=(255, 255, 0), thickness=2)

        cv2.imshow("Collect Dataset", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and angles is not None:
            row = np.append(angles, LABEL)
            dataset.append(row)
            img_path = os.path.join(IMG_DIR, f"sample_{len(dataset):04d}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved sample {len(dataset)}: {row} | image: {img_path}")
        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

if dataset:
    new_data = np.array(dataset, dtype=np.float32)
    if os.path.exists(CSV_PATH):
        existing = np.genfromtxt(CSV_PATH, delimiter=",")
        if existing.ndim == 1:
            existing = existing.reshape(1, -1)
        combined = np.vstack([existing, new_data])
    else:
        combined = new_data
    np.savetxt(CSV_PATH, combined, delimiter=",")
    print(f"Saved {new_data.shape[0]} samples (total: {combined.shape[0]}) -> {CSV_PATH}")
else:
    print("No data collected.")
