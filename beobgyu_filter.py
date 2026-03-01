import os
import numpy as np
import cv2
from hand_tracker import HandTracker
from gestures import GESTURE_MAP

# label -> 이름 역매핑
LABEL_NAME = {v: k for k, v in GESTURE_MAP.items()}

# 관절 연결 인덱스 (부모 → 자식)
PARENT = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
CHILD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# 각도를 구할 벡터 쌍 인덱스
ANGLE_V1 = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
ANGLE_V2 = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

# KNN 모델 학습
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

CAPTURE_DIR = "data/captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)
capture_count = 0

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

with HandTracker(num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as tracker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = tracker.detect(frame, fps)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                joint = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

                v = joint[CHILD] - joint[PARENT]
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                dot = np.einsum("nt,nt->n", v[ANGLE_V1], v[ANGLE_V2])
                dot = np.clip(dot, -1.0, 1.0)
                angles = np.degrees(np.arccos(dot))

                # KNN 예측
                data = angles.reshape(1, -1).astype(np.float32)
                ret_knn, results, neighbours, dist = knn.findNearest(data, 3)
                predicted_label = int(results[0][0])

                # bbox 계산
                h, w, _ = frame.shape
                xs = [int(lm.x * w) for lm in hand_landmarks]
                ys = [int(lm.y * h) for lm in hand_landmarks]
                margin = 30
                x1 = max(0, min(xs) - margin)
                y1 = max(0, min(ys) - margin)
                x2 = min(w, max(xs) + margin)
                y2 = min(h, max(ys) + margin)

                gesture_name = LABEL_NAME.get(predicted_label, "unknown")

                tracker.draw_landmarks(frame, hand_landmarks)
                box_color = (0, 255, 0)

                # bbox + 제스처 이름 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, gesture_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        cv2.imshow("Beobgyu Filter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            capture_count += 1
            path = os.path.join(CAPTURE_DIR, f"capture_{capture_count:04d}.jpg")
            cv2.imwrite(path, frame)
            print(f"Captured: {path}")
        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
