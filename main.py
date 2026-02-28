import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
frame_count = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        timestamp_ms = int(frame_count * 1000 / cap.get(cv2.CAP_PROP_FPS))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w, _ = frame.shape
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                for i, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img=frame, center=(cx, cy), radius=4, color=(0, 255, 0), thickness=-1)
                    cv2.putText(img=frame, text=str(i), org=(cx + 5, cy - 5),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(0, 0, 255), thickness=2)

                for conn in HAND_CONNECTIONS:
                    start = hand_landmarks[conn.start]
                    end = hand_landmarks[conn.end]
                    sx, sy = int(start.x * w), int(start.y * h)
                    ex, ey = int(end.x * w), int(end.y * h)
                    cv2.line(img=frame, pt1=(sx, sy), pt2=(ex, ey), color=(255, 255, 255), thickness=2)

        cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
