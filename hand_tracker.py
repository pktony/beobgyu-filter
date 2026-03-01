import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


class HandTracker:
    def __init__(
        self,
        model_path="hand_landmarker.task",
        num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ):
        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = None
        self.frame_count = 0
        self.fps = None

    def __enter__(self):
        self.landmarker = HandLandmarker.create_from_options(self.options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.landmarker:
            self.landmarker.close()

    def detect(self, frame, fps):
        self.frame_count += 1
        timestamp_ms = int(self.frame_count * 1000 / fps)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def draw_landmarks(self, frame, hand_landmarks):
        h, w, _ = frame.shape
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
