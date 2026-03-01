import cv2
from hand_tracker import HandTracker

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

with HandTracker(num_hands=2) as tracker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = tracker.detect(frame, fps)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                tracker.draw_landmarks(frame, hand_landmarks)

        cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
