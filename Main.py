import pyttsx3
from ultralytics import YOLO

import cv2
import time
from xbox360controller import Xbox360Controller
from collections import defaultdict
import speech_recognition as sr

r = sr.Recognizer()

model = YOLO("yolov8s.pt")

track_history = defaultdict(lambda: [])

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

start_time = time.perf_counter()

display_time = 1
fc = 0
FPS = 0
total_frames = 0
prog_start = time.perf_counter()

FRAME_SIZE = (640, 640)

IN_SIZE = (640, 640)

frame = cap.read()[1]
frame = cv2.resize(frame, FRAME_SIZE)
x_scale_factor = frame.shape[1] / IN_SIZE[0]
y_scale_factor = frame.shape[0] / IN_SIZE[1]
x_orig, y_orig = frame.shape[1], frame.shape[0]

while True:
    total_frames += 1
    TIME = time.perf_counter() - start_time
    success, frame = cap.read()

    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    original_frame = frame.copy()
    original_frame = cv2.resize(original_frame, FRAME_SIZE)
    frame = cv2.resize(frame, IN_SIZE)

    frame_area = frame.shape[0] * frame.shape[1]

    fc += 1

    if (TIME) >= display_time:
        FPS = fc / (TIME)
        fc = 0
        start_time = time.perf_counter()

    fps_disp = "FPS: " + str(FPS)[:5]

    results = model.predict(frame)

    original_frame = cv2.putText(
        original_frame, fps_disp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    original_frame = cv2.putText(original_frame, "Press k to pause", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                 (0, 255, 0), 2)

    original_frame = cv2.putText(original_frame, "Press ESC to exit", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                 (0, 255, 0), 2)

    original_frame = cv2.putText(original_frame, "Press r to restart", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                 (0, 255, 0), 2)
    for pred in results:
        names = pred.names

        for i in range(len(pred.boxes)):
            name = names.get(int(pred.boxes.cls[i]))
            confidence = pred.boxes.conf[i]
            bounding_box = pred.boxes[i].xyxy[0]
            bounding_box = [
                bounding_box[0] * x_scale_factor,
                bounding_box[1] * y_scale_factor,
                bounding_box[2] * x_scale_factor,
                bounding_box[3] * y_scale_factor
            ]

            x, y = int(bounding_box[0]), int(bounding_box[1])
            w, h = int(bounding_box[2] - bounding_box[0]), int(bounding_box[3] - bounding_box[1])

            # # Calculate area of bounding box
            #
            area = (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])

            #

            if 0.4 < area / frame_area and 320 >= (bounding_box[0] + bounding_box[2]) / 2.0:
                with Xbox360Controller(0) as controller:
                    controller.set_rumble(1, 1, 2000)
            if 0.4 < area / frame_area and 320 < (bounding_box[0] + bounding_box[2]) / 2.0:
                with Xbox360Controller(1) as controller1:
                    controller1.set_rumble(1, 1, 2000)
    cv2.imshow("result", original_frame)
    c = cv2.waitKey(1)
    if c == 107:
        time.sleep(0.1)
        while True:
            c = cv2.waitKey(1)
            if c == 107 or c == 27:
                break

            if cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
                break

    if c == 27:
        break

    if cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
        break

    if c == 114:
        track_history.clear()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cap.release()
cv2.destroyAllWindows()

print(f"Avg FPS: {total_frames / (time.perf_counter() - prog_start)}")

