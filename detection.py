import cv2 as cv
from ultralytics import YOLO
import sys as sy
import time as ti
from uart_transmitter import build

START = 7
AVOID = 117
KICK = 155
HOLD = 123
DANCE = 111
APPROACH = 156
EMPTY = 0


FRAME_WIDTH = 640
CENTRE_CALC = 255 / FRAME_WIDTH
FPS_SMOOTHING = 0.1
priority = ["person", "pipe", "sports ball"]
mappings = {
    "sports ball": KICK,
    "person": APPROACH,
    "pipe": APPROACH,
}


# this function converts the degree of confidence into bytes to be ready for transmitting
def confidence_to_byte(confidence):

    new_value = confidence * 255
    new_value = round(new_value)

    new_value = min(255, new_value)
    new_value = max(0, new_value)

    return new_value


# calculates the centre of the direction of the identified object
def centre_to_direction(centre):

    transmit = centre * CENTRE_CALC
    transmit = round(transmit)

    transmit = min(255, transmit)
    transmit = max(0, transmit)

    return transmit


cap = cv.VideoCapture(2)

if not cap.isOpened():
    print("Cannot open camera")
    sy.exit()

cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)

model = YOLO("yolo11n.pt")

prevtime = ti.perf_counter()
smoothedFPS = 0

while True:
    ret, img = cap.read()
    currentTime = ti.perf_counter()
    gap = currentTime - prevtime

    instant_fps = 1 / gap
    prevtime = currentTime

    if smoothedFPS == 0:
        smoothedFPS = instant_fps
    else:
        smoothedFPS = FPS_SMOOTHING * instant_fps + (1 - FPS_SMOOTHING) * smoothedFPS

    if not ret:
        print("Can't receive frame")
        break

    results = model.predict(img)

    for r in results:
        im_bgr = r.plot()  # type: ignore
        cv.putText(
            im_bgr,
            f"FPS: {smoothedFPS:.1f}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv.imshow("Webcam", im_bgr)
        boxes = r.boxes  # type: ignore

        best_rank = float("inf")
        best_name = None
        best_conf = 0
        best_direction = 0

        if boxes is not None:
            for box in boxes:
                cls = box.cls
                confidence = box.conf
                centre = box.xywh

                if confidence.item() < 0.5:
                    continue

                name = r.names[int(cls.item())]

                if name in priority:
                    rank = priority.index(name)
                else:
                    rank = len(priority)

                def_conf = confidence_to_byte(confidence.item())
                direction = centre_to_direction(centre[0][0].item())

                if rank < best_rank or (rank == best_rank and def_conf > best_conf):
                    best_rank = rank
                    best_conf = def_conf
                    best_direction = direction
                    best_name = name

        if best_name:
            cmd_codes = mappings.get(best_name, AVOID)
            built_bytes = build(cmd_codes, best_direction, best_conf)
            print(built_bytes)
        else:
            cmd_codes = EMPTY
            built_bytes = build(cmd_codes, best_direction, best_conf)
            print(built_bytes)

    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
