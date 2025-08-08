############################################
############ import statements #############
############################################
import cv2
import numpy as np
# import winsound
import time
import os
from collections import deque
import csv
import mediapipe as mp

############################################
######### Initialize video capture #########
############################################
cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture('videos\WIN_20250729_10_35_56_Pro.mp4')
# cap = cv2.VideoCapture('videos\WhatsApp Video 2025-08-05 um 15.23.47_b4e1a516.mp4')
os.makedirs("movement_clips", exist_ok=True)

############################################
################ Parameters ################
############################################
motion_threshold = 3.5
cooldown = 5  # seconds
recording_fps = 20
buffer = 2  # seconds
buffer_size = int(recording_fps * buffer)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

video_writer = None
pre_motion_buffer = deque(maxlen=buffer_size)  # ring buffer for pre-motion frames
recently_alerted = False
alert_time = 0

paused = False
prev_gray_roi = None

# dense optical flow
pyr_scale=0.5     # pyramid scale
levels=3          # number of pyramid levels
winsize=15        # window size for motion estimation
iterations=3      # iterations at each pyramid level
poly_n=7          # neighborhood size for polynomial expansion
poly_sigma=1.5    # Gaussian smoothing before expansion
flags=0           # no special flags

# mediapipe
static_image_mode = False
max_num_hands = 1
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

############################################
#### Initialize mediapipe hand detection ###
############################################
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = static_image_mode, max_num_hands = max_num_hands,
                       min_detection_confidence = min_detection_confidence,
                       min_tracking_confidence = min_tracking_confidence)
mp_drawing = mp.solutions.drawing_utils

############################################
############# helper functions #############
############################################
# play alert
def play_alert():
    # try:
    #     winsound.Beep(1000, 500)
    # except:
    #     print("[Warning] beep failed.")
    pass

# logging movement
def log_movement_csv(timestamp, filename, event="Movement detected"):
    log_file_path = "movement_log.csv"
    new_file = not os.path.exists(log_file_path)
    try:
        with open(log_file_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if new_file:
                writer.writerow(["Timestamp", "Filename", "Event"])
            writer.writerow([timestamp, filename, event])
        print(f"Movement detected: {timestamp}.")
    except Exception as e:
        print(f"[Error] Failed to write log: {e}")

############################################
################ main loop #################
############################################
while True:
    current_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_out = frame.copy()
    pre_motion_buffer.append(frame.copy())

    # hand detection with mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    roi = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)

            padding = 10
            x = max(x_min - padding, 0)
            y = max(y_min - padding, 0)
            x2 = min(x_max + padding, w)
            y2 = min(y_max + padding, h)

            roi = frame[y:y2, x:x2]
            cv2.rectangle(frame_out, (x,y), (x2,y2), (255,0,0), 2)
            break

    if roi is None or roi.size == 0:
        prev_gray_roi = None
        cv2.imshow("Motion detection (ROI)", frame_out)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or cv2.getWindowProperty("Motion detection (ROI)", cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord('p'):
            paused = not paused
            if paused:
                print("Paused")
            else:
                print("Resumed - waiting 2 seconds before reactivating motion detection...")
                time.sleep(1)
            time.sleep(0.3)
        continue

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # resize previous frame (if needed)
    if prev_gray_roi is not None and gray_roi.shape != prev_gray_roi.shape:
        prev_gray_roi = cv2.resize(prev_gray_roi, (gray_roi.shape[1], gray_roi.shape[0]))

    if prev_gray_roi is None:
        prev_gray_roi = gray_roi.copy()
        continue

    if not paused:
        # optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray_roi, gray_roi, None, 
                                            pyr_scale=pyr_scale, levels=levels, 
                                            winsize=winsize, iterations=iterations,
                                            poly_n=poly_n, poly_sigma=poly_sigma,
                                            flags=flags)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold the magnitude to filter out noise
        motion_mask = cv2.threshold(mag, motion_threshold, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

        # refine with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

        # Find contours, detect motion, alert
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False
        for cont in contours:
            if cv2.contourArea(cont) > 20:
                mx, my, mw, mh = cv2.boundingRect(cont)
                cv2.rectangle(frame_out, (x+mx, y+my), (x+mx+mw, y+my+mh), (0, 255, 0), 2)
                motion_detected = True

        if motion_detected and not recently_alerted:
            alert_time = current_time
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

            # Start recording
            filename = f"movement_clips/movement_{timestamp}.avi"
            video_writer = cv2.VideoWriter(filename, fourcc, recording_fps, (frame_width, frame_height))

            # write pre-motion frames
            for fr in pre_motion_buffer:
                video_writer.write(fr)

            log_movement_csv(timestamp, filename)

            # play alert
            play_alert()
            recently_alerted = True
            alert_time = current_time

        # Write frame if currently recording
        if recently_alerted and video_writer:
            video_writer.write(frame)

        # reset cooldown and stop recording
        if recently_alerted and (current_time - alert_time > cooldown):
            recently_alerted = False
            if video_writer:
                video_writer.release()
                video_writer = None

        # Update previous frame
        prev_gray_roi = gray_roi.copy()

    if paused:
        cv2.putText(frame_out, "Paused", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Motion detection (ROI)", frame_out)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or cv2.getWindowProperty("Motion detection (ROI)", cv2.WND_PROP_VISIBLE) < 1:
        break
    elif key == ord('p'):
        paused = not paused
        print("Paused" if paused else "Resumed")
        time.sleep(0.3)

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()