############################################
############ import statements #############
############################################
import cv2
import numpy as np
import time
import os
from collections import deque
import csv
import mediapipe as mp

############################################
######### Initialize video capture #########
############################################
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('videos\WIN_20250729_10_35_56_Pro.mp4')
# cap = cv2.VideoCapture('videos\WhatsApp Video 2025-08-05 um 15.23.47_b4e1a516.mp4')
os.makedirs("movement_clips", exist_ok=True)

############################################
################ Parameters ################
############################################
# motion
motion_threshold = 3.5
cooldown = 5  # seconds
recording_fps = 20
buffer = 2  # seconds
buffer_size = int(recording_fps * buffer)

# frame size placeholder
frame_width = 0
frame_height = 0

# cross-platform
if os.name == "nt":
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    file_ext = ".avi"
else:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    file_ext = ".mp4"

video_writer = None
pre_motion_buffer = deque(maxlen=buffer_size)  # ring buffer for pre-motion frames
last_motion_time = 0.0

paused = False
prev_gray_roi = None

# dense optical flow
pyr_scale = 0.5     # pyramid scale
levels = 3          # number of pyramid levels
winsize = 15        # window size for motion estimation
iterations = 3      # iterations at each pyramid level
poly_n = 7          # neighborhood size for polynomial expansion
poly_sigma = 1.5    # Gaussian smoothing before expansion
flags = 0           # no special flags

# mediapipe
static_image_mode = False
max_num_hands = 1
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

# pause handling
resume_grace = 2.0      # seconds to ignore motion after resume
resume_until = 0.0      # timestamp until which motion is ignored
motion_arm_frames = 3   # require this many consecutive motion frames
motion_frames = 0       # counter of consecutive motion frames

WINDOW_NAME = "Motion detection (ROI)"

############################################
############# helper functions #############
############################################
# play alert
def play_alert():
    try:
        import winsound
        winsound.Beep(1000, 500)
        return
    except Exception:
        pass
    try:
        print("\a", end="", flush=True)
    except Exception:
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

# pause
def toggle_pause():
    global paused, resume_until, prev_gray_roi, motion_frames
    paused = not paused
    if paused:
        print("Paused")
    else:
        print(f"Resumed - arming in {resume_grace:.1f}s")
        resume_until = time.time() + resume_grace
        prev_gray_roi = None        # reset optical flow baseline
        motion_frames = 0           # reset debounce
        pre_motion_buffer.clear()   # drop old pre-roll frames
    time.sleep(0.3)

# ui key window close and so on
def ui_step(frame, win=WINDOW_NAME):
    cv2.imshow(win, frame)
    key = cv2.waitKey(1) & 0xFF
    window_gone = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
    if key in (27, ord('q')) or window_gone:
        return "quit"
    if key == ord('p'):
        toggle_pause()
    return None

# draw status - overlay status text
def draw_status(frame, paused=False, rearming=False):
    if paused:
        cv2.putText(frame, "PAUSED", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    if rearming:
        cv2.putText(frame, "PAUSED", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)

# reset motion baseline and debounce
def reset_motion_state():
    global prev_gray_roi, motion_frames
    prev_gray_roi = None
    motion_frames = 0

############################################
#### Initialize mediapipe hand detection ###
############################################
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = static_image_mode, max_num_hands = max_num_hands,
                       min_detection_confidence = min_detection_confidence,
                       min_tracking_confidence = min_tracking_confidence)

############################################
########### pre-built resources ############
############################################
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

############################################
################ main loop #################
############################################
try:
    while True:
        current_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # init real sizes/gps from the first valid frame
        if frame_width == 0 or frame_height == 0:
            fh, fw = frame.shape[:2]
            if fw == 0 or fh == 0:
                # show blank frame for keeping UI responsive
                blank = np.zeros((240,320,3), dtype=np.uint8)
                if ui_step(blank) == "quit":
                    break
                continue
            frame_width, frame_height = fw, fh
            cam_fps = cap.get(cv2.CAP_PROP_FPS)
            if cam_fps and cam_fps > 0:
                # if better fps, update writer fps and buffer length
                if abs(cam_fps - recording_fps) > 0.1:
                    recording_fps = float(cam_fps)
                    buffer_size = int(recording_fps * buffer)
                    pre_motion_buffer = deque(maxlen=buffer_size)

        frame_out = frame.copy()
        pre_motion_buffer.append(frame)

        # hand detection with mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        roi = None
        x=y=x2=y2 = 0
        # bounding box
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
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

        # reset and ui step when ROI absent or too small
        if roi is None or roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            reset_motion_state()
            draw_status(frame_out, paused=paused)
            if ui_step(frame_out) == "quit":
                break
            continue

        # ROI grey & alignment
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # resize previous frame (if needed)
        if prev_gray_roi is not None and gray_roi.shape != prev_gray_roi.shape:
            prev_gray_roi = cv2.resize(prev_gray_roi, (gray_roi.shape[1], gray_roi.shape[0]))

        # bootstrap baseline if missing
        if prev_gray_roi is None:
            prev_gray_roi = gray_roi.copy()
            draw_status(frame_out, paused=paused)
            if ui_step(frame_out) == "quit":
                break
            continue

        # re-arming grace window - ignore motion but keep baseline fresh
        if current_time < resume_until:
            prev_gray_roi = gray_roi.copy()
            draw_status(frame_out, paused=paused, rearming=True)
            if ui_step(frame_out) == "quit":
                break
            continue

        if not paused:
            # optical flow
            flow = cv2.calcOpticalFlowFarneback( prev_gray_roi, gray_roi, None,
                pyr_scale=pyr_scale, levels=levels, winsize=winsize, iterations=iterations, 
                poly_n=poly_n, poly_sigma=poly_sigma, flags=flags)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Threshold the magnitude to filter out noise
            motion_mask = cv2.threshold(mag, motion_threshold, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

            # refine with morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

            # motion decision: area fraction (more stable than tiny contour area)
            area = int(np.count_nonzero(motion_mask))
            area_frac = area / float(motion_mask.size) if motion_mask.size > 0 else 0.0
            motion_detected = area_frac > 0.01

            # draw contours for visualization
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for cont in contours:
                if cv2.contourArea(cont) > 20:
                    mx, my, mw, mh = cv2.boundingRect(cont)
                    cv2.rectangle(frame_out, (x+mx, y+my), (x+mx+mw, y+my+mh), (0, 255, 0), 2)

            # enforce consecutive frame debounce
            motion_frames = motion_frames + 1 if motion_detected else 0
            ready = motion_frames >= motion_arm_frames

            if ready and video_writer is None:
                last_motion_time = current_time
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

                # Start recording
                filename = f"movement_clips/movement_{timestamp}{file_ext}"
                video_writer = cv2.VideoWriter(filename, fourcc, recording_fps, (frame_width, frame_height))

                # write pre-motion frames
                for fr in pre_motion_buffer:
                    video_writer.write(fr)

                log_movement_csv(timestamp, filename)

                # play alert
                play_alert()

            # Write frame if currently recording
            if video_writer is not None:
                video_writer.write(frame)

            # Update last motion time and stop after a quiet gap
            if motion_detected:
                last_motion_time = current_time

            if video_writer is not None and (current_time - last_motion_time > cooldown):
                video_writer.release()
                video_writer = None

            # Update previous frame
            prev_gray_roi = gray_roi.copy()

        # draw paused banner and pump ui once per iteration
        draw_status(frame_out, paused=paused)
        if ui_step(frame_out) == "quit":
            break

finally:
    cap.release()
    if video_writer:
        video_writer.release()
    try:
        hands.close()
    except Exception:
        pass
    cv2.destroyAllWindows()