import time
import cv2
import numpy as np
from ultralytics import YOLO

# ============================
# USER SETTINGS
# ============================
VIDEO_SOURCE = "test.webm"      # 0 for webcam or path to video
OUTPUT_FILE  = "test_output.mp4"
SIDE = "left"                   # "left" or "right"
MODEL_NAME = "yolov8n-pose.pt"  # or "yolov8s-pose.pt"
DOWN_DEG = 90
UP_DEG = 160
MIN_HOLD_DOWN_MS = 150
SHOW_FPS = True
DISPLAY_HEIGHT = 900
DISPLAY_WINDOW = True           # False = skip live window, still save video
OUTPUT_FPS = 30
PERSON_CONF_THRESH = 0.7        # *** minimum detection confidence (70%) ***
# ============================

# COCO keypoints
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP           = 11, 12
L_KNEE, R_KNEE         = 13, 14
L_ANKLE, R_ANKLE       = 15, 16

def angle_3pt(a, b, c):
    a, b, c = map(lambda p: np.array(p, float), (a, b, c))
    ba, bc = a - b, c - b
    cosang = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9), -1, 1)
    return float(np.degrees(np.arccos(cosang)))

def put_text_bg(img, text, org, scale=0.7, color=(255,255,255), bg=(0,0,0), thick=2):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x, y - h - base), (x + w, y + base), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

class SquatCounter:
    def __init__(self, down_deg=90, up_deg=160, min_hold_down_ms=150):
        self.down_deg, self.up_deg = down_deg, up_deg
        self.min_hold_down_ms = min_hold_down_ms
        self.state, self.reps, self._down_t = "up", 0, None
    def update(self, knee_angle, now_ms):
        if self.state == "up":
            if knee_angle < self.down_deg:
                if self._down_t is None:
                    self._down_t = now_ms
                elif now_ms - self._down_t >= self.min_hold_down_ms:
                    self.state = "down"
        elif knee_angle > self.up_deg:
            self.reps += 1
            self.state, self._down_t = "up", None
        return self.reps, self.state

def open_capture(src):
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src, str) else cv2.VideoCapture(src)
    if not cap.isOpened() and isinstance(src, str):
        cap = cv2.VideoCapture(src)  # fallback
    return cap

def pick_person(result):
    """Return best (17,3) [x,y,conf] keypoints of a person with conf >= 70%."""
    kps = result.keypoints
    boxes = result.boxes
    if kps is None or len(kps) == 0: 
        return None

    xy = kps.xy.cpu().numpy()
    conf_kp = None if kps.conf is None else kps.conf.cpu().numpy()
    if conf_kp is not None and conf_kp.ndim == 3 and conf_kp.shape[-1] == 1:
        conf_kp = conf_kp.squeeze(-1)

    # If boxes exist, filter by PERSON_CONF_THRESH
    if boxes is not None and boxes.conf is not None:
        box_conf = boxes.conf.cpu().numpy().reshape(-1)
        valid_idx = np.where(box_conf >= PERSON_CONF_THRESH)[0]
        if len(valid_idx) == 0:
            return None  # nobody above threshold
    else:
        # fall back to keypoint confidence mean
        if conf_kp is not None:
            mean_conf = np.nanmean(conf_kp, axis=1)
            valid_idx = np.where(mean_conf >= PERSON_CONF_THRESH)[0]
            if len(valid_idx) == 0:
                return None
        else:
            valid_idx = [0]

    # pick best among remaining
    best_i = valid_idx[0]
    if boxes is not None and boxes.conf is not None:
        best_i = valid_idx[int(np.argmax(boxes.conf.cpu().numpy()[valid_idx]))]
    elif conf_kp is not None:
        means = np.nanmean(conf_kp[valid_idx], axis=1)
        best_i = valid_idx[int(np.nanargmax(means))]

    key_xy = xy[best_i]
    if conf_kp is not None:
        kc = conf_kp[best_i]
        if kc.ndim == 1: kc = kc.reshape(-1, 1)
    else:
        kc = np.ones((key_xy.shape[0], 1))
    return np.hstack([key_xy, kc])

def main():
    model = YOLO(MODEL_NAME)
    side = SIDE.lower()
    counter = SquatCounter(DOWN_DEG, UP_DEG, MIN_HOLD_DOWN_MS)

    cap = open_capture(VIDEO_SOURCE)
    if not cap.isOpened(): 
        raise RuntimeError("Could not open video source.")

    ok, frame0 = cap.read()
    if not ok: raise RuntimeError("Failed to read first frame.")
    h0, w0 = frame0.shape[:2]
    aspect = w0 / max(1, h0)
    out_w, out_h = int(DISPLAY_HEIGHT * aspect), DISPLAY_HEIGHT

    # Prepare output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, OUTPUT_FPS, (out_w, out_h))

    prev_t, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (out_w, out_h))

        results = model.predict(frame, verbose=False)
        result = results[0]
        annotated = result.plot()

        k = pick_person(result)
        if k is not None:
            idx = (L_SHOULDER, L_HIP, L_KNEE, L_ANKLE) if side == "left" else (R_SHOULDER, R_HIP, R_KNEE, R_ANKLE)
            shoulder, hip, knee, ankle = [k[i, :2] for i in idx]
            knee_angle = angle_3pt(hip, knee, ankle)
            hip_angle  = angle_3pt(shoulder, hip, knee)
            reps, state = counter.update(knee_angle, int(time.time() * 1000))

            for p in (shoulder, hip, knee, ankle):
                cv2.circle(annotated, tuple(p.astype(int)), 6, (0, 255, 255), -1)
            put_text_bg(annotated, f"{side.capitalize()} Knee: {knee_angle:.1f}", (10, 30))
            put_text_bg(annotated, f"{side.capitalize()} Hip:  {hip_angle:.1f}", (10, 60))
            put_text_bg(annotated, f"Reps: {reps} State: {state}", (10, 95), 0.85, bg=(40, 80, 40))
            put_text_bg(annotated, f"Down<{DOWN_DEG} Up>{UP_DEG}", (annotated.shape[1] - 280, 30), 0.6)

        if SHOW_FPS:
            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_t)
            prev_t = now
            put_text_bg(annotated, f"FPS: {fps:.1f}", (10, annotated.shape[0] - 15), 0.6, bg=(60, 60, 60))

        writer.write(annotated)
        if DISPLAY_WINDOW:
            cv2.imshow("Squat Analysis (YOLO Pose)", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved annotated video to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
