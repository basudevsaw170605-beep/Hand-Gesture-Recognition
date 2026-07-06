"""
Real-Time Hand Gesture Recognition System
------------------------------------------
Pipeline: Webcam -> MediaPipe Hand Landmarker -> Distance Features
          -> Scaler -> Trained Model -> Gesture Prediction
Features: Prediction smoothing, Confidence threshold, FPS display
"""
import os, sys, numpy as np, cv2, joblib, warnings, collections, time
warnings.filterwarnings('ignore', category=UserWarning)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

# ----------------------- CONFIG -----------------------
MODEL_PATH      = r'hand_landmarker.task'
PIPELINE_PATH   = r'gesture_pipeline.pkl'
MIN_DET_CONF    = 0.6
MIN_TRACK_CONF  = 0.5
NUM_HANDS       = 1
CONFIDENCE_THRESHOLD = 60.0   # % - below this shows "Unknown"
SMOOTHING_WINDOW = 10         # frames for majority voting
SHOW_FPS        = True
CAMERA_ID       = 0
# ------------------------------------------------------

# Label mapping (0-4 -> gesture names)
GESTURE_NAMES = {0: 'UP', 1: 'DOWN', 2: 'PLAY', 3: 'PAUSE', 4: 'STOP'}

# -------------------- LOAD MODELS ---------------------
print("[INFO] Loading MediaPipe Hand Landmarker...")
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=NUM_HANDS,
    min_hand_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRACK_CONF,
)
detector = vision.HandLandmarker.create_from_options(options)

print("[INFO] Loading Gesture Pipeline...")
pipeline = joblib.load(PIPELINE_PATH)
clf_model = pipeline['model']
scaler    = pipeline['scaler']
label_enc = pipeline['label_encoder']
print(f"  Model: {type(clf_model).__name__}")
print(f"  Labels: {label_enc.classes_}")

# ------------------- FEATURE EXTRACTION ---------------
def extract_distance_features(landmarks_xyznorm):
    """
    Input: list of 21 MediaPipe landmarks (each has .x, .y, .z)
    Output: numpy array of 210 pairwise Euclidean distances
    """
    pts = [(lm.x, lm.y, lm.z) for lm in landmarks_xyznorm]
    feats = []
    for i in range(21):
        x1, y1, z1 = pts[i]
        for j in range(i + 1, 21):
            x2, y2, z2 = pts[j]
            d = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) ** 0.5
            feats.append(d)
    return np.array(feats, dtype=np.float32)

# ------------------- PREDICTION SMOOTHING -------------
pred_history = collections.deque(maxlen=SMOOTHING_WINDOW)

def get_smoothed_prediction(probs_array):
    label = np.argmax(probs_array)
    pred_history.append(label)
    # Majority vote
    counter = collections.Counter(pred_history)
    most_common = counter.most_common(1)[0][0]
    confidence = probs_array[most_common] * 100.0
    return most_common, confidence

# ------------------- MAIN LOOP ------------------------
print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    sys.exit(1)

# Set lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()
frame_count = 0
fps = 0

print("[INFO] Press ESC to exit\n")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Mirror
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

    # Detect
    result = detector.detect(mp_image)

    # FPS calc
    frame_count += 1
    if frame_count >= 30:
        curr_time = time.time()
        fps = frame_count / (curr_time - prev_time)
        frame_count = 0
        prev_time = curr_time

    gesture_str = "No Hand"
    conf_str = ""
    hand_str = ""

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        hand_str = result.handedness[0][0].category_name if result.handedness else ""

        # Draw landmarks
        landmarks_xy = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for i, (lx, ly) in enumerate(landmarks_xy):
            cv2.circle(frame, (lx, ly), 4, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (lx + 4, ly - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

        # Draw connections
        connections = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
        ]
        for i, j in connections:
            cv2.line(frame, landmarks_xy[i], landmarks_xy[j], (0, 255, 0), 1)

        # Extract features -> predict
        features = extract_distance_features(hand_landmarks)
        features_scaled = scaler.transform(features.reshape(1, -1))
        probs = clf_model.predict_proba(features_scaled)[0]

        smoothed_label, confidence = get_smoothed_prediction(probs)

        if confidence >= CONFIDENCE_THRESHOLD:
            gesture_str = GESTURE_NAMES.get(smoothed_label, str(smoothed_label))
            conf_str = f"{confidence:.1f}%"
        else:
            gesture_str = "Unknown"
            conf_str = f"{confidence:.1f}%"

    # --- Overlay UI ---
    # Gesture label
    cv2.rectangle(frame, (5, 5), (310, 85), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesture: {gesture_str}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, f"Conf: {conf_str}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Hand: {hand_str}", (15, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

    if SHOW_FPS:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Real-Time Hand Gesture Recognition", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("[INFO] System stopped.")
