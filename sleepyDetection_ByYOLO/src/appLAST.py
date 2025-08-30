# =============================================================
# 🚗 Driver Safety Detector (YOLO11 + Streamlit + Alarm System)
# =============================================================

import streamlit as st
import cv2
import tempfile
import time
import threading
from collections import defaultdict
from ultralytics import YOLO
import pygame
import pandas as pd

# -----------------------------
# Initialize pygame mixer (for audio siren)
# -----------------------------
pygame.mixer.init()

SIREN_FILE = "siren.wav"
try:
    siren = pygame.mixer.Sound(SIREN_FILE)
except:
    st.error("❌ Siren sound file not found. Please place siren.wav in your project folder.")

# -----------------------------
# Load YOLO Model
# -----------------------------
MODEL_PATH = "best.pt"   # make sure best.pt is in the same folder
model = YOLO(MODEL_PATH)

# -----------------------------
# Risky behavior thresholds (in seconds)
# -----------------------------
SIREN_THRESHOLDS = {
    "Eyes Closed": 3.0,
    "Nodding Off": 3.0,
    "Texting": 8.0,
    "Talking on the phone": 8.0,
    "Yawning": 8.0,
    "Drinking": 8.0,
    "Operating the Radio": 8.0,
    "Reaching Behind": 8.0,
    "Hair and Makeup": 8.0,
    "Talking to Passenger": 8.0,
}

# -----------------------------
# Alarm control variables
# -----------------------------
audio_active = False
alarm_active = False
alarm_volume = 0.2
siren.set_volume(alarm_volume)

# -----------------------------
# Alarm Functions
# -----------------------------
def play_siren():
    """Background loop that plays siren while audio_active=True."""
    global audio_active, alarm_volume
    while audio_active:
        siren.play()
        if alarm_volume < 1.0:   # gradually increase volume
            alarm_volume = min(1.0, alarm_volume + 0.05)
            siren.set_volume(alarm_volume)
        time.sleep(0.5)

def start_alarm():
    """Start alarm thread if not already active."""
    global audio_active, alarm_active, alarm_volume
    if not audio_active:
        audio_active = True
        alarm_active = True
        alarm_volume = 0.2
        siren.set_volume(alarm_volume)
        threading.Thread(target=play_siren, daemon=True).start()

def stop_alarm():
    """Stop alarm and reset flags."""
    global audio_active, alarm_active
    audio_active = False
    alarm_active = False
    siren.stop()

# =============================================================
# Streamlit User Interface
# =============================================================

st.title("🚗 Driver Safety Detector (YOLO11)")
st.markdown(
    """
    Upload a driver video, and the app will detect unsafe behaviors.  
    If a risky behavior lasts beyond its threshold, a **siren alarm** will trigger 🚨  
    """
)

# -----------------------------
# File Uploader
# -----------------------------
uploaded_file = st.file_uploader("Upload Driver Video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Open video with OpenCV
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps if fps > 0 else 0.03   # seconds per frame

    # State trackers
    active_behaviors = defaultdict(float)   # behavior durations
    stframe = st.empty()                    # video display
    alert_placeholder = st.empty()          # alert text
    progress_placeholder = st.empty()       # progress bar

    st.markdown("### ⏳ Video Processing...")

    frame_counter = 0

    # -----------------------------
    # Process video frame by frame
    # -----------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break   # video finished

        # Run YOLO detection
        results = model(frame)
        risky_detected = False
        risky_class = None
        frame_counter += 1

        # Loop over detections
        for r in results:
            boxes = r.boxes
            names = model.names

            for box in boxes:
                cls_id = int(box.cls[0])
                cls = names[cls_id]

                # Clean class name (remove prefixes like "c1 - Texting")
                cls_clean = cls.split(" - ")[-1].strip() if " - " in cls else cls
                x1, y1, x2, y2 = map(int, box.xyxy[0])   # bounding box

                # If risky class
                if cls_clean in SIREN_THRESHOLDS:
                    active_behaviors[cls_clean] += frame_time
                    elapsed = active_behaviors[cls_clean]
                    threshold = SIREN_THRESHOLDS[cls_clean]

                    if elapsed >= threshold:
                        # 🚨 Dangerous behavior detected
                        risky_detected = True
                        risky_class = cls_clean

                        # Blinking rectangle (red/yellow)
                        alert_red = (frame_counter // 5) % 2 == 0
                        alert_color = (0, 0, 255) if alert_red else (0, 255, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), alert_color, 4)

                        # Start alarm if not active
                        if not alarm_active:
                            start_alarm()
                    else:
                        # Behavior seen but below threshold → green box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    # Non-risky behavior → cyan box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                # Add class name on top of box
                cv2.putText(frame, cls_clean, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # -----------------------------
        # Update Streamlit UI (Alerts & Progress)
        # -----------------------------
        if risky_detected:
            elapsed = active_behaviors[risky_class]
            threshold = SIREN_THRESHOLDS[risky_class]
            progress_value = min(1.0, elapsed / threshold)

            alert_placeholder.markdown(
                f"<h2 style='color:red;'>⚠️ ALERT: {risky_class} detected! 🚨</h2>",
                unsafe_allow_html=True
            )
            progress_placeholder.progress(progress_value, text=f"{elapsed:.1f}s / {threshold:.1f}s")

        else:
            # No danger → clear alert and stop alarm
            alert_placeholder.empty()
            progress_placeholder.empty()
            stop_alarm()

        # Show frame in Streamlit
        stframe.image(frame, channels="BGR")

    # Release video after loop
    cap.release()
    stop_alarm()

    # =============================================================
    # Summary Table
    # =============================================================
    summary_data = []
    for behavior, duration in active_behaviors.items():
        threshold = SIREN_THRESHOLDS.get(behavior, 0)
        alarm_triggered = "Yes" if duration >= threshold else "No"
        summary_data.append({
            "Behavior": behavior,
            "Total Duration (s)": round(duration, 2),
            "Alarm Triggered": alarm_triggered
        })

    df_summary = pd.DataFrame(summary_data)

    # Highlight rows where alarm triggered
    def highlight_alarm(row):
        return ['background-color: #ffcccc' if row['Alarm Triggered'] == 'Yes' else '' for _ in row]

    st.markdown("### 📊 Detected Risky Behaviors Summary")
    st.dataframe(df_summary.style.apply(highlight_alarm, axis=1))

    st.success("✅ Video processed successfully!")
