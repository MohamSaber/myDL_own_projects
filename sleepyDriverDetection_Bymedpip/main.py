# main.py
import cv2
import mediapipe as mp
import time
import threading

from settings import (EAR_THRESHOLD, CONSECUTIVE_FRAMES,
                      HEAD_YAW_THRESHOLD, HEAD_PITCH_THRESHOLD,
                      LEFT_EYE_IDX, RIGHT_EYE_IDX, NOSE_IDX,
                      SOUND_ENABLED, SOUND_FREQ, SOUND_DURATION_MS, SOUND_REPEAT_DELAY)
from utils import (landmarks_to_points, calculate_EAR, head_direction, eye_center)

# attempt to import winsound (Windows). If unavailable, sound will be disabled at runtime.
try:
    import winsound
except Exception:
    winsound = None

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def draw_status(frame, ear, direction, closed_frames):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"Head: {direction}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"Closed frames: {closed_frames}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    closed_frames = 0
    alert_on = False
    last_alert_time = 0

    # sound control
    sound_event = threading.Event()
    sound_thread = None

    def play_alert_sound(stop_event):
        """Background loop that beeps while stop_event is set."""
        # if winsound not available, just exit (no blocking)
        if not winsound:
            return
        while stop_event.is_set():
            try:
                winsound.Beep(SOUND_FREQ, int(SOUND_DURATION_MS))
            except Exception:
                # if any issue playing sound, stop trying
                break
            time.sleep(SOUND_REPEAT_DELAY)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            ear_val = 0.0
            head_dir = "unknown"

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # draw mesh (optional)
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style()
                    )

                    pts = landmarks_to_points(face_landmarks, frame.shape)
                    left_ear = calculate_EAR(pts, LEFT_EYE_IDX)
                    right_ear = calculate_EAR(pts, RIGHT_EYE_IDX)
                    ear_val = (left_ear + right_ear) / 2.0

                    # head direction heuristic
                    head_dir, dx, dy = head_direction(pts, LEFT_EYE_IDX, RIGHT_EYE_IDX, nose_idx=NOSE_IDX)

                    # draw eye centers
                    lcenter = eye_center(pts, LEFT_EYE_IDX)
                    rcenter = eye_center(pts, RIGHT_EYE_IDX)
                    cv2.circle(frame, lcenter, 3, (0,255,0), -1)
                    cv2.circle(frame, rcenter, 3, (0,255,0), -1)

                    # draw nose point if available
                    try:
                        nose_pt = pts[NOSE_IDX]
                        cv2.circle(frame, nose_pt, 3, (255,0,0), -1)
                    except Exception:
                        pass

                    # DRAW head direction text near forehead
                    draw_status(frame, ear_val, head_dir, closed_frames)

                    # drowsiness logic
                    if ear_val < EAR_THRESHOLD:
                        closed_frames += 1
                    else:
                        closed_frames = 0
                        # stop visual/sound alert when eyes open
                        alert_on = False
                        if sound_event.is_set():
                            sound_event.clear()

                    # If eyes closed long enough -> alert
                    if closed_frames >= CONSECUTIVE_FRAMES:
                        # prevent constant flicker by regulating alert frequency
                        now = time.time()
                        if not alert_on or (now - last_alert_time) > 2.0:
                            alert_on = True
                            last_alert_time = now

                            # start sound alert thread if enabled and available
                            if SOUND_ENABLED and winsound:
                                if not sound_event.is_set():
                                    sound_event.set()
                                    sound_thread = threading.Thread(target=play_alert_sound, args=(sound_event,), daemon=True)
                                    sound_thread.start()

                        # visual alert
                        cv2.putText(frame, "SLEEP ALERT!", (frame.shape[1]//3, frame.shape[0]//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255), 4)

                    # Head rotation alert
                    # Using dx/dy normalized already inside head_direction; compare to thresholds
                    if abs(dx) > HEAD_YAW_THRESHOLD:
                        cv2.putText(frame, f"HEAD TURN: {head_dir.upper()}", (10, frame.shape[0]-40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,165,255), 2)
                    if dy > HEAD_PITCH_THRESHOLD:
                        cv2.putText(frame, "HEAD DOWN", (10, frame.shape[0]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,165,255), 2)

            else:
                # no face detected
                draw_status(frame, ear_val, "no-face", closed_frames)

            cv2.imshow("Driver Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord('q'):
                break

    # cleanup: stop sound thread if running
    if sound_event.is_set():
        sound_event.clear()
        # daemon thread will exit; no join necessary

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
