# settings.py

# Eye Aspect Ratio threshold and consecutive frame count before alert
EAR_THRESHOLD = 0.25        # lower => more tolerant (eyes must be more closed)
CONSECUTIVE_FRAMES = 20     # number of frames eyes must remain closed to trigger alert

# Head rotation thresholds (normalized by eye distance)
HEAD_YAW_THRESHOLD = 0.18   # left/right turning sensitivity
HEAD_PITCH_THRESHOLD = 0.18 # down/up sensitivity

# Eye landmark indices for MediaPipe FaceMesh
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Nose landmark index used for simple head direction heuristic
NOSE_IDX = 1  # commonly used tip-of-nose index in MediaPipe FaceMesh

# Sound alert configuration
SOUND_ENABLED = True          # master switch for alarm sound
SOUND_FREQ = 1000             # beep frequency in Hz (Windows winsound.Beep)
SOUND_DURATION_MS = 500       # single beep duration in milliseconds
SOUND_REPEAT_DELAY = 0.4      # seconds between beeps while alert is active
