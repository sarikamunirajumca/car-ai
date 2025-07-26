# --- Yawn Detection Helper ---
def mouth_aspect_ratio(landmarks):
    # Use mouth landmarks 60-67 (outer mouth)
    mouth = landmarks[60:68]
    A = np.linalg.norm(mouth[2] - mouth[10 % 8])  # 62-66
    B = np.linalg.norm(mouth[4] - mouth[8 % 8])   # 64-68
    C = np.linalg.norm(mouth[0] - mouth[6])       # 60-66
    mar = (A + B) / (2.0 * C)
    return mar

import cv2
import numpy as np
import pyttsx3
import time
import os
import dlib
from imutils import face_utils
import sys
import collections

# Initialize voice engine
tts_engine = pyttsx3.init()

def voice_alert(message):
    print(f"VOICE ALERT: {message}")  # Debug print
    try:
        # Use macOS 'say' command for reliable TTS
        os.system(f'say "{message}"')
    except Exception as e:
        print(f"say command failed: {e}, falling back to pyttsx3")
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()
        time.sleep(0.5)  # Allow audio system to reset

# Load dlib's face detector and facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download this file and place in project root

def eyes_closed(landmarks):
    # Calculate eye aspect ratio (EAR) for both eyes
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    def ear(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)
    left_ear = ear(left_eye)
    right_ear = ear(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    print(f"EAR: {avg_ear:.3f}")  # Debug print for tuning
    return avg_ear < 0.21  # Threshold for closed eyes

# For head pose estimation (motion sickness detection)
HEAD_HISTORY_LEN = 30  # Number of frames to track
head_positions = collections.deque(maxlen=HEAD_HISTORY_LEN)
last_alert_time = 0
last_motion_alert_time = 0

# Helper for head pose estimation
# 3D model points of facial landmarks (nose, eyes, mouth, chin)
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

def get_head_pose(shape, frame):
    # 2D image points from detected landmarks
    image_points = np.array([
        shape[30],     # Nose tip
        shape[8],      # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corner
        shape[48],     # Left mouth corner
        shape[54]      # Right mouth corner
    ], dtype=np.float64)
    size = frame.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = np.float64
    )
    dist_coeffs = np.zeros((4,1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    # Convert rotation vector to Euler angles (pitch, yaw, roll)
    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(rmat[2,1], rmat[2,2])
        yaw = np.arctan2(-rmat[2,0], sy)
        roll = np.arctan2(rmat[1,0], rmat[0,0])
    else:
        pitch = np.arctan2(-rmat[1,2], rmat[1,1])
        yaw = np.arctan2(-rmat[2,0], sy)
        roll = 0
    # Convert to degrees
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)
    return rotation_vector, translation_vector, (pitch, yaw, roll)

def process_frame(frame, closed_eyes_frames):
    global head_positions, last_alert_time, last_motion_alert_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 0)
    import time
    # For improved motion sickness detection
    if not hasattr(process_frame, 'angle_history'):
        process_frame.angle_history = collections.deque(maxlen=HEAD_HISTORY_LEN)
    # --- Distraction/Yawn Alert Timers ---
    if not hasattr(process_frame, 'last_distraction_alert_time'):
        process_frame.last_distraction_alert_time = 0
    if not hasattr(process_frame, 'last_yawn_alert_time'):
        process_frame.last_yawn_alert_time = 0

    # Set alert delays (seconds)
    ALERT_DELAY = 30

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # Draw face landmarks
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        # Draw rectangle around face
        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        now = time.time()
        # Drowsiness detection
        if eyes_closed(shape):
            closed_eyes_frames += 1
            cv2.putText(frame, f"Eyes closed: {closed_eyes_frames}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            if closed_eyes_frames > 15 and (now - last_alert_time > ALERT_DELAY):
                print("ALERT: Drowsiness detected!")
                cv2.putText(frame, "ALERT: Drowsiness detected!", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                voice_alert("Warning: Driver appears drowsy or sleeping!")
                last_alert_time = now
        else:
            closed_eyes_frames = 0

        # --- Distraction Detection (head turned away) ---
        rot_vec, _, (pitch, yaw, roll) = get_head_pose(shape, frame)
        head_positions.append(rot_vec.flatten())
        process_frame.angle_history.append([pitch, yaw, roll])
        # Show angles for debugging
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (x, y+h+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (x, y+h+75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.putText(frame, f"Roll: {roll:.1f}", (x, y+h+90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        # Distraction if yaw or pitch is out of normal range
        if (abs(yaw) > 25 or abs(pitch) > 20) and (now - process_frame.last_distraction_alert_time > ALERT_DELAY):
            print(f"ALERT: Driver distracted! Yaw: {yaw:.1f}, Pitch: {pitch:.1f}")
            cv2.putText(frame, "ALERT: Driver distracted!", (x, y+h+110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,140,255), 2)
            voice_alert("Warning: Driver appears distracted. Please focus on the road!")
            process_frame.last_distraction_alert_time = now

        # --- Yawn Detection ---
        mar = mouth_aspect_ratio(shape)
        cv2.putText(frame, f"MAR: {mar:.2f}", (x, y+h+125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        if mar > 0.7 and (now - process_frame.last_yawn_alert_time > ALERT_DELAY):
            print(f"ALERT: Yawn detected! MAR: {mar:.2f}")
            cv2.putText(frame, "ALERT: Yawn detected!", (x, y+h+140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            voice_alert("Warning: Driver appears to be yawning. Please stay alert!")
            process_frame.last_yawn_alert_time = now

        # --- Improved Motion Sickness Detection ---
        # Calculate std deviation of angles
        if len(process_frame.angle_history) == HEAD_HISTORY_LEN:
            angles = np.array(process_frame.angle_history)
            stds = np.std(angles, axis=0)
            # If any angle's stddev is high, trigger alert
            if (stds > [8, 8, 8]).any() and (now - last_motion_alert_time > ALERT_DELAY):
                print(f"ALERT: Erratic head movement detected! stds: pitch={stds[0]:.2f}, yaw={stds[1]:.2f}, roll={stds[2]:.2f}")
                cv2.putText(frame, "ALERT: Erratic head movement!", (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
                voice_alert("Warning: Rapid or erratic head movement detected. Possible motion sickness!")
                last_motion_alert_time = now
    return frame, closed_eyes_frames

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else 0
    is_image = False
    if isinstance(input_path, str) and (input_path.lower().endswith('.jpg') or input_path.lower().endswith('.png')):
        is_image = True
    if is_image:
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Cannot open image: {input_path}")
            return
        closed_eyes_frames = 0
        frame, closed_eyes_frames = process_frame(frame, closed_eyes_frames)
        cv2.imshow('Car AI Monitor', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video source: {input_path}")
        return
    print("Monitoring started. Press 'q' to quit.")
    closed_eyes_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break
        frame, closed_eyes_frames = process_frame(frame, closed_eyes_frames)
        cv2.imshow('Car AI Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
