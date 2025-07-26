import cv2
import numpy as np
import pyttsx3
import time
import dlib
from imutils import face_utils
import sys

# Initialize voice engine
tts_engine = pyttsx3.init()

def voice_alert(message):
    print(f"VOICE ALERT: {message}")  # Debug print
    tts_engine.say(message)
    tts_engine.runAndWait()

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

def process_frame(frame, alert_given, closed_eyes_frames, last_alert_time):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 0)
    import time
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # Draw face landmarks
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        # Draw rectangle around face
        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Drowsiness detection
        if eyes_closed(shape):
            closed_eyes_frames += 1
            cv2.putText(frame, f"Eyes closed: {closed_eyes_frames}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            now = time.time()
            if closed_eyes_frames > 15 and (now - last_alert_time > 5):
                print("ALERT: Drowsiness detected!")
                cv2.putText(frame, "ALERT: Drowsiness detected!", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                voice_alert("Warning: Driver appears drowsy or sleeping!")
                last_alert_time = now
        else:
            closed_eyes_frames = 0
            alert_given = False
        if closed_eyes_frames == 0:
            alert_given = False
    return frame, alert_given, closed_eyes_frames, last_alert_time

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
        alert_given = False
        closed_eyes_frames = 0
        last_alert_time = 0
        frame, alert_given, closed_eyes_frames, last_alert_time = process_frame(frame, alert_given, closed_eyes_frames, last_alert_time)
        cv2.imshow('Car AI Monitor', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video source: {input_path}")
        return
    print("Monitoring started. Press 'q' to quit.")
    alert_given = False
    closed_eyes_frames = 0
    last_alert_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break
        frame, alert_given, closed_eyes_frames, last_alert_time = process_frame(frame, alert_given, closed_eyes_frames, last_alert_time)
        cv2.imshow('Car AI Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
