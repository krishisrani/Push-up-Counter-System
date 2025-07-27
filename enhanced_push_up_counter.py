import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime

# Helper to format the counter (just for fun)
def format_counter(count):
    return f"[PUSH-UPS: {count}]"

class EnhancedPushUpCounter:
    def __init__(self):
        # MediaPipe pose setup (I like higher confidence for this one)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # OpenCV video capture (default cam)
        self.cap = cv2.VideoCapture(0)
        
        # Counter variables
        self.counter = 0
        self.stage = "up"  # Start in the 'up' position
        self.left_angle = 0
        self.right_angle = 0
        self.avg_angle = 0
        self.ready = False  # Wait for user to get into position
        # TODO: Add a beep on each rep?
        
        # Timing variables
        self.rep_start_time = None
        self.rep_times = []
        self.last_rep_time = None
        
        # Thresholds for push-up detection
        self.UP_THRESHOLD = 160
        self.DOWN_THRESHOLD = 90
        
        # Form tracking
        self.back_angle = 0
        self.form_warnings = []
        
    def calculate_angle(self, a, b, c):
        # Calculate the angle at b (elbow)
        if a is None or b is None or c is None:
            print("Landmark missing, skipping angle calc")
            return 0
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angle = np.degrees(angle)
        return angle
    
    def calculate_back_angle(self, landmarks):
        # Check back straightness (shoulder-hip-knee)
        try:
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
            return self.calculate_angle(left_shoulder, left_hip, left_knee)
        except:
            print("Back angle landmarks missing!")
            return 0
    
    def check_form(self, landmarks):
        warnings = []
        if self.back_angle < 150:
            warnings.append("Keep your back straight!")
        angle_diff = abs(self.left_angle - self.right_angle)
        if angle_diff > 20:
            warnings.append("Keep both arms at similar angles!")
        # TODO: Add more form checks (e.g., head position)?
        return warnings
    
    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            self.left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            self.right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            self.avg_angle = (self.left_angle + self.right_angle) / 2
            self.back_angle = self.calculate_back_angle(landmarks)
            self.form_warnings = self.check_form(landmarks)
            if 40 < self.avg_angle < 180:
                if not self.ready:
                    if self.avg_angle > self.UP_THRESHOLD:
                        print("Ready! Arms extended, starting counter...")
                        self.ready = True
                    self.mp_drawing.draw_landmarks(
                        image, 
                        results.pose_landmarks, 
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    return image  # Don't count until ready
                # Debug print
                print(f"Avg Angle: {self.avg_angle:.1f}, Stage: {self.stage}, Count: {self.counter}")
                if self.avg_angle > self.UP_THRESHOLD:
                    if self.stage == "down":
                        self.stage = "up"
                        if self.rep_start_time:
                            rep_time = time.time() - self.rep_start_time
                            self.rep_times.append(rep_time)
                            self.last_rep_time = rep_time
                        self.counter += 1
                        print(f"Push-up counted! {format_counter(self.counter)}")
                elif self.avg_angle < self.DOWN_THRESHOLD:
                    if self.stage == "up":
                        self.stage = "down"
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return image
    
    def add_text_overlay(self, image):
        # Counter display (using my helper)
        cv2.putText(image, format_counter(self.counter), 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f'Left Angle: {int(self.left_angle)} deg', 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f'Right Angle: {int(self.right_angle)} deg', 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f'Avg Angle: {int(self.avg_angle)} deg', 
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f'Back Angle: {int(self.back_angle)} deg', 
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if self.stage:
            color = (0, 255, 0) if self.stage == "up" else (0, 0, 255)
            cv2.putText(image, f'Stage: {self.stage.upper()}', 
                        (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if self.last_rep_time:
            cv2.putText(image, f'Last Rep: {self.last_rep_time:.1f}s', 
                        (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.rep_times:
            avg_time = sum(self.rep_times) / len(self.rep_times)
            cv2.putText(image, f'Avg Rep: {avg_time:.1f}s', 
                        (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for i, warning in enumerate(self.form_warnings):
            cv2.putText(image, warning, 
                        (10, 280 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, 'Press "q" to quit, "r" to reset', 
                    (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if not self.ready:
            cv2.putText(image, 'Get into push-up position (arms extended) to start',
                        (10, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # TODO: Show a motivational quote here?
        return image
    
    def reset_counter(self):
        self.counter = 0
        self.stage = "up"
        self.rep_times = []
        self.last_rep_time = None
        self.rep_start_time = None
        self.form_warnings = []
        self.ready = False # Reset ready state
        print("Counter reset!")
    
    def run(self):
        print("Enhanced Push-up Counter Started!")
        print("Position yourself so both arms are visible to the camera")
        print("Press 'q' to quit, 'r' to reset counter")
        # (No full screen, I like to see my desktop too)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            processed_frame = self.process_frame(frame)
            output_frame = self.add_text_overlay(processed_frame)
            cv2.imshow('Enhanced Push-up Counter', output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_counter()
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession ended!")
        print(f"Total push-ups: {self.counter}")
        if self.rep_times:
            print(f"Average rep time: {sum(self.rep_times) / len(self.rep_times):.1f}s")
            print(f"Fastest rep: {min(self.rep_times):.1f}s")
            print(f"Slowest rep: {max(self.rep_times):.1f}s")
        # TODO: Save results to a file?

def main():
    # Main entry point
    try:
        counter = EnhancedPushUpCounter()
        counter.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 