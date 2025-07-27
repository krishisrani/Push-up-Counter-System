import cv2
import mediapipe as mp
import numpy as np
import math

# Helper to format the counter (just for fun)
def format_counter(count):
    return f"[PUSH-UPS: {count}]"

class PushUpCounter:
    def __init__(self):
        # MediaPipe pose setup (I like 0.5 confidence, seems to work well)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # OpenCV video capture (default cam)
        self.cap = cv2.VideoCapture(0)
        
        # Counter variables
        self.counter = 0
        self.stage = "up"  # Start in the 'up' position
        self.angle = 0
        self.ready = False  # Wait for user to get into position
        # TODO: Add sound feedback on count?
        
        # Thresholds (tweak if needed)
        self.UP_THRESHOLD = 160
        self.DOWN_THRESHOLD = 90
        
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
    
    def process_frame(self, frame):
        # Convert BGR to RGB for MediaPipe
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
            self.angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            # Only process realistic angles
            if 40 < self.angle < 180:
                if not self.ready:
                    if self.angle > self.UP_THRESHOLD:
                        print("Ready! Arms extended, starting counter...")
                        self.ready = True
                    self.mp_drawing.draw_landmarks(
                        image, 
                        results.pose_landmarks, 
                        self.mp_pose.POSE_CONNECTIONS
                    )
                    return image  # Don't count until ready
                # Debug print
                print(f"Angle: {self.angle:.1f}, Stage: {self.stage}, Count: {self.counter}")
                if self.angle > self.UP_THRESHOLD:
                    if self.stage == "down":
                        self.stage = "up"
                        self.counter += 1
                        print(f"Push-up counted! {format_counter(self.counter)}")
                elif self.angle < self.DOWN_THRESHOLD:
                    if self.stage == "up":
                        self.stage = "down"
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
        return image
    
    def add_text_overlay(self, image):
        # Counter display (using my helper)
        cv2.putText(image, format_counter(self.counter), 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Angle display
        cv2.putText(image, f'Angle: {int(self.angle)} deg', 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Stage display
        if self.stage:
            cv2.putText(image, f'Stage: {self.stage}', 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Instructions
        cv2.putText(image, 'Press "q" to quit', 
                    (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if not self.ready:
            cv2.putText(image, 'Get into push-up position (arms extended) to start',
                        (10, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # TODO: Show a motivational quote here?
        return image
    
    def run(self):
        print("Push-up Counter Started!")
        print("Position yourself so your left arm is visible to the camera")
        print("Press 'q' to quit")
        # (No full screen, I like to see my desktop too)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            processed_frame = self.process_frame(frame)
            output_frame = self.add_text_overlay(processed_frame)
            cv2.imshow('Push-up Counter', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession ended! Total push-ups: {self.counter}")
        # TODO: Save results to a file?

def main():
    # Main entry point
    try:
        counter = PushUpCounter()
        counter.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 