# Push-Up Counter Using Pose Detection

A real-time push-up counting system that uses computer vision and pose estimation to track and count push-up repetitions.

## Features

-  **Real-time pose detection** using MediaPipe
-  **Live push-up counter** with visual feedback
-  **Angle calculation** for precise movement detection
-  **Smart detection** of up/down positions
-  **Webcam integration** with OpenCV
-  **Visual overlay** showing counter, angle, and stage

## Installation

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

## Usage

1. **Run the application:**
   ```bash
   python push_up_counter.py
   ```

2. **Position yourself:**
   - Stand or position yourself so your **left arm** is clearly visible to the camera
   - Make sure you're in a well-lit environment
   - The camera should capture your full upper body

3. **Start doing push-ups:**
   - The system will detect your pose and track your elbow angle
   - When your arm is straight (>160°), it shows "up" stage
   - When you lower yourself (<90°), it shows "down" stage
   - Each complete repetition (up → down → up) increments the counter

4. **Quit the application:**
   - Press `q` to quit the program

## How It Works

### Pose Detection
- Uses MediaPipe's pose estimation to detect 33 body landmarks
- Focuses on left arm landmarks: shoulder, elbow, and wrist

### Angle Calculation
- Calculates the angle at the elbow joint using vector mathematics
- Formula: `angle = arccos(dot(ba, bc) / (|ba| * |bc|))`

### Push-up Detection Logic
- **Up position**: Angle > 160° (arm extended)
- **Down position**: Angle < 90° (arm bent)
- **Counter increment**: Only when transitioning from "up" to "down" stage

### Visual Feedback
- Real-time pose skeleton overlay
- Live counter display
- Current angle measurement
- Current stage (up/down)
- Instructions for quitting

## Technical Details

### Dependencies
- **OpenCV**: Video capture and image processing
- **MediaPipe**: Pose estimation and landmark detection
- **NumPy**: Mathematical calculations for angle computation

### Key Components
1. **Video Capture**: `cv2.VideoCapture(0)` for webcam input
2. **Pose Detection**: MediaPipe Pose model for landmark detection
3. **Angle Calculation**: Vector math to compute elbow angle
4. **State Machine**: Tracks up/down states for accurate counting
5. **Visual Overlay**: Real-time text and skeleton drawing

## Troubleshooting

### Common Issues

1. **Camera not found:**
   - Make sure your webcam is connected and not in use by another application
   - Try changing the camera index in the code (e.g., `cv2.VideoCapture(1)`)

2. **Poor detection:**
   - Ensure good lighting conditions
   - Make sure your left arm is clearly visible
   - Try adjusting your position relative to the camera

3. **Installation errors:**
   - Make sure you have Python 3.7+ installed
   - Try upgrading pip: `pip install --upgrade pip`
   - Install Visual C++ build tools if on Windows

### Performance Tips

- Close other applications using the camera
- Ensure adequate lighting
- Position yourself 2-3 feet from the camera
- Wear clothing that doesn't obscure your arms

## Future Enhancements

- [ ] Support for both arms (left + right)
- [ ] Form scoring and feedback
- [ ] Rep timing and speed analysis
- [ ] Data export to CSV
- [ ] GUI version with Streamlit
- [ ] Multiple exercise types
- [ ] Calibration mode for different body types

## License

This project is created by Krish Israni from Scrap.

## Contributing


Feel free to submit issues, feature requests, or pull requests to improve this project! 
