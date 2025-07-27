# Quick Setup Guide

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Installation
```bash
python test_installation.py
```

### 3. Run the Application

**Option A: Basic Version (Left Arm Only)**
```bash
python push_up_counter.py
```

**Option B: Enhanced Version (Both Arms + Form Feedback)**
```bash
python enhanced_push_up_counter.py
```

**Option C: Use Scripts**
- Windows: Double-click `run_push_up_counter.bat`
- Mac/Linux: Run `./run_push_up_counter.sh`

## 📋 System Requirements

- **Python**: 3.7 or higher
- **Camera**: Webcam or built-in camera
- **RAM**: At least 4GB recommended
- **OS**: Windows, macOS, or Linux

## 🎯 How to Use

1. **Position yourself** so your arms are visible to the camera
2. **Start doing push-ups** - the system will automatically detect and count
3. **Watch the display** for real-time feedback:
   - Push-up counter
   - Current angle measurements
   - Stage (up/down)
   - Form warnings (enhanced version)
4. **Press 'q'** to quit the application

## 🔧 Troubleshooting

### Camera Issues
- Make sure no other application is using the camera
- Try changing camera index in code: `cv2.VideoCapture(1)`

### Detection Issues
- Ensure good lighting
- Position yourself 2-3 feet from camera
- Wear clothing that doesn't obscure your arms

### Installation Issues
- Update pip: `pip install --upgrade pip`
- On Windows, install Visual C++ build tools if needed

## 📊 Features Comparison

| Feature | Basic Version | Enhanced Version |
|---------|---------------|------------------|
| Single arm detection | ✅ Left arm only | ✅ Both arms |
| Angle display | ✅ | ✅ |
| Counter | ✅ | ✅ |
| Form feedback | ❌ | ✅ |
| Rep timing | ❌ | ✅ |
| Reset counter | ❌ | ✅ (press 'r') |
| Session summary | ❌ | ✅ |

## 🎮 Controls

- **'q'**: Quit application
- **'r'**: Reset counter (enhanced version only)

## 📈 Understanding the Display

- **Push-ups**: Total count of completed repetitions
- **Left/Right Angle**: Individual arm angles in degrees
- **Avg Angle**: Average of both arm angles
- **Stage**: Current position (UP/DOWN)
- **Back Angle**: Form check for back straightness
- **Last Rep**: Time of last completed repetition
- **Avg Rep**: Average time per repetition

## 🏆 Tips for Best Results

1. **Good lighting** - Natural light or bright room lighting
2. **Clear background** - Avoid busy backgrounds
3. **Proper positioning** - Full upper body visible
4. **Consistent form** - Maintain proper push-up technique
5. **Steady camera** - Avoid camera movement during use

## 📞 Support

If you encounter issues:
1. Run `python test_installation.py` to check dependencies
2. Check the README.md for detailed documentation
3. Ensure your camera is working in other applications
4. Try the basic version first, then upgrade to enhanced 