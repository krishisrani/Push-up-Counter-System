#!/usr/bin/env python3
"""
Test script to verify that all dependencies are properly installed
"""

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import cv2
        print("✅ OpenCV imported successfully")
        print(f"   Version: {cv2.__version__}")
    except ImportError as e:
        print("❌ OpenCV import failed:", e)
        return False
    
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
        print(f"   Version: {mp.__version__}")
    except ImportError as e:
        print("❌ MediaPipe import failed:", e)
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        print(f"   Version: {np.__version__}")
    except ImportError as e:
        print("❌ NumPy import failed:", e)
        return False
    
    return True

def test_camera():
    """Test if camera is accessible"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera is accessible")
            ret, frame = cap.read()
            if ret:
                print(f"   Frame size: {frame.shape}")
            cap.release()
            return True
        else:
            print("❌ Camera is not accessible")
            return False
    except Exception as e:
        print("❌ Camera test failed:", e)
        return False

def test_mediapipe_pose():
    """Test if MediaPipe pose detection works"""
    try:
        import mediapipe as mp
        import numpy as np
        
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Initialize pose detection
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Process the dummy image
        results = pose.process(dummy_image)
        
        print("✅ MediaPipe pose detection initialized successfully")
        pose.close()
        return True
    except Exception as e:
        print("❌ MediaPipe pose test failed:", e)
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Push-up Counter Dependencies")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    print()
    
    # Test camera
    if not test_camera():
        all_tests_passed = False
    
    print()
    
    # Test MediaPipe pose
    if not test_mediapipe_pose():
        all_tests_passed = False
    
    print()
    print("=" * 40)
    
    if all_tests_passed:
        print("🎉 All tests passed! You're ready to run the push-up counter.")
        print("\nTo start the basic version:")
        print("   python push_up_counter.py")
        print("\nTo start the enhanced version:")
        print("   python enhanced_push_up_counter.py")
    else:
        print("❌ Some tests failed. Please check the installation.")
        print("\nTry running:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 