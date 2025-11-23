"""
Test script for Hand Detection and Visualization
Tests the hand_detector.py and visualization.py implementations
"""

import cv2
import sys
import os
import numpy as np

# Add src and utils directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

def test_imports():
    """Test if all imports work"""
    print("Testing imports...")
    try:
        from hand_detector import HandDetector
        from visualization import draw_landmarks, draw_connections, display_gesture_info
        print("[OK] All imports successful!")
        return True
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        if "mediapipe" in str(e).lower():
            print("\n[WARNING] MediaPipe not installed or incompatible Python version.")
            print("   MediaPipe requires Python 3.11 or earlier.")
            print("   Current Python version may be too new.")
        return False

def test_hand_detector_init():
    """Test HandDetector initialization"""
    print("\nTesting HandDetector initialization...")
    try:
        from hand_detector import HandDetector
        detector = HandDetector()
        print("[OK] HandDetector initialized successfully!")
        return detector
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        return None

def test_coordinate_conversion(detector):
    """Test coordinate conversion"""
    print("\nTesting coordinate conversion...")
    try:
        # Create sample normalized landmarks
        normalized_landmarks = [
            [0.5, 0.5, 0.0],  # Center
            [0.3, 0.3, 0.0],  # Top-left
            [0.7, 0.7, 0.0],  # Bottom-right
        ]
        
        image_shape = (480, 640)  # height, width
        pixel_coords = detector.get_landmark_coordinates(normalized_landmarks, image_shape)
        
        print(f"[OK] Coordinate conversion successful!")
        print(f"  Normalized: {normalized_landmarks[0]}")
        print(f"  Pixel: {pixel_coords[0]}")
        return True
    except Exception as e:
        print(f"[ERROR] Coordinate conversion failed: {e}")
        return False

def test_visualization_functions():
    """Test visualization functions"""
    print("\nTesting visualization functions...")
    try:
        from visualization import draw_landmarks, draw_connections, display_gesture_info
        
        # Create a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image.fill(50)  # Dark gray background
        
        # Test landmarks (normalized)
        sample_landmarks = [
            [0.5, 0.5, 0.0],  # Center
            [0.4, 0.4, 0.0],  # Top-left
            [0.6, 0.4, 0.0],  # Top-right
        ]
        
        # Test draw_landmarks
        image_with_landmarks = draw_landmarks(test_image.copy(), sample_landmarks)
        print("[OK] draw_landmarks() works")
        
        # Test draw_connections
        connections = [(0, 1), (0, 2)]
        image_with_connections = draw_connections(
            image_with_landmarks, sample_landmarks, connections
        )
        print("[OK] draw_connections() works")
        
        # Test display_gesture_info
        final_image = display_gesture_info(
            image_with_connections, "Test Gesture", 0.95
        )
        print("[OK] display_gesture_info() works")
        
        return True
    except Exception as e:
        print(f"[ERROR] Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_webcam_detection(detector):
    """Test hand detection with webcam"""
    print("\nTesting webcam hand detection...")
    print("Press 'q' to quit, 's' to save a test image")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    hands_detected_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Could not read frame")
                break
            
            # Detect hands
            landmarks_list, annotated_frame = detector.detect_hands(frame)
            
            # Count detections
            if landmarks_list:
                hands_detected_count += 1
                info_text = f"Hands: {len(landmarks_list)} | Landmarks: {len(landmarks_list[0]) if landmarks_list else 0}"
                cv2.putText(annotated_frame, info_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Hand Detection Test', annotated_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('test_hand_detection.png', annotated_frame)
                print(f"  Saved test image: test_hand_detection.png")
            
            frame_count += 1
            
            # Auto-quit after 300 frames (10 seconds at 30fps)
            if frame_count > 300:
                print("  Auto-quitting after 10 seconds...")
                break
        
        print(f"[OK] Webcam test completed!")
        print(f"  Frames processed: {frame_count}")
        print(f"  Frames with hands detected: {hands_detected_count}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Webcam test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()

def main():
    """Run all tests"""
    print("=" * 60)
    print("Hand Detection and Visualization Test Suite")
    print("=" * 60)
    
    # Test 1: Imports
    if not test_imports():
        print("\n[WARNING] Cannot continue - imports failed.")
        print("   Please install MediaPipe: pip install mediapipe")
        print("   Note: Requires Python 3.11 or earlier")
        print("\n   Testing visualization functions without MediaPipe...")
        test_visualization_functions()
        return
    
    # Test 2: HandDetector initialization
    detector = test_hand_detector_init()
    if not detector:
        print("\n[WARNING] Cannot continue - HandDetector initialization failed.")
        return
    
    # Test 3: Coordinate conversion
    test_coordinate_conversion(detector)
    
    # Test 4: Visualization functions
    test_visualization_functions()
    
    # Test 5: Webcam (optional)
    print("\n" + "=" * 60)
    user_input = input("Run webcam test? (y/n): ").strip().lower()
    if user_input == 'y':
        test_webcam_detection(detector)
    else:
        print("Skipping webcam test")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()

