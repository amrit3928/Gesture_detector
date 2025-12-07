"""
Convert HaGRID Images to MediaPipe Landmarks

This script processes HaGRID dataset images and extracts MediaPipe hand landmarks,
saving them in the format expected by the training pipeline.

Usage:
    python convert_hagrid_to_landmarks.py --input path/to/hagrid/images/ --output data/processed/
"""

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import config


def get_gesture_id_from_folder(folder_name):
    """
    Map HaGRID folder names to gesture IDs
    
    Maps HaGRID gesture names to our 10 gesture IDs (0-9)
    Common gestures: one, peace, fist, call, ok, like, point, rock, three, four
    """
    gesture_mapping = {
        'one': 0,            # One finger
        'peace': 1,          # Peace sign
        'fist': 2,           # Fist
        'call': 3,           # Call gesture
        'ok': 4,             # OK sign
        'like': 5,           # Thumbs up
        'point': 6,          # Point
        'rock': 7,           # Rock
        'three': 8,          # Three Gun (includes 'three' and 'three2')
        'three2': 8,         # Three Gun variant
        'three_gun': 8,      # Three Gun (explicit)
        'four': 9,           # Four fingers
    }
    
    folder_lower = folder_name.lower()
    
    if 'three2' in folder_lower:
        return 8
    
    for key, value in gesture_mapping.items():
        if key in folder_lower:
            return value
    
    return None


def process_image(image_path, hand_detector):
    """
    Process a single image and extract landmarks
    
    Returns:
        landmarks array (21, 3) or None if no hand detected
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    landmarks_list, _ = hand_detector.detect_hands(image)
    
    if len(landmarks_list) == 0:
        return None
    
    landmarks = np.array(landmarks_list[0])
    
    if landmarks.shape != (21, 3):
        return None
    
    return landmarks


def convert_hagrid_dataset(input_dir, output_dir):
    """
    Convert HaGRID images to MediaPipe landmarks
    
    Args:
        input_dir: Directory containing HaGRID images (organized by gesture folders)
        output_dir: Output directory for landmark files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing MediaPipe for static image processing...")
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE
    )
    
    class StaticHandDetector:
        def detect_hands(self, image):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)
            landmarks_list = []
            annotated_image = image.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                    landmarks_list.append(landmarks)
            return landmarks_list, annotated_image
    
    hand_detector = StaticHandDetector()
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    
    total_processed = 0
    total_skipped = 0
    
    print(f"Processing images from: {input_dir}")
    print(f"Saving landmarks to: {output_dir}")
    print("-" * 60)
    
    if os.path.isdir(input_dir):
        gesture_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
        
        if len(gesture_folders) == 0:
            print("No gesture folders found. Processing images directly...")
            image_files = [f for f in os.listdir(input_dir) 
                          if f.lower().endswith(supported_formats)]
            
            gesture_id = 0
            sample_count = {}
            
            for image_file in tqdm(image_files, desc="Processing images"):
                image_path = os.path.join(input_dir, image_file)
                landmarks = process_image(image_path, hand_detector)
                
                if landmarks is not None:
                    if gesture_id not in sample_count:
                        sample_count[gesture_id] = 0
                    sample_count[gesture_id] += 1
                    
                    output_filename = f"gesture_{gesture_id}_sample_{sample_count[gesture_id]}.npy"
                    output_path = os.path.join(output_dir, output_filename)
                    np.save(output_path, landmarks)
                    total_processed += 1
                else:
                    total_skipped += 1
        else:
            sample_count_per_gesture = {}
            
            for gesture_folder in gesture_folders:
                gesture_id = get_gesture_id_from_folder(gesture_folder)
                
                if gesture_id is None:
                    print(f"Warning: Could not map folder '{gesture_folder}' to gesture ID. Skipping...")
                    continue
                
                if gesture_id not in sample_count_per_gesture:
                    sample_count_per_gesture[gesture_id] = 0
                
                gesture_path = os.path.join(input_dir, gesture_folder)
                image_files = [f for f in os.listdir(gesture_path) 
                              if f.lower().endswith(supported_formats)]
                
                print(f"\nProcessing gesture {gesture_id} ({gesture_folder}): {len(image_files)} images")
                
                folder_sample_count = 0
                
                for image_file in tqdm(image_files, desc=f"Gesture {gesture_id}"):
                    image_path = os.path.join(gesture_path, image_file)
                    landmarks = process_image(image_path, hand_detector)
                    
                    if landmarks is not None:
                        sample_count_per_gesture[gesture_id] += 1
                        folder_sample_count += 1
                        output_filename = f"gesture_{gesture_id}_sample_{sample_count_per_gesture[gesture_id]}.npy"
                        output_path = os.path.join(output_dir, output_filename)
                        np.save(output_path, landmarks)
                        total_processed += 1
                    else:
                        total_skipped += 1
                
                print(f"  -> Saved {folder_sample_count} samples for gesture {gesture_id} (total: {sample_count_per_gesture[gesture_id]})")
    
    hands.close()
    
    print("\n" + "=" * 60)
    print(f"Conversion complete!")
    print(f"  Processed: {total_processed} images")
    print(f"  Skipped (no hand detected): {total_skipped} images")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert HaGRID images to MediaPipe landmarks"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing HaGRID images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/',
        help='Output directory for landmark files (default: data/processed/)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        return
    
    convert_hagrid_dataset(args.input, args.output)


if __name__ == "__main__":
    main()

