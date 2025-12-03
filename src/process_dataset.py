"""
Dataset Processing Script

This script processes a dataset of images organized in folders (one folder per gesture class)
and extracts hand landmarks using MediaPipe. The extracted landmarks are saved as .npy files
ready for training.

Usage:
    python src/process_dataset.py --dataset_path path/to/dataset --output_path data/processed
"""

import os
import cv2
import numpy as np
import argparse
import sys
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add parent directory to path for utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_detector import HandDetector
from utils.data_utils import save_landmarks

# Define mapping from folder names to gesture IDs
GESTURE_MAP = {
    'train_val_four': 0,
    'train_val_like': 1,
    'train_val_mute': 2,
    'train_val_ok': 3,
    'train_val_one': 4,
    'train_val_palm': 5,
    'train_val_peace': 6,
    'train_val_three': 7,
    'train_val_two_up': 8,
    # Add more mappings if needed
}

def process_dataset(dataset_path, output_path):
    """
    Process the dataset and save landmarks
    
    Args:
        dataset_path: Path to the root dataset directory containing gesture folders
        output_path: Path to save the processed .npy files
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    detector = HandDetector()
    
    total_samples = 0
    processed_samples = 0
    
    print(f"Processing dataset from: {dataset_path}")
    print(f"Saving results to: {output_path}")
    
    # Iterate through each gesture folder
    for folder_name, gesture_id in GESTURE_MAP.items():
        folder_path = os.path.join(dataset_path, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
            
        print(f"\nProcessing gesture: {folder_name} (ID: {gesture_id})")
        
        # Get list of image files
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in tqdm(image_files, desc=f"Processing {folder_name}"):
            img_path = os.path.join(folder_path, img_file)
            
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image: {img_path}")
                continue
                
            # Detect hands
            try:
                landmarks_list, _ = detector.detect_hands(image)
                
                # If hands detected, save landmarks
                if landmarks_list:
                    for i, landmarks in enumerate(landmarks_list):
                        # Convert list to numpy array
                        landmarks_np = np.array(landmarks)
                        
                        # Create output filename
                        # Format: gesture_{id}_sample_{count}.npy
                        output_filename = f"gesture_{gesture_id}_sample_{total_samples}.npy"
                        output_filepath = os.path.join(output_path, output_filename)
                        
                        # Save landmarks
                        # We save the raw landmarks here; preprocessing happens during training loading
                        save_landmarks(landmarks_np, gesture_id, output_filepath)
                        
                        total_samples += 1
                        processed_samples += 1
                        
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue

    print("\n" + "="*50)
    print(f"Processing complete!")
    print(f"Total samples saved: {processed_samples}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image dataset for hand gesture recognition")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_path', type=str, default='data/processed', help='Path to save processed data')
    
    args = parser.parse_args()
    
    process_dataset(args.dataset_path, args.output_path)
