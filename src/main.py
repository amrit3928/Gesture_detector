"""
Main Entry Point for Hand Gesture Recognition System

This is the command-line interface (CLI) entry point for the hand gesture recognition
system. It provides three main modes of operation:
1. Live mode: Process real-time video from webcam
2. Video mode: Process pre-recorded video files
3. Train mode: Train the gesture classification model

Usage examples:
    python src/main.py --mode live
    python src/main.py --mode video --input video.mp4
    python src/main.py --mode train --data data/processed/
"""

import argparse
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add parent directory to path for utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_processor import VideoProcessor
from gesture_classifier import GestureClassifier
from utils.data_utils import load_landmarks_from_directory, preprocess_landmarks, augment_data
import config


def main():
    """
    Main function to run the hand gesture recognition system
    """
    parser = argparse.ArgumentParser(
        description="Hand Gesture Recognition System"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['live', 'video', 'train'],
        default='live',
        help='Mode: live (webcam), video (pre-recorded), or train (model training)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input video file path (for video mode)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output video file path (for video mode)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to training data directory (for train mode)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'live':
        # Live webcam processing
        processor = VideoProcessor()
        if args.model:
            processor.load_model(args.model)
        processor.process_live_video()
        
    elif args.mode == 'video':
        # Pre-recorded video processing
        if not args.input:
            print("Error: --input is required for video mode")
            return
        
        processor = VideoProcessor()
        if args.model:
            processor.load_model(args.model)
        processor.process_video_file(args.input, args.output)
        
    elif args.mode == 'train':
        # Model training
        if not args.data:
            print("Error: --data is required for train mode")
            print("Usage: python src/main.py --mode train --data data/processed/")
            return

        if not os.path.exists(args.data):
            print(f"Error: Data directory not found: {args.data}")
            return

        print("=" * 60)
        print("Hand Gesture Recognition - Training Mode")
        print("=" * 60)

        # Load training data
        print(f"\n[1/6] Loading training data from: {args.data}")
        X, y = load_landmarks_from_directory(args.data)

        if len(X) == 0:
            print("Error: No training data found in the specified directory")
            return

        print(f"Loaded {len(X)} samples")

        # Preprocess data
        print("\n[2/6] Preprocessing landmarks...")
        X_processed = preprocess_landmarks(X)

        # Augment data
        print("\n[3/6] Augmenting training data...")
        X_augmented, y_augmented = augment_data(X_processed, y)
        print(f"Dataset size after augmentation: {len(X_augmented)} samples")

        # Split into train/val/test sets
        print("\n[4/6] Splitting data into train/validation/test sets...")
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_augmented, y_augmented,
            test_size=config.TEST_SPLIT,
            random_state=42,
            stratify=y_augmented
        )

        # Second split: separate train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=config.VALIDATION_SPLIT / (1 - config.TEST_SPLIT),
            random_state=42,
            stratify=y_temp
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        # Convert labels to one-hot encoding
        # num_classes = len(np.unique(y_augmented)) # Alternative way to get number of classes
        num_classes = config.NUM_GESTURES
        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_val_cat = to_categorical(y_val, num_classes=num_classes)
        y_test_cat = to_categorical(y_test, num_classes=num_classes)

        # Initialize GestureClassifier
        print(f"\n[5/6] Initializing GestureClassifier with {num_classes} gesture classes...")
        classifier = GestureClassifier(num_gestures=num_classes)
        classifier.build_model(input_shape=X_train.shape[1:])

        print("\nModel architecture:")
        classifier.model.summary()

        # Train model
        print(f"\n[6/6] Training model for {config.EPOCHS} epochs...")
        print("-" * 60)
        history = classifier.train(
            X_train, y_train_cat,
            X_val, y_val_cat,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE
        )

        # Evaluate on test set
        print("\n" + "=" * 60)
        print("Evaluating model on test set...")
        test_loss, test_accuracy = classifier.model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Save trained model
        model_path = args.model if args.model else os.path.join(config.MODELS_DIR, "gesture_model.h5")
        print(f"\nSaving trained model to: {model_path}")
        classifier.save_model(model_path)

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    main()

