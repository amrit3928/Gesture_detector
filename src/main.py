"""
Main Entry Point for Hand Gesture Recognition System
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_processor import VideoProcessor
from gesture_classifier import GestureClassifier
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
        print("Training mode not yet implemented.")
        print("This will be implemented in Week 2.")
        # TODO: Load training data
        # TODO: Preprocess data
        # TODO: Split into train/val/test sets
        # TODO: Initialize GestureClassifier
        # TODO: Build and train model
        # TODO: Save trained model


if __name__ == "__main__":
    main()

