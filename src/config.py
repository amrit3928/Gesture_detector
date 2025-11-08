"""
Configuration settings for the Hand Gesture Recognition System
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# MediaPipe Hand Detection Settings
HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2

# Model Training Settings
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Video Processing Settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Gesture Recognition Settings
GESTURE_THRESHOLD = 0.7  # Minimum confidence for gesture recognition
NUM_GESTURES = 10  # Number of gestures to recognize

# Visualization Settings
SHOW_LANDMARKS = True
SHOW_CONNECTIONS = True
LANDMARK_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

