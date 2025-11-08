# File Descriptions

This document provides a comprehensive overview of all files in the Hand Gesture Recognition System project.

## Source Code Files (`src/`)

### `src/main.py`
**Main Entry Point for Hand Gesture Recognition System**

This is the command-line interface (CLI) entry point for the hand gesture recognition system. It provides three main modes of operation:
1. Live mode: Process real-time video from webcam
2. Video mode: Process pre-recorded video files
3. Train mode: Train the gesture classification model

**Usage examples:**
- `python src/main.py --mode live`
- `python src/main.py --mode video --input video.mp4`
- `python src/main.py --mode train --data data/processed/`

---

### `src/config.py`
**Configuration settings for the Hand Gesture Recognition System**

This file contains all configuration parameters and settings used throughout the project. It centralizes configuration to make it easy to adjust parameters without modifying multiple files. It also automatically creates necessary directories for data storage.

**Key configurations:**
- MediaPipe hand detection settings (confidence thresholds, max hands)
- Model training parameters (batch size, epochs, learning rate)
- Video processing settings (resolution, FPS)
- Gesture recognition thresholds
- Visualization colors and display options

---

### `src/hand_detector.py`
**Hand Detection Module - Extracts hand landmarks from images/video using MediaPipe**

This module handles the detection and tracking of hands in images and video frames. It uses Google's MediaPipe framework to detect hands and extract 21 landmark points per hand (wrist, thumb, and each finger joint). The landmarks are returned as normalized coordinates (0-1) which can be converted to pixel coordinates.

**Key functionality:**
- Detect hands in images/video frames
- Extract 21 landmark coordinates per detected hand
- Draw landmarks and connections for visualization
- Convert normalized coordinates to pixel coordinates

---

### `src/gesture_classifier.py`
**Gesture Classification Module - Trains and uses a model to classify hand gestures from landmarks**

This module implements a neural network model for classifying hand gestures based on the 21 landmark points extracted by the HandDetector. It uses TensorFlow/Keras to build, train, and use a deep learning model for gesture recognition.

**Key functionality:**
- Build neural network architecture for gesture classification
- Train model on collected landmark data
- Predict gestures from new landmark data
- Save and load trained models
- Manage gesture names for display purposes

---

### `src/video_processor.py`
**Video Processing Module - Handles live webcam and pre-recorded video processing**

This module orchestrates the complete hand gesture recognition pipeline for video streams. It combines the HandDetector (for landmark extraction) and GestureClassifier (for gesture recognition) to process video frames in real-time or from files.

**Key functionality:**
- Process live video from webcam with real-time gesture recognition
- Process pre-recorded video files
- Process individual frames for hand detection and gesture classification
- Display results with annotations (landmarks, gesture names, confidence)
- Save processed videos with annotations

---

### `src/__init__.py`
**Package initialization file**

This package contains the main source code for the hand gesture recognition system. It includes modules for hand detection, gesture classification, video processing, and configuration management.

---

## Utility Files (`utils/`)

### `utils/visualization.py`
**Visualization utilities for hand gesture recognition**

This module provides helper functions for visualizing hand landmarks and gesture information on images. It includes functions to draw landmarks as dots, draw connections between landmarks, and display gesture names and confidence scores.

**Key functionality:**
- Draw hand landmarks as circles/dots on images
- Draw connections between landmarks (finger joints, etc.)
- Display gesture names and confidence scores as text overlays
- Convert normalized coordinates to pixel coordinates for drawing

---

### `utils/data_utils.py`
**Data processing utilities for training data preparation**

This module provides utilities for loading, preprocessing, and augmenting training data for the gesture classifier. It handles reading landmark data from files, normalizing landmarks, applying data augmentation techniques, and saving/loading processed data.

**Key functionality:**
- Load landmark data and labels from directories
- Preprocess landmarks (normalization, scaling, etc.)
- Augment training data (rotation, scaling, noise, etc.)
- Save and load processed data to/from files

---

### `utils/__init__.py`
**Package initialization file**

This package contains utility modules for data processing and visualization. It provides helper functions for loading/preprocessing training data and visualizing hand landmarks and gesture information.

---

## Test Files (`tests/`)

### `tests/test_hand_detector.py`
**Tests for HandDetector class**

This file contains unit tests for the HandDetector module. It tests the initialization, hand detection functionality, and coordinate conversion methods. These tests ensure the hand detection module works correctly before integration with other components.

**Test coverage:**
- HandDetector initialization
- Hand detection on sample images
- Landmark coordinate conversion (normalized to pixel)

---

### `tests/__init__.py`
**Package initialization file**

This package contains unit tests for the hand gesture recognition system. Tests ensure that each module works correctly before integration.

---

## Documentation Files

### `README.md`
**Project documentation and overview**

Contains project description, features, installation instructions, usage examples, and milestone information.

---

### `SETUP.md`
**Setup instructions**

Provides step-by-step instructions for setting up the development environment, installing dependencies, and running the project.

---

### `GIT_SETUP.md`
**Git repository setup instructions**

Contains instructions for connecting the local project to the GitHub repository.

---

### `FILE_DESCRIPTIONS.md` (this file)
**File descriptions overview**

Provides comprehensive descriptions of all files in the project.

---

## Configuration Files

### `requirements.txt`
**Python package dependencies**

Lists all Python packages required for the project, including:
- Computer vision libraries (OpenCV, MediaPipe)
- Machine learning frameworks (TensorFlow)
- Data processing libraries (NumPy, Pandas)
- Development tools (pytest, black, flake8)

---

### `.gitignore`
**Git ignore rules**

Specifies files and directories that should not be tracked by Git, including:
- Python cache files (`__pycache__/`)
- Virtual environments (`venv/`)
- IDE files (`.vscode/`, `.idea/`)
- Large data files and model files
- Video files

---

## Project Structure Summary

```
CSE_Final/
├── src/                    # Main source code
│   ├── main.py            # CLI entry point
│   ├── config.py          # Configuration settings
│   ├── hand_detector.py   # Hand detection module
│   ├── gesture_classifier.py  # Gesture classification
│   └── video_processor.py  # Video processing
├── utils/                  # Utility functions
│   ├── visualization.py   # Visualization helpers
│   └── data_utils.py      # Data processing
├── tests/                  # Unit tests
│   └── test_hand_detector.py
├── data/                   # Data directories (auto-created)
│   ├── raw/               # Raw video/images
│   ├── processed/         # Processed training data
│   └── models/            # Saved models
└── Documentation files     # README, SETUP, etc.
```

