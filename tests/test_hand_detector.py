"""
Tests for HandDetector class
"""

import pytest
import numpy as np
import cv2
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hand_detector import HandDetector


def test_hand_detector_initialization():
    """Test that HandDetector initializes correctly"""
    detector = HandDetector()
    assert detector is not None
    detector.release()


def test_detect_hands_with_image():
    """Test hand detection on a sample image"""
    detector = HandDetector()
    
    # Create a dummy image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    landmarks, annotated_image = detector.detect_hands(test_image)
    
    assert isinstance(landmarks, list)
    assert annotated_image.shape == test_image.shape
    
    detector.release()


def test_get_landmark_coordinates():
    """Test landmark coordinate conversion"""
    detector = HandDetector()
    
    # Create dummy normalized landmarks
    landmarks = [[0.5, 0.5, 0.0], [0.6, 0.6, 0.0]]
    image_shape = (480, 640)
    
    pixel_coords = detector.get_landmark_coordinates(landmarks, image_shape)
    
    assert pixel_coords.shape[0] == len(landmarks)
    assert pixel_coords[0][0] == 320  # 0.5 * 640
    assert pixel_coords[0][1] == 240  # 0.5 * 480
    
    detector.release()


if __name__ == "__main__":
    pytest.main([__file__])

