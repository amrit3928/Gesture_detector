"""
Visualization utilities for hand gesture recognition
"""

import cv2
import numpy as np
from typing import List, Tuple
import config


def draw_landmarks(image: np.ndarray, landmarks: List, 
                  color: Tuple[int, int, int] = config.LANDMARK_COLOR) -> np.ndarray:
    """
    Draw hand landmarks on an image
    
    Args:
        image: Input image
        landmarks: List of landmark coordinates
        color: Color for landmarks (BGR format)
        
    Returns:
        Image with landmarks drawn
    """
    # TODO: Draw circles/dots for each landmark
    # TODO: Convert normalized coordinates to pixel coordinates
    # TODO: Return annotated image
    
    annotated_image = image.copy()
    return annotated_image


def draw_connections(image: np.ndarray, landmarks: List,
                   connections: List[Tuple[int, int]],
                   color: Tuple[int, int, int] = config.CONNECTION_COLOR) -> np.ndarray:
    """
    Draw connections between landmarks
    
    Args:
        image: Input image
        landmarks: List of landmark coordinates
        connections: List of (start_idx, end_idx) tuples
        color: Color for connections (BGR format)
        
    Returns:
        Image with connections drawn
    """
    # TODO: Draw lines between connected landmarks
    # TODO: Convert normalized coordinates to pixel coordinates
    # TODO: Return annotated image
    
    annotated_image = image.copy()
    return annotated_image


def display_gesture_info(image: np.ndarray, gesture_name: str, 
                         confidence: float) -> np.ndarray:
    """
    Display gesture information on image
    
    Args:
        image: Input image
        gesture_name: Name of the detected gesture
        confidence: Confidence score
        
    Returns:
        Image with gesture info displayed
    """
    # TODO: Add text to image showing gesture name and confidence
    # TODO: Use cv2.putText
    # TODO: Return annotated image
    
    annotated_image = image.copy()
    return annotated_image

