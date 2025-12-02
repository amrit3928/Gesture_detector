"""
Visualization utilities for hand gesture recognition

This module provides helper functions for visualizing hand landmarks and gesture
information on images. It includes functions to draw landmarks as dots, draw
connections between landmarks, and display gesture names and confidence scores.

Key functionality:
- Draw hand landmarks as circles/dots on images
- Draw connections between landmarks (finger joints, etc.)
- Display gesture names and confidence scores as text overlays
- Convert normalized coordinates to pixel coordinates for drawing
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'src')
)
from src import config


def draw_landmarks(image, landmarks, color=config.LANDMARK_COLOR):
    """
    Draw hand landmarks on an image
    
    Args:
        image: Input image
        landmarks: List of landmark coordinates (normalized 0-1 or pixel coordinates)
        color: Color for landmarks (BGR format)
        
    Returns:
        Image with landmarks drawn
    """
    annotated_image = image.copy()
    height, width = image.shape[:2]
    
    for landmark in landmarks:
        if isinstance(landmark, (list, np.ndarray)) and len(landmark) >= 2:
            if landmark[0] <= 1.0 and landmark[1] <= 1.0:
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
            else:
                x = int(landmark[0])
                y = int(landmark[1])
            
            cv2.circle(annotated_image, (x, y), 5, color, -1)
    
    return annotated_image


def draw_connections(image, landmarks, connections, color=config.CONNECTION_COLOR):
    """
    Draw connections between landmarks
    
    Args:
        image: Input image
        landmarks: List of landmark coordinates (normalized 0-1 or pixel coordinates)
        connections: List of (start_idx, end_idx) tuples
        color: Color for connections (BGR format)
        
    Returns:
        Image with connections drawn
    """
    annotated_image = image.copy()
    height, width = image.shape[:2]
    
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_landmark = landmarks[start_idx]
            end_landmark = landmarks[end_idx]
            
            if isinstance(start_landmark, (list, np.ndarray)) and len(start_landmark) >= 2:
                if start_landmark[0] <= 1.0 and start_landmark[1] <= 1.0:
                    x1 = int(start_landmark[0] * width)
                    y1 = int(start_landmark[1] * height)
                else:
                    x1 = int(start_landmark[0])
                    y1 = int(start_landmark[1])
                
                if end_landmark[0] <= 1.0 and end_landmark[1] <= 1.0:
                    x2 = int(end_landmark[0] * width)
                    y2 = int(end_landmark[1] * height)
                else:
                    x2 = int(end_landmark[0])
                    y2 = int(end_landmark[1])
                
                cv2.line(annotated_image, (x1, y1), (x2, y2), color, 2)
    
    return annotated_image


def display_gesture_info(image, gesture_name, confidence):
    """
    Display gesture information on image
    
    Args:
        image: Input image
        gesture_name: Name of the detected gesture
        confidence: Confidence score (0-1)
        
    Returns:
        Image with gesture info displayed
    """
    annotated_image = image.copy()
    
    text = f"{gesture_name}: {confidence:.2%}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_color = config.TEXT_COLOR
    
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )
    
    x = 10
    y = 30
    padding = 5
    
    cv2.rectangle(
        annotated_image,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        (0, 0, 0),
        -1
    )
    
    cv2.putText(
        annotated_image,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )
    
    return annotated_image

