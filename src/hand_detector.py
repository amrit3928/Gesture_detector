"""
Hand Detection Module
Extracts hand landmarks from images/video using MediaPipe

This module handles the detection and tracking of hands in images and video frames.
It uses Google's MediaPipe framework to detect hands and extract 21 landmark points
per hand (wrist, thumb, and each finger joint). The landmarks are returned as
normalized coordinates (0-1) which can be converted to pixel coordinates.

Key functionality:
- Detect hands in images/video frames
- Extract 21 landmark coordinates per detected hand
- Draw landmarks and connections for visualization
- Convert normalized coordinates to pixel coordinates
"""

import cv2
import mediapipe as mp
import numpy as np
import config


class HandDetector:
    """
    Hand detector using MediaPipe for landmark extraction
    """
    
    def __init__(self):
        """
        Initialize the hand detector with MediaPipe
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def detect_hands(self, image):
        """
        Detect hands and extract landmarks from an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            landmarks_list: List of landmark coordinates for each detected hand
            annotated_image: Image with landmarks drawn
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        landmarks_list = []
        annotated_image = image.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])
                landmarks_list.append(landmarks)
                
                if config.SHOW_LANDMARKS or config.SHOW_CONNECTIONS:
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
        
        return landmarks_list, annotated_image
    
    def get_landmark_coordinates(self, landmarks, image_shape):
        """
        Convert normalized landmark coordinates to pixel coordinates
        
        Args:
            landmarks: List of normalized landmark coordinates (x, y, z) where x,y are 0-1
            image_shape: (height, width) of the image
            
        Returns:
            Array of pixel coordinates
        """
        height, width = image_shape
        pixel_coords = []
        
        for landmark in landmarks:
            x = int(landmark[0] * width)
            y = int(landmark[1] * height)
            z = landmark[2]
            pixel_coords.append([x, y, z])
        
        return np.array(pixel_coords)
    
    def release(self):
        """
        Release resources
        """
        if hasattr(self, 'hands'):
            self.hands.close()

