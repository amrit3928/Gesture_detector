"""
Video Processing Module
Handles live webcam and pre-recorded video processing

This module orchestrates the complete hand gesture recognition pipeline for video
streams. It combines the HandDetector (for landmark extraction) and GestureClassifier
(for gesture recognition) to process video frames in real-time or from files.

Key functionality:
- Process live video from webcam with real-time gesture recognition
- Process pre-recorded video files
- Process individual frames for hand detection and gesture classification
- Display results with annotations (landmarks, gesture names, confidence)
- Save processed videos with annotations
"""

import cv2
import numpy as np
from typing import Optional, Callable
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
# import logging
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier
from utils.data_utils import preprocess_landmarks


class VideoProcessor:
    """
    Process video streams (live or pre-recorded) for hand gesture recognition
    """
    
    def __init__(self):
        """
        Initialize video processor with hand detector and gesture classifier
        """
        self.hand_detector = HandDetector()
        self.gesture_classifier = GestureClassifier()
        self.cap = None
        
    def process_live_video(self, callback: Optional[Callable] = None):
        """
        Process live video from webcam
        
        Args:
            callback: Optional callback function to process each frame
        """

        # open webcam
        self.cap = cv2.VideoCapture(0)

        # set cam properties
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FPS, config.FPS)

        # loop through frames
        while self.cap.isOpened():
            grabbed, frame = self.cap.read()

            if not grabbed:
                break

            # process each frame
            annotated_frame = self.process_frame(frame)

            if callback:
                callback(annotated_frame)

            # display frame
            cv2.imshow('video frame', annotated_frame)

            # 'q' key to quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # cleanup
        self.cleanup()

    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None):
        """
        Process a pre-recorded video file
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save processed video
        """

        # open video file
        self.cap = cv2.VideoCapture(video_path)

        # get video properties
        vid_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # create writer if output path exists
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc= fourcc, fps= vid_fps, frameSize= (vid_width, vid_height))

        # loop through frames
        while self.cap.isOpened():
            grabbed, frame = self.cap.read()
            
            if not grabbed:
                break

            # process each frame
            annotated_frame = self.process_frame(frame)

            if writer: 
                writer.write(annotated_frame)

            # display frame
            cv2.imshow('Hand Gesture', annotated_frame)

            # handle 'q' key to quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # cleanup resources
        self.cleanup()

        if writer:
            writer.release()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for hand gesture recognition
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Processed frame with annotations
        """
        
        annotated_frame = frame.copy()

        h, w, _ = frame.shape

        list_landmarks, annotated_frame = self.hand_detector.detect_hands(annotated_frame)

        for landmarks in list_landmarks:
            coordinates = self.hand_detector.get_landmark_coordinates(landmarks, (h, w))

            if self.gesture_classifier.model is None:
                return annotated_frame
            
            landmarks_normalized = np.array(landmarks).reshape(21, 3)
            landmarks_preprocessed = preprocess_landmarks(landmarks_normalized)
            
            gesture, confidence = self.gesture_classifier.predict(landmarks_preprocessed[0])

            gesture_name = self.gesture_classifier.get_gesture_name(gesture)

            cv2.putText(
                annotated_frame, 
                f'{gesture_name} ({confidence:.2f})',
                (int(coordinates[0][0]) + 15, int(coordinates[0][1]) - 15),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                config.TEXT_COLOR,
                2
            )

        return annotated_frame
    
    def load_model(self, model_path: str):
        """
        Load a trained gesture classification model
        
        Args:
            model_path: Path to the model file
        """   
        try:
            self.gesture_classifier.load_model(model_path)
        except Exception as E:
            print(f"Error loading model: {E}")   
    
    def cleanup(self):
        """
        Release resources
        """
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()

        if self.hand_detector is not None:
            self.hand_detector.release()
            self.hand_detector = None

        #print("Resources have been cleaned up.")
