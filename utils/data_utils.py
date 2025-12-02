"""
Data processing utilities for training data preparation

This module provides utilities for loading, preprocessing, and augmenting training
data for the gesture classifier. It handles reading landmark data from files,
normalizing landmarks, applying data augmentation techniques, and saving/loading
processed data.

Key functionality:
- Load landmark data and labels from directories
- Preprocess landmarks (normalization, scaling, etc.)
- Augment training data (rotation, scaling, noise, etc.)
- Save and load processed data to/from files
"""

import numpy as np
import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from src import config


def load_landmarks_from_directory(directory):
    """
    Load landmark data from a directory
    
    Args:
        directory: Directory containing landmark data files
        
    Returns:
        Tuple of (landmarks, labels)
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    landmarks_list = []
    labels_list = []
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if os.path.isfile(filepath):
            try:
                if filename.endswith('.npy'):
                    landmarks = np.load(filepath)
                    label = _extract_label_from_filename(filename)
                    landmarks_list.append(landmarks)
                    labels_list.append(label)
                    
                elif filename.endswith('.npz'):
                    data = np.load(filepath)
                    if 'landmarks' in data and 'label' in data:
                        landmarks_list.append(data['landmarks'])
                        labels_list.append(int(data['label']))
                    elif 'X' in data and 'y' in data:
                        landmarks_list.extend(data['X'])
                        labels_list.extend(data['y'])
                        
                elif filename.endswith('.csv'):
                    landmarks, label = _load_from_csv(filepath)
                    if landmarks is not None:
                        landmarks_list.append(landmarks)
                        labels_list.append(label)
                        
                elif filename.endswith('.json'):
                    landmarks, label = _load_from_json(filepath)
                    if landmarks is not None:
                        landmarks_list.append(landmarks)
                        labels_list.append(label)
                        
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                continue
    
    if len(landmarks_list) == 0:
        return np.array([]), np.array([])
    
    landmarks_array = np.array(landmarks_list)
    labels_array = np.array(labels_list)
    
    return landmarks_array, labels_array


def _extract_label_from_filename(filename):
    """Extract label from filename (e.g., 'gesture_0_sample_1.npy' -> 0)"""
    parts = filename.replace('.npy', '').replace('.npz', '').replace('.csv', '').replace('.json', '').split('_')
    for part in parts:
        if part.isdigit():
            return int(part)
    return 0


def _load_from_csv(filepath):
    """Load landmarks from CSV file"""
    try:
        data = np.genfromtxt(filepath, delimiter=',')
        if data.shape[1] == 64:
            landmarks = data[:63].reshape(21, 3)
            label = int(data[63])
            return landmarks, label
    except:
        pass
    return None, None


def _load_from_json(filepath):
    """Load landmarks from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if 'landmarks' in data and 'label' in data:
                landmarks = np.array(data['landmarks']).reshape(21, 3)
                label = int(data['label'])
                return landmarks, label
    except:
        pass
    return None, None


def preprocess_landmarks(landmarks):
    """
    Preprocess landmarks for model input
    
    Args:
        landmarks: Raw landmark coordinates
        
    Returns:
        Preprocessed landmarks
    """
    if len(landmarks.shape) == 2:
        landmarks = landmarks.reshape(1, *landmarks.shape)
    
    processed = []
    
    for landmark_set in landmarks:
        if landmark_set.shape != (21, 3):
            landmark_set = landmark_set.reshape(21, 3)
        
        wrist = landmark_set[0].copy()
        
        normalized = landmark_set - wrist
        
        scale = np.max(np.abs(normalized))
        if scale > 0:
            normalized = normalized / scale
        
        processed.append(normalized)
    
    return np.array(processed)


def augment_data(landmarks, labels):
    """
    Augment training data
    
    Args:
        landmarks: Landmark data
        labels: Corresponding labels
        
    Returns:
        Augmented (landmarks, labels)
    """
    if len(landmarks) == 0:
        return landmarks, labels
    
    augmented_landmarks = []
    augmented_labels = []
    
    for i in range(len(landmarks)):
        landmark = landmarks[i]
        label = labels[i]
        
        augmented_landmarks.append(landmark)
        augmented_labels.append(label)
        
        if len(landmark.shape) == 2:
            landmark_2d = landmark
        else:
            landmark_2d = landmark.reshape(21, 3)
        
        aug_rotated = _rotate_landmarks(landmark_2d)
        augmented_landmarks.append(aug_rotated)
        augmented_labels.append(label)
        
        aug_scaled = _scale_landmarks(landmark_2d)
        augmented_landmarks.append(aug_scaled)
        augmented_labels.append(label)
        
        aug_noise = _add_noise(landmark_2d)
        augmented_landmarks.append(aug_noise)
        augmented_labels.append(label)
    
    all_landmarks = np.array(augmented_landmarks)
    all_labels = np.array(augmented_labels)
    
    return all_landmarks, all_labels


def _rotate_landmarks(landmarks, angle_range=15):
    """Rotate landmarks around z-axis"""
    angle = np.random.uniform(-angle_range, angle_range) * np.pi / 180
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    rotated = landmarks @ rotation_matrix.T
    return rotated


def _scale_landmarks(landmarks, scale_range=(0.9, 1.1)):
    """Scale landmarks"""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return landmarks * scale


def _add_noise(landmarks, noise_level=0.01):
    """Add random noise to landmarks"""
    noise = np.random.normal(0, noise_level, landmarks.shape)
    return landmarks + noise


def save_landmarks(landmarks, labels, filepath):
    """
    Save landmarks and labels to file
    
    Args:
        landmarks: Landmark data
        labels: Corresponding labels
        filepath: Path to save file
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez(filepath, X=landmarks, y=labels)


def load_landmarks(filepath):
    """
    Load landmarks and labels from file
    
    Args:
        filepath: Path to data file
        
    Returns:
        Tuple of (landmarks, labels)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = np.load(filepath)
    
    if 'X' in data and 'y' in data:
        landmarks = data['X']
        labels = data['y']
    elif 'landmarks' in data and 'label' in data:
        landmarks = data['landmarks']
        labels = data['label']
    else:
        raise ValueError("File does not contain expected keys (X/y or landmarks/label)")
    
    return landmarks, labels

