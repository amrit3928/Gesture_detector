"""
Gesture Classification Module
Trains and uses a model to classify hand gestures from landmarks

This module implements a neural network model for classifying hand gestures based
on the 21 landmark points extracted by the HandDetector. It uses TensorFlow/Keras
to build, train, and use a deep learning model for gesture recognition.

Key functionality:
- Build neural network architecture for gesture classification
- Train model on collected landmark data
- Predict gestures from new landmark data
- Save and load trained models
- Manage gesture names for display purposes
"""

import numpy as np
import tensorflow as tf
import keras
from typing import List, Tuple, Optional
import os
import config


class GestureClassifier:
    """
    Neural network model for classifying hand gestures from landmarks
    """
    
    def __init__(self, num_gestures: int = config.NUM_GESTURES):
        """
        Initialize the gesture classifier
        
        Args:
            num_gestures: Number of gesture classes to recognize
        """
        self.num_gestures = num_gestures
        self.model = None
        self.gesture_names = []
        
    def build_model(self, input_shape: Tuple[int, int] = (21, 3)) -> keras.Model:
        """
        Build the neural network model for gesture classification

        Args:
            input_shape: Shape of input landmarks (num_landmarks, coordinates)

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Flatten(),

            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            keras.layers.Dense(self.num_gestures, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = config.EPOCHS,
              batch_size: int = config.BATCH_SIZE) -> dict:
        """
        Train the gesture classification model

        Args:
            X_train: Training landmark data
            y_train: Training labels (one-hot encoded)
            X_val: Validation landmark data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history.history
    
    def predict(self, landmarks: np.ndarray) -> Tuple[int, float]:
        """
        Predict gesture from landmarks

        Args:
            landmarks: Hand landmark coordinates

        Returns:
            Tuple of (gesture_id, confidence)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")

        if len(landmarks.shape) == 2:
            landmarks = np.expand_dims(landmarks, axis=0)

        predictions = self.model.predict(landmarks, verbose=0)
        gesture_id = np.argmax(predictions[0])
        confidence = float(predictions[0][gesture_id])

        return int(gesture_id), confidence
    
    def load_model(self, model_path: str):
        """
        Load a saved model

        Args:
            model_path: Path to the saved model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def save_model(self, model_path: str):
        """
        Save the current model

        Args:
            model_path: Path where to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Please train or build a model first.")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def set_gesture_names(self, names: List[str]):
        """
        Set the list of gesture names for display
        
        Args:
            names: List of gesture names corresponding to gesture IDs
        """
        self.gesture_names = names
    
    def get_gesture_name(self, gesture_id: int) -> str:
        """
        Get the name of a gesture by ID
        
        Args:
            gesture_id: Gesture ID
            
        Returns:
            Gesture name or "Unknown" if not found
        """
        if gesture_id < len(self.gesture_names) and len(self.gesture_names) > 0:
            return self.gesture_names[gesture_id]
        
        default_names = {
            0: "One",
            1: "Peace",
            2: "Fist",
            3: "Call",
            4: "OK",
            5: "Like",
            6: "Point",
            7: "Rock",
            8: "Three Gun",
            9: "Four"
        }
        
        return default_names.get(gesture_id, f"Gesture_{gesture_id}")

