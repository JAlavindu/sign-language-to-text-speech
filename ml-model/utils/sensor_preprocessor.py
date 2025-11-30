import numpy as np
import joblib
import os

class SensorPreprocessor:
    """
    Prepares sensor data for training and inference.
    Handles normalization, scaling, and sequence creation.
    """
    
    def __init__(self, window_size=50, num_features=16):
        """
        Initialize the preprocessor.
        
        Args:
            window_size (int): Number of time steps in each sequence.
            num_features (int): Number of sensor features (excluding timestamp/label).
        """
        self.window_size = window_size
        self.num_features = num_features
        self.scaler = None
        
    def fit(self, data):
        """
        Compute the mean and std to be used for later scaling.
        
        Args:
            data (np.array): Training data of shape (N, num_features)
        """
        # TODO: Implement fitting logic (e.g., using StandardScaler)
        pass
        
    def transform(self, data):
        """
        Apply scaling to the data.
        
        Args:
            data (np.array): Raw sensor data.
            
        Returns:
            np.array: Scaled data.
        """
        # TODO: Implement transformation logic
        # Hint: Flex sensors (0-4095) -> 0-1
        # Hint: IMU data -> Standard Scaler
        return data
        
    def create_sequences(self, data, labels=None):
        """
        Convert linear data into sequences for LSTM/CNN.
        
        Args:
            data (np.array): Scaled data of shape (N, num_features).
            labels (np.array, optional): Corresponding labels.
            
        Returns:
            X (np.array): Sequences of shape (num_sequences, window_size, num_features)
            y (np.array): Labels for each sequence (if labels provided)
        """
        X = []
        y = []
        
        # TODO: Implement sliding window logic
        
        return np.array(X), np.array(y) if labels is not None else np.array(X)

    def save(self, path):
        """Save the preprocessor state."""
        # joblib.dump(self.scaler, path)
        pass
        
    def load(self, path):
        """Load the preprocessor state."""
        # self.scaler = joblib.load(path)
        pass
