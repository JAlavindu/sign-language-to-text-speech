import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

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
        self.scaler = StandardScaler()
        
    def fit(self, data):
        """
        Compute the mean and std to be used for later scaling.
        
        Args:
            data (np.array): Training data of shape (N, num_features)
        """
        self.scaler.fit(data)
        
    def transform(self, data):
        """
        Apply scaling to the data.
        
        Args:
            data (np.array): Raw sensor data.
            
        Returns:
            np.array: Scaled data.
        """
        return self.scaler.transform(data)
        
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
        
        if len(data) < self.window_size:
            return np.array(X), np.array(y)
            
        # Sliding window with step=1
        for i in range(len(data) - self.window_size + 1):
            window = data[i : i + self.window_size]
            X.append(window)
            if labels is not None:
                # Use the label of the last frame in the window
                y.append(labels[i + self.window_size - 1])
                
        return np.array(X), np.array(y) if labels is not None else np.array(X)

    def save(self, path):
        """Save the preprocessor state."""
        joblib.dump(self.scaler, path)
        
    def load(self, path):
        """Load the preprocessor state."""
        self.scaler = joblib.load(path)
