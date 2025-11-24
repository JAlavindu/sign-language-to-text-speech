from collections import deque, Counter

class TemporalSmoother:
    """
    Smooths predictions over time to reduce jitter.
    Uses a sliding window to return the most common prediction.
    """
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.prediction_buffer = deque(maxlen=buffer_size)

    def add_prediction(self, prediction):
        """
        Adds a prediction to the buffer.
        prediction: The predicted class label or index.
        """
        self.prediction_buffer.append(prediction)

    def get_smoothed_prediction(self):
        """
        Returns the most common prediction in the buffer.
        Returns None if buffer is empty.
        """
        if not self.prediction_buffer:
            return None
        
        # Count occurrences of each prediction
        counts = Counter(self.prediction_buffer)
        # Get the most common one
        most_common = counts.most_common(1)[0][0]
        return most_common
