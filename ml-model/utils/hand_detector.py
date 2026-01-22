import cv2
import mediapipe as mp
# Explicitly import solutions to workaround potential import issues
try:
    import mediapipe.python.solutions
except ImportError:
    pass
import numpy as np

class HandDetector:
    """
    Wrapper for MediaPipe Hands to detect and crop hands from video frames.
    """
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        Initialize the MediaPipe Hands model.
        
        Args:
            max_num_hands (int): Maximum number of hands to detect.
            min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for detection.
            min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for tracking.
        """
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
        except AttributeError:
            import mediapipe.python.solutions as solutions
            self.mp_hands = solutions.hands
            self.mp_draw = solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame, padding=20):
        """
        Process a frame to detect hands.
        
        Args:
            frame (np.array): Input image (BGR).
            padding (int): Padding around the detected hand in pixels.
            
        Returns:
            tuple: (landmarks, bbox)
                - landmarks: MediaPipe landmarks object or None if no hand detected.
                - bbox: Tuple (x, y, w, h) of the bounding box or None.
        """
        h, w, c = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # We only take the first hand detected since max_num_hands=1
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Calculate bounding box
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y
            
            # Add padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            
            bbox = (x_min, y_min, bbox_w, bbox_h)
            
            return hand_landmarks, bbox
            
        return None, None

    def crop_hand(self, frame, bbox, target_size=(224, 224)):
        """
        Crop the hand from the frame and resize it.
        
        Args:
            frame (np.array): Input image.
            bbox (tuple): Bounding box (x, y, w, h).
            target_size (tuple): Target size (width, height).
            
        Returns:
            np.array: Cropped and resized hand image.
        """
        if bbox is None:
            return None
            
        x, y, w, h = bbox
        
        # Ensure bbox is valid
        if w <= 0 or h <= 0:
            return None
            
        # Crop
        hand_img = frame[y:y+h, x:x+w]
        
        # Check if crop is empty (can happen if bbox is completely outside, though clamping prevents most)
        if hand_img.size == 0:
            return None
            
        # Resize
        try:
            hand_img_resized = cv2.resize(hand_img, target_size)
            return hand_img_resized
        except Exception as e:
            print(f"Error resizing hand image: {e}")
            return None

    def draw_landmarks(self, frame, landmarks):
        """
        Draw hand landmarks on the frame.
        
        Args:
            frame (np.array): Input image.
            landmarks: MediaPipe landmarks object.
        """
        if landmarks:
            self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
