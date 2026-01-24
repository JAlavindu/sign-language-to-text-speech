import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Define connections manually in case solutions is missing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

class LandmarkListWrapper:
    def __init__(self, landmarks):
        self.landmark = landmarks

class HandDetector:
    """
    Wrapper for MediaPipe Hands to detect and crop hands from video frames.
    """
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        Initialize the MediaPipe Hands model.
        """
        self.use_tasks = False
        self.handedness = "Unknown"
        self.mp_hands = None
        self.mp_draw = None
        self.landmarker = None

        try:
            # Try legacy solutions API
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        except (AttributeError, ImportError):
            print("Warning: mediapipe.solutions not found. Falling back to mediapipe.tasks.")
            self.use_tasks = True
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                # Resolving absolute path to the model file
                model_path = os.path.join(os.getcwd(), 'ml-model', 'models', 'hand_landmarker.task')
                # If running from ml-model folder, adjust
                if not os.path.exists(model_path):
                     model_path = os.path.join(os.getcwd(), 'models', 'hand_landmarker.task')

                if not os.path.exists(model_path):
                     raise FileNotFoundError(f"Model file not found at: {model_path}")

                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                    running_mode=vision.RunningMode.VIDEO 
                )
                self.landmarker = vision.HandLandmarker.create_from_options(options)
                self.timestamp_ms = 0
            except Exception as e:
                print(f"Error initializing MediaPipe Tasks: {e}")
                print("Please ensure 'models/hand_landmarker.task' exists.")
                raise e

    def process(self, frame, padding=20):
        """
        Process a frame to detect hands.
        """
        h, w, c = frame.shape
        hand_landmarks = None
        
        if self.use_tasks:
            # Task API
            # Convert to mp.Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Update timestamp
            self.timestamp_ms = int(time.time() * 1000)
            
            try:
                result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
                
                if result.hand_landmarks:
                    # Wrap the first hand's landmarks
                    raw_landmarks = result.hand_landmarks[0]
                    hand_landmarks = LandmarkListWrapper(raw_landmarks)
            except Exception as e:
                pass
                
        else:
            # Legacy API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Capture handedness
            if results.multi_handedness:
                self.handedness = results.multi_handedness[0].classification[0].label
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

        if hand_landmarks:
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
        """
        if bbox is None:
            return None
            
        x, y, w, h = bbox
        
        # Ensure bbox is valid
        if w <= 0 or h <= 0:
            return None
            
        # Crop
        hand_img = frame[y:y+h, x:x+w]
        
        # Check if crop is empty
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
        """
        if not landmarks:
            return

        h, w, c = frame.shape

        if self.use_tasks or not self.mp_draw:
            # Manual drawing
            for connection in HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                lm1 = landmarks.landmark[start_idx]
                lm2 = landmarks.landmark[end_idx]
                
                x1, y1 = int(lm1.x * w), int(lm1.y * h)
                x2, y2 = int(lm2.x * w), int(lm2.y * h)
                
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                
            for lm in landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        else:
            self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
