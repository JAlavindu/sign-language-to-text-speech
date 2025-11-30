import time
import numpy as np

class GestureSegmenter:
    """
    Logic to detect when a sign starts and ends based on motion thresholds.
    Implements a state machine: IDLE -> MOTION -> STABILIZING -> STABLE -> COOLDOWN
    
    This is used to determine the best moment to capture a frame for classification,
    avoiding blurry frames during motion and ensuring the user is holding the sign.
    """
    
    def __init__(self, motion_threshold=0.03, stable_duration=0.5, cooldown=1.0):
        """
        Initialize the gesture segmenter.
        
        Args:
            motion_threshold (float): Minimum movement (normalized distance) to be considered 'motion'.
                                      Adjust based on camera resolution and distance.
            stable_duration (float): Time (seconds) the hand must be still to trigger a 'STABLE' state.
            cooldown (float): Time (seconds) to wait after a successful detection before detecting again.
        """
        self.motion_threshold = motion_threshold
        self.stable_duration = stable_duration
        self.cooldown = cooldown
        
        # State variables
        self.state = "IDLE"  # IDLE, MOTION, STABILIZING, STABLE, COOLDOWN
        self.prev_center = None
        self.stable_start_time = 0
        self.last_sign_time = 0
        self.current_velocity = 0.0
        
    def process(self, landmarks):
        """
        Process a new frame of landmarks to update the state machine.
        
        Args:
            landmarks: List of landmark objects (e.g., from MediaPipe) having .x and .y attributes.
                       Can be None if no hand is detected.
                       
        Returns:
            dict: Status dictionary containing:
                - 'state': Current state string
                - 'trigger_prediction': Boolean, True if this frame should be predicted
                - 'velocity': Current calculated velocity
        """
        current_time = time.time()
        trigger = False
        
        # 1. Handle case where no hand is detected
        if landmarks is None:
            self.state = "IDLE"
            self.prev_center = None
            self.stable_start_time = 0
            return {
                'state': self.state,
                'trigger_prediction': False,
                'velocity': 0.0
            }
            
        # 2. Calculate center of the hand (average of all landmarks)
        # Using numpy for efficient calculation
        points = np.array([[lm.x, lm.y] for lm in landmarks])
        center = np.mean(points, axis=0)
        
        # Initialize previous center if this is the first frame
        if self.prev_center is None:
            self.prev_center = center
            self.state = "IDLE"
            return {
                'state': self.state,
                'trigger_prediction': False,
                'velocity': 0.0
            }
            
        # 3. Calculate velocity (Euclidean distance between current and prev center)
        velocity = np.linalg.norm(center - self.prev_center)
        self.current_velocity = velocity
        self.prev_center = center
        
        # 4. State Machine Logic
        
        # If we are in cooldown, check if enough time has passed
        if self.state == "COOLDOWN":
            if current_time - self.last_sign_time > self.cooldown:
                self.state = "IDLE"
            else:
                return {
                    'state': self.state,
                    'trigger_prediction': False,
                    'velocity': velocity
                }

        # Check for motion
        if velocity > self.motion_threshold:
            # Hand is moving significantly
            self.state = "MOTION"
            self.stable_start_time = 0
        else:
            # Hand is relatively still
            if self.state == "MOTION" or self.state == "IDLE":
                # Transition from moving to stabilizing
                self.state = "STABILIZING"
                self.stable_start_time = current_time
                
            elif self.state == "STABILIZING":
                # Check if we've been stable long enough
                if current_time - self.stable_start_time >= self.stable_duration:
                    self.state = "STABLE"
                    trigger = True  # Trigger prediction!
                    self.last_sign_time = current_time
                    
            elif self.state == "STABLE":
                # We just triggered, now go to cooldown to prevent double triggers
                self.state = "COOLDOWN"
                
        return {
            'state': self.state,
            'trigger_prediction': trigger,
            'velocity': velocity
        }

    def reset(self):
        """Reset the segmenter state."""
        self.state = "IDLE"
        self.prev_center = None
        self.stable_start_time = 0
        self.last_sign_time = 0
