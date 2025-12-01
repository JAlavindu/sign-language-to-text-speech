import numpy as np

class AdaptiveFusion:
    """
    Combines predictions from Camera and Sensor models using adaptive weighting.
    """
    def __init__(self, initial_cam_weight=0.6, initial_sens_weight=0.4):
        self.cam_weight = initial_cam_weight
        self.sens_weight = initial_sens_weight
        
    def fuse(self, cam_probs, sens_probs):
        """
        Fuse probabilities from both models.
        
        Args:
            cam_probs (np.array): Probability vector from camera model (shape: [num_classes])
            sens_probs (np.array): Probability vector from sensor model (shape: [num_classes])
            
        Returns:
            tuple: (fused_probs, final_prediction_index, confidence)
        """
        # Handle cases where one model might be missing (e.g., camera occlusion or sensor disconnect)
        if cam_probs is None and sens_probs is None:
            return None, None, 0.0
            
        if cam_probs is None:
            return sens_probs, np.argmax(sens_probs), np.max(sens_probs)
            
        if sens_probs is None:
            return cam_probs, np.argmax(cam_probs), np.max(cam_probs)
            
        # 1. Calculate dynamic weights based on confidence
        cam_conf = np.max(cam_probs)
        sens_conf = np.max(sens_probs)
        
        w_c = self.cam_weight
        w_s = self.sens_weight
        
        # Boost the more confident model
        if cam_conf > 0.9 and sens_conf < 0.6:
            w_c = 0.8
            w_s = 0.2
        elif sens_conf > 0.9 and cam_conf < 0.6:
            w_c = 0.2
            w_s = 0.8
            
        # Normalize weights
        total = w_c + w_s
        w_c /= total
        w_s /= total
        
        # 2. Weighted Average
        fused_probs = (cam_probs * w_c) + (sens_probs * w_s)
        
        # 3. Get Result
        final_pred = np.argmax(fused_probs)
        final_conf = np.max(fused_probs)
        
        return fused_probs, final_pred, final_conf
