import asyncio
import cv2
import numpy as np
import tensorflow as tf
import joblib
import os
import sys
import time
from bleak import BleakClient, BleakScanner
from collections import deque

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.hand_detector import HandDetector
from utils.temporal_smoother import TemporalSmoother
from utils.sensor_preprocessor import SensorPreprocessor
from utils.adaptive_fusion import AdaptiveFusion

# Configuration
DEVICE_NAME = "ASL_Glove_001"
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
CHARACTERISTIC_UUID_TX = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# Model Paths
IMG_MODEL_PATH = os.path.join(current_dir, "models", "asl_model_final.h5")
SENS_MODEL_PATH = os.path.join(current_dir, "models", "sensor_model_final.h5")
SENS_SCALER_PATH = os.path.join(current_dir, "models", "sensor_scaler.pkl")
SENS_LABELS_PATH = os.path.join(current_dir, "models", "sensor_labels.pkl")
IMG_LABELS_PATH = os.path.join(current_dir, "datasets", "processed", "class_mapping.json")

class MultimodalSystem:
    def __init__(self):
        print("Initializing Multimodal System...")
        
        # 1. Load Models
        self.load_models()
        
        # 2. Initialize Components
        self.detector = HandDetector(max_num_hands=1, min_detection_confidence=0.7)
        self.fusion = AdaptiveFusion()
        self.smoother = TemporalSmoother(buffer_size=8)
        
        # 3. Sensor State
        self.sensor_buffer = []
        self.sensor_window_size = 50
        self.sensor_features = 16
        self.latest_sensor_probs = None
        self.ble_connected = False
        
        # 4. Camera State
        self.cap = None
        self.latest_cam_probs = None
        
    def load_models(self):
        try:
            # Camera Model
            print("Loading Camera Model...")
            self.img_model = tf.keras.models.load_model(IMG_MODEL_PATH)
            
            # Sensor Model
            print("Loading Sensor Model...")
            self.sens_model = tf.keras.models.load_model(SENS_MODEL_PATH)
            self.sens_scaler = joblib.load(SENS_SCALER_PATH)
            self.sens_preprocessor = SensorPreprocessor(window_size=50, num_features=16)
            self.sens_preprocessor.scaler = self.sens_scaler
            self.sens_encoder = joblib.load(SENS_LABELS_PATH)
            
            # Load Labels
            import json
            with open(IMG_LABELS_PATH, 'r') as f:
                mapping = json.load(f)
                self.labels = {int(k): v for k, v in mapping.get('idx_to_class', mapping).items()}
                
            print("✓ All models loaded.")
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)

    def process_camera(self, frame):
        """Run camera inference."""
        landmarks, bbox = self.detector.process(frame)
        
        if bbox:
            # Crop and Preprocess
            hand_img = self.detector.crop_hand(frame, bbox)
            if hand_img is not None:
                # Normalize (assuming model trained on 0-1 or -1 to 1)
                # Check your training script! Usually MobileNet expects [-1, 1]
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(hand_img.astype(np.float32))
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                probs = self.img_model.predict(img_array, verbose=0)[0]
                self.latest_cam_probs = probs
                
                # Draw
                self.detector.draw_landmarks(frame, landmarks)
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                return True
        
        self.latest_cam_probs = None
        return False

    def process_sensor_packet(self, data):
        """Handle BLE packet."""
        import struct
        try:
            if len(data) != 43: return
            unpacked = struct.unpack('<I5H3f3f5B', data)
            features = list(unpacked[1:17]) # Skip timestamp
            
            self.sensor_buffer.append(features)
            if len(self.sensor_buffer) > self.sensor_window_size:
                self.sensor_buffer.pop(0)
                
            if len(self.sensor_buffer) == self.sensor_window_size:
                # Inference
                arr = np.array(self.sensor_buffer)
                scaled = self.sens_preprocessor.transform(arr)
                input_data = scaled.reshape(1, self.sensor_window_size, self.sensor_features)
                self.latest_sensor_probs = self.sens_model.predict(input_data, verbose=0)[0]
                
        except Exception as e:
            print(f"Sensor Error: {e}")

    async def run(self):
        # Start Camera
        self.cap = cv2.VideoCapture(0)
        
        # Start BLE
        print(f"Scanning for {DEVICE_NAME}...")
        device = await BleakScanner.find_device_by_name(DEVICE_NAME)
        
        if device:
            print(f"Connecting to {DEVICE_NAME}...")
            async with BleakClient(device) as client:
                self.ble_connected = True
                await client.start_notify(CHARACTERISTIC_UUID_TX, lambda s, d: self.process_sensor_packet(d))
                print("✓ BLE Connected. Starting Fusion Loop...")
                
                await self.main_loop()
        else:
            print("⚠ Glove not found. Running in Camera-Only mode.")
            self.ble_connected = False
            await self.main_loop()

    async def main_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # 1. Camera Inference
            hand_detected = self.process_camera(frame)
            
            # 2. Fusion
            fused_probs, pred_idx, conf = self.fusion.fuse(self.latest_cam_probs, self.latest_sensor_probs)
            
            # 3. Smoothing
            final_label = "?"
            if pred_idx is not None:
                self.smoother.add_prediction(pred_idx)
                smoothed_idx = self.smoother.get_smoothed_prediction()
                if smoothed_idx is not None:
                    final_label = self.labels.get(smoothed_idx, "?")
            
            # 4. Display
            # Info Panel
            cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)
            
            # Result
            color = (0, 255, 0) if conf > 0.8 else (0, 255, 255)
            cv2.putText(frame, f"Sign: {final_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(frame, f"Conf: {conf*100:.1f}%", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Status
            cam_status = "CAM: ON" if hand_detected else "CAM: SEARCH"
            sens_status = "SENS: ON" if self.latest_sensor_probs is not None else "SENS: WAIT"
            cv2.putText(frame, f"{cam_status} | {sens_status}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Multimodal Fusion System", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Yield to asyncio loop (important for BLE!)
            await asyncio.sleep(0.01)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = MultimodalSystem()
    try:
        asyncio.run(system.run())
    except KeyboardInterrupt:
        pass
